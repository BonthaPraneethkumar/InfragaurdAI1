from __future__ import annotations

import base64
import json
import math
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"
DATA = BASE / "data"
UPLOADS = BASE / "uploads"
EVIDENCE = UPLOADS / "evidence"
RESOLUTION = UPLOADS / "resolution"
DB = DATA / "infraguard.db"

for d in (DATA, EVIDENCE, RESOLUTION):
    d.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE / ".env")

SARVAM_KEY = os.getenv("SARVAM_API_KEY", "").strip()
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
SARVAM_CHAT_MODEL = os.getenv("SARVAM_CHAT_MODEL", "sarvam-105b")
SARVAM_VISION_MODEL = os.getenv("SARVAM_VISION_MODEL", "gemma4")
SARVAM_VISION_ENABLED = os.getenv("SARVAM_VISION_ENABLED", "true").lower() == "true"
PRIVACY_STRICT = os.getenv("PRIVACY_STRICT", "true").lower() == "true"
AZURE_MAPS_KEY = os.getenv("AZURE_MAPS_KEY", "").strip()
DEMO_FALLBACK = os.getenv("APP_DEMO_FALLBACK", "true").lower() == "true"

app = FastAPI(title="InfraGuard AI", version="2.0.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOADS), name="uploads")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    c = conn()
    c.execute("""
    CREATE TABLE IF NOT EXISTS reports(
      id TEXT PRIMARY KEY,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      issue_type TEXT NOT NULL,
      severity TEXT NOT NULL,
      department TEXT NOT NULL,
      summary TEXT NOT NULL,
      reason TEXT DEFAULT '',
      confidence REAL DEFAULT 0,
      description TEXT NOT NULL,
      transcript TEXT DEFAULT '',
      location_label TEXT DEFAULT '',
      latitude REAL,
      longitude REAL,
      evidence_path TEXT DEFAULT '',
      resolution_path TEXT DEFAULT '',
      status TEXT NOT NULL DEFAULT 'Registered',
      officer_notes TEXT DEFAULT '',
      ai_source TEXT DEFAULT '',
      duplicate_count INTEGER DEFAULT 1,
      priority_score INTEGER DEFAULT 0,
      redactions_json TEXT DEFAULT '{}'
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS notifications(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      report_id TEXT,
      message TEXT NOT NULL,
      created_at TEXT NOT NULL,
      is_read INTEGER DEFAULT 0
    )
    """)
    c.commit()
    c.close()


@app.on_event("startup")
def startup():
    init_db()


def sarvam_ready() -> bool:
    return bool(SARVAM_KEY and SARVAM_KEY.lower() not in {"your_key_here", "paste_key_here"})


def blur_box(img: np.ndarray, x: int, y: int, w: int, h: int, pad: int = 8):
    H, W = img.shape[:2]
    x1, y1 = max(0, x-pad), max(0, y-pad)
    x2, y2 = min(W, x+w+pad), min(H, y+h+pad)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    kx = max(31, (roi.shape[1]//7)|1)
    ky = max(31, (roi.shape[0]//7)|1)
    kx = min(kx, 151 if 151 % 2 else 149)
    ky = min(ky, 151 if 151 % 2 else 149)
    img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kx, ky), 0)


def _merge_boxes(boxes: list[tuple[int, int, int, int]], iou_threshold: float = 0.25) -> list[tuple[int, int, int, int]]:
    """Merge overlapping detector boxes so we redact one larger region."""
    if not boxes:
        return []
    boxes = [list(map(int, b)) for b in boxes]
    merged: list[list[int]] = []
    for box in boxes:
        x, y, w, h = box
        x2, y2 = x + w, y + h
        combined = False
        for m in merged:
            mx, my, mw, mh = m
            mx2, my2 = mx + mw, my + mh
            ix1, iy1 = max(x, mx), max(y, my)
            ix2, iy2 = min(x2, mx2), min(y2, my2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            union = w * h + mw * mh - inter
            if union and inter / union >= iou_threshold:
                nx1, ny1 = min(x, mx), min(y, my)
                nx2, ny2 = max(x2, mx2), max(y2, my2)
                m[:] = [nx1, ny1, nx2 - nx1, ny2 - ny1]
                combined = True
                break
        if not combined:
            merged.append(box)
    return [tuple(x) for x in merged]


def _detect_privacy_regions(img: np.ndarray) -> dict[str, list[tuple[int, int, int, int]]]:
    """Privacy-first detector set. It intentionally favors over-redaction."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = img.shape[:2]
    regions: dict[str, list[tuple[int, int, int, int]]] = {
        "faces": [], "people": [], "license_plates": [], "qr_codes": [], "text_like": []
    }

    # Multiple face cascades catch frontal, alternate, and profile views.
    cascade_names = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_profileface.xml",
    ]
    for name in cascade_names:
        try:
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + name)
            found = cascade.detectMultiScale(gray, 1.08, 4, minSize=(28, 28))
            regions["faces"].extend([tuple(map(int, r)) for r in found])
            # Profile detector can miss the mirrored direction.
            if "profile" in name:
                flipped = cv2.flip(gray, 1)
                found2 = cascade.detectMultiScale(flipped, 1.08, 4, minSize=(28, 28))
                for x, y, w, h in found2:
                    regions["faces"].append((W - int(x) - int(w), int(y), int(w), int(h)))
        except Exception:
            pass
    regions["faces"] = _merge_boxes(regions["faces"])

    # Full-body human detector. This is conservative by design.
    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        rects, weights = hog.detectMultiScale(img, winStride=(8, 8), padding=(12, 12), scale=1.05)
        for (x, y, w, h), wt in zip(rects, np.array(weights).flatten()):
            if float(wt) >= 0.35:
                regions["people"].append((int(x), int(y), int(w), int(h)))
        regions["people"] = _merge_boxes(regions["people"])
    except Exception:
        pass

    # License plate heuristic cascade.
    try:
        plate = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
        found = plate.detectMultiScale(gray, 1.05, 3, minSize=(36, 12))
        regions["license_plates"] = [tuple(map(int, r)) for r in found]
    except Exception:
        pass

    # QR codes.
    try:
        qr = cv2.QRCodeDetector()
        ok, _, points, _ = qr.detectAndDecodeMulti(img)
        if ok and points is not None:
            for pts in points:
                pts = np.array(pts).reshape(-1, 2)
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                regions["qr_codes"].append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
        else:
            _, pts, _ = qr.detectAndDecode(img)
            if pts is not None:
                pts = np.array(pts).reshape(-1, 2)
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                regions["qr_codes"].append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
    except Exception:
        pass

    # Strict mode: detect text-like clusters with MSER and blur them. This intentionally
    # over-redacts signs/documents to reduce the chance of exposing phone/ID/address text.
    if PRIVACY_STRICT:
        try:
            mser = cv2.MSER_create(_min_area=35, _max_area=max(2000, int(H*W*0.025)))
        except TypeError:
            mser = cv2.MSER_create()
            mser.setMinArea(35)
            mser.setMaxArea(max(2000, int(H*W*0.025)))
        try:
            raw_regions, _ = mser.detectRegions(gray)
            candidates = []
            for pts in raw_regions:
                x, y, w, h = cv2.boundingRect(pts.reshape(-1, 1, 2))
                area = w*h
                if h < 10 or w < 10 or area > H*W*0.08:
                    continue
                aspect = w / max(h, 1)
                if 0.15 <= aspect <= 12 and h <= H*0.22:
                    candidates.append((x, y, w, h))
            # Merge nearby text candidates into larger blocks.
            candidates = sorted(candidates, key=lambda b: (b[1], b[0]))[:500]
            expanded = []
            for x, y, w, h in candidates:
                pad_x, pad_y = int(w*0.35)+4, int(h*0.35)+4
                expanded.append((max(0,x-pad_x), max(0,y-pad_y), min(W-x+pad_x,w+2*pad_x), min(H-y+pad_y,h+2*pad_y)))
            regions["text_like"] = _merge_boxes(expanded, 0.08)[:80]
        except Exception:
            pass

    return regions


def privacy_redact(data: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    """Strict privacy pipeline.

    Important: no computer-vision detector can honestly guarantee 100% PII detection.
    This pipeline therefore combines aggressive redaction with a fail-closed re-scan:
    if a person/face/plate/QR remains detectable after redaction, the image is marked
    unsafe for AI and the citizen is asked to retake it.
    """
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Unsupported or corrupted image.")

    H0, W0 = img.shape[:2]
    if max(H0, W0) > 1800:
        scale = 1800 / max(H0, W0)
        img = cv2.resize(img, (int(W0*scale), int(H0*scale)))

    regions = _detect_privacy_regions(img)
    counts = {k: len(v) for k, v in regions.items()}

    # Pixelation + heavy blur gives stronger concealment than light Gaussian blur alone.
    for category, boxes in regions.items():
        for x, y, w, h in boxes:
            H, W = img.shape[:2]
            pad = 18 if category in {"faces", "people"} else 10
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(W, x+w+pad), min(H, y+h+pad)
            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            # Downscale brutally, upscale nearest-neighbor, then blur.
            sw, sh = max(2, roi.shape[1]//24), max(2, roi.shape[0]//24)
            pix = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
            pix = cv2.resize(pix, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
            k = max(31, min(151, ((min(roi.shape[:2])//3)|1)))
            if k % 2 == 0:
                k += 1
            img[y1:y2, x1:x2] = cv2.GaussianBlur(pix, (k, k), 0)

    # Re-scan the protected output. Text-like regions are not used in the fail-closed
    # decision because MSER intentionally finds many harmless texture regions.
    verify = _detect_privacy_regions(img)
    residual = {
        "faces": len(verify["faces"]),
        "people": len(verify["people"]),
        "license_plates": len(verify["license_plates"]),
        "qr_codes": len(verify["qr_codes"]),
    }
    counts["total_redactions"] = sum(counts.values())
    counts["residual_sensitive"] = sum(residual.values())
    counts["safe_for_ai"] = counts["residual_sensitive"] == 0
    counts["verification"] = residual
    return img, counts

def save_jpeg(img: np.ndarray, folder: Path, stem: str) -> str:
    filename = f"{stem}.jpg"
    path = folder / filename
    ok = cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(500, "Could not save processed image.")
    return filename


def normalize_issue(value: str) -> str:
    allowed = {
        "Traffic": "Traffic",
        "Garbage Waste": "Garbage Waste",
        "Streetlight": "Streetlight",
        "Road Pothole": "Road Pothole",
        "Water Leaks": "Water Leaks",
        "Electric Wires": "Electric Wires",
        "Cut down trees on road": "Cut down trees on road",
        "Other": "Other",
    }
    return allowed.get(value, "Other")


DEPT = {
    "Traffic": "Traffic Department",
    "Garbage Waste": "Municipal Sanitation",
    "Streetlight": "Electrical Department",
    "Road Pothole": "Roads Department",
    "Water Leaks": "Water & Utilities",
    "Electric Wires": "Electrical Department",
    "Cut down trees on road": "Municipal / Parks Department",
    "Other": "Municipal Control Room",
}


def fallback_analysis(text: str) -> dict[str, Any]:
    t = text.lower()
    issue = "Other"
    tests = [
        ("Electric Wires", ("wire", "electric cable", "power line", "live wire")),
        ("Road Pothole", ("pothole", "road hole", "road damage", "broken road")),
        ("Water Leaks", ("water leak", "pipe leak", "leaking water", "water leakage")),
        ("Garbage Waste", ("garbage", "waste", "trash", "rubbish", "overflowing bin")),
        ("Streetlight", ("streetlight", "street light", "lamp post", "light not working")),
        ("Traffic", ("traffic", "parking", "parked vehicle", "vehicle blocking")),
        ("Cut down trees on road", ("fallen tree", "cut tree", "tree on road", "tree blocking")),
    ]
    for label, keys in tests:
        if any(k in t for k in keys):
            issue = label
            break

    severity = "Medium"
    if any(k in t for k in ("live wire", "fire", "electrocution", "collapse", "life threat", "severe flooding")):
        severity = "Critical"
    elif any(k in t for k in ("large", "danger", "dangerous", "accident", "blocking", "major", "exposed")):
        severity = "High"
    elif any(k in t for k in ("small", "minor", "slight")):
        severity = "Low"

    return {
        "issue_type": issue,
        "severity": severity,
        "department": DEPT[issue],
        "summary": text.strip()[:220],
        "reason": "Local fallback classification used for the hackathon demo.",
        "confidence": 0.74,
        "ai_source": "local-fallback"
    }


def parse_json_from_text(text: str) -> dict[str, Any]:
    t = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I)
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, flags=re.S)
        if not m:
            raise ValueError("No JSON object in AI response")
        return json.loads(m.group(0))


def sanitize_ai(a: dict[str, Any], source: str) -> dict[str, Any]:
    issue = normalize_issue(str(a.get("issue_type", "Other")).strip())
    severity = str(a.get("severity", "Medium")).strip().title()
    if severity not in {"Low", "Medium", "High", "Critical"}:
        severity = "Medium"
    department = str(a.get("department") or DEPT[issue]).strip()
    try:
        conf = max(0.0, min(1.0, float(a.get("confidence", .85))))
    except Exception:
        conf = .85
    return {
        "issue_type": issue,
        "severity": severity,
        "department": department or DEPT[issue],
        "summary": str(a.get("summary", "")).strip()[:260],
        "reason": str(a.get("reason", "")).strip()[:360],
        "confidence": conf,
        "ai_source": source
    }


async def sarvam_photo_analyze(evidence_path: Path) -> dict[str, Any]:
    """Analyze the privacy-protected photo with Sarvam Gemma 4 when beta access exists.

    The raw citizen photo is never sent. Only the locally privacy-redacted JPEG is used.
    If the account lacks /v2 beta access, the caller receives a structured unavailable
    result and can still combine voice/text with local visual cues.
    """
    if not (sarvam_ready() and SARVAM_VISION_ENABLED):
        return {
            "available": False,
            "issue_type": "Other",
            "severity": "Medium",
            "observations": "Photo vision is not enabled.",
            "confidence": 0.0,
            "source": "vision-disabled",
        }
    try:
        raw = evidence_path.read_bytes()
        if len(raw) > 6 * 1024 * 1024:
            # Reduce encoded request size while preserving enough detail for classification.
            img = cv2.imread(str(evidence_path))
            if img is not None:
                h, w = img.shape[:2]
                scale = min(1.0, 1280 / max(h, w))
                if scale < 1:
                    img = cv2.resize(img, (int(w*scale), int(h*scale)))
                ok, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                if ok:
                    raw = enc.tobytes()
        data_uri = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
        prompt = (
            "You are a civic infrastructure image classifier. Inspect this privacy-redacted street/city photo. "
            "Classify only visible infrastructure evidence. Allowed issue_type values: Traffic, Garbage Waste, "
            "Streetlight, Road Pothole, Water Leaks, Electric Wires, Cut down trees on road, Other. "
            "Allowed severity: Low, Medium, High, Critical. Do not infer private identity. "
            "Return one compact JSON object with issue_type, severity, observations, confidence."
        )
        payload = {
            "model": SARVAM_VISION_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }],
            "temperature": 0.1,
            "max_tokens": 300,
        }
        async with httpx.AsyncClient(timeout=65) as client:
            r = await client.post(
                "https://api.sarvam.ai/v2/chat/completions",
                headers={"api-subscription-key": SARVAM_KEY, "Content-Type": "application/json"},
                json=payload,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
        parsed = parse_json_from_text(content)
        issue = normalize_issue(str(parsed.get("issue_type", "Other")))
        sev = str(parsed.get("severity", "Medium")).title()
        if sev not in {"Low", "Medium", "High", "Critical"}:
            sev = "Medium"
        try:
            conf = max(0.0, min(1.0, float(parsed.get("confidence", 0.75))))
        except Exception:
            conf = 0.75
        return {
            "available": True,
            "issue_type": issue,
            "severity": sev,
            "observations": str(parsed.get("observations", "")).strip()[:450],
            "confidence": conf,
            "source": SARVAM_VISION_MODEL,
        }
    except Exception as exc:
        return {
            "available": False,
            "issue_type": "Other",
            "severity": "Medium",
            "observations": "Photo vision unavailable for this API key; voice/text analysis continues.",
            "confidence": 0.0,
            "source": "vision-fallback",
            "error": str(exc)[:240],
        }


def local_photo_cues(image_path: Path) -> dict[str, Any]:
    """Lightweight visual cues used only when multimodal vision is unavailable.

    These cues do not pretend to identify a pothole with certainty. They provide evidence
    such as vehicle density / line density / dark road damage regions to the text model.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return {"available": False, "observations": "Image could not be read.", "confidence": 0.0, "source": "opencv-cues"}
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    edge_ratio = float(np.count_nonzero(edges)) / max(1, edges.size)
    dark_ratio = float(np.count_nonzero(gray < 55)) / max(1, gray.size)
    bright_ratio = float(np.count_nonzero(gray > 210)) / max(1, gray.size)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=70, minLineLength=max(30, w//10), maxLineGap=12)
    line_count = 0 if lines is None else min(999, len(lines))
    obs = f"Local visual cues: edge-density={edge_ratio:.2f}, dark-region={dark_ratio:.2f}, bright-region={bright_ratio:.2f}, strong-lines={line_count}."
    return {"available": True, "issue_type": "Other", "severity": "Medium", "observations": obs, "confidence": 0.20, "source": "opencv-cues"}


async def sarvam_analyze(text: str, location: str, photo: dict[str, Any] | None = None) -> dict[str, Any]:
    photo = photo or {}
    prompt = f"""
You are InfraGuard AI's civic infrastructure triage engine.
Use BOTH sources when available: (1) the citizen's English complaint and (2) photo observations.
If photo and voice disagree, prefer concrete visible infrastructure evidence but mention uncertainty.
Do not invent evidence that is absent.

Citizen complaint: {text or "No usable voice/text supplied"}
Location: {location or "GPS captured; readable address unavailable"}
Photo analysis source: {photo.get('source', 'none')}
Photo issue suggestion: {photo.get('issue_type', 'Other')}
Photo severity suggestion: {photo.get('severity', 'Medium')}
Photo confidence: {photo.get('confidence', 0)}
Photo observations: {photo.get('observations', 'No photo observations available')}

Return ONLY valid JSON with:
issue_type, severity, department, summary, reason, confidence

issue_type must be exactly one of:
Traffic
Garbage Waste
Streetlight
Road Pothole
Water Leaks
Electric Wires
Cut down trees on road
Other

severity must be exactly: Low, Medium, High, or Critical.

Department rules:
Traffic -> Traffic Department
Garbage Waste -> Municipal Sanitation
Streetlight -> Electrical Department
Road Pothole -> Roads Department
Water Leaks -> Water & Utilities
Electric Wires -> Electrical Department
Cut down trees on road -> Municipal / Parks Department
Other -> Municipal Control Room

Severity:
Critical = immediate serious threat to life/public safety.
High = significant accident/safety risk or major obstruction.
Medium = normal infrastructure service issue.
Low = minor inconvenience.

The summary and reason must explicitly reflect the combined evidence when photo information is available.
confidence must be 0 to 1.
"""
    headers = {"api-subscription-key": SARVAM_KEY, "Content-Type": "application/json"}
    payload = {
        "model": SARVAM_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": "Return only structured civic-triage JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 550
    }
    async with httpx.AsyncClient(timeout=55) as client:
        r = await client.post("https://api.sarvam.ai/v1/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return sanitize_ai(parse_json_from_text(content), f"{SARVAM_CHAT_MODEL}+photo")

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2-lat1)
    dl = math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))


def priority_score(severity: str, duplicates: int) -> int:
    base = {"Low": 20, "Medium": 45, "High": 75, "Critical": 95}.get(severity, 45)
    return min(100, base + min(duplicates-1, 5)*4)


class AnalyzeBody(BaseModel):
    text: str = Field(default="", max_length=3000)
    location_label: str = ""
    evidence_token: str | None = None


class SubmitBody(BaseModel):
    description: str = Field(min_length=3, max_length=3000)
    transcript: str = ""
    location_label: str = ""
    latitude: float | None = None
    longitude: float | None = None
    evidence_token: str
    redactions: dict[str, Any] = {}
    analysis: dict[str, Any]


class StatusBody(BaseModel):
    status: str
    officer_notes: str = ""


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/manifest.webmanifest")
def manifest():
    return FileResponse(STATIC / "manifest.webmanifest", media_type="application/manifest+json")


@app.get("/api/config")
def config():
    return {
        "sarvam": sarvam_ready(),
        "sarvam_stt": SARVAM_STT_MODEL,
        "sarvam_chat": SARVAM_CHAT_MODEL,
        "sarvam_vision": SARVAM_VISION_MODEL if SARVAM_VISION_ENABLED else None,
        "azure_maps": bool(AZURE_MAPS_KEY),
        "privacy_strict": PRIVACY_STRICT,
        "privacy": "Raw evidence is processed in memory. Only the privacy-redacted JPEG is persisted. Strict mode blocks AI use if residual sensitive regions remain detectable."
    }


@app.post("/api/redact-image")
async def redact_image(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "No image received.")
    if len(raw) > 15 * 1024 * 1024:
        raise HTTPException(413, "Photo must be under 15 MB.")

    redacted, counts = privacy_redact(raw)
    token = uuid.uuid4().hex
    filename = save_jpeg(redacted, EVIDENCE, token)
    safe_for_ai = bool(counts.get("safe_for_ai", True))
    return {
        "evidence_token": filename if safe_for_ai else None,
        "redacted_url": f"/uploads/evidence/{filename}",
        "redactions": counts,
        "safe_for_ai": safe_for_ai,
        "message": (
            "Strict privacy verification passed. Only the protected image was stored."
            if safe_for_ai else
            "Strict privacy verification detected residual sensitive content. Retake the photo before AI analysis."
        )
    }


@app.get("/api/reverse-geocode")
async def reverse_geocode(lat: float, lon: float):
    fallback = f"{lat:.6f}, {lon:.6f}"
    if not AZURE_MAPS_KEY:
        return {"label": fallback, "source": "coordinates"}
    try:
        params = {
            "api-version": "2025-01-01",
            "coordinates": f"{lon},{lat}",
            "view": "IN",
            "subscription-key": AZURE_MAPS_KEY
        }
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get("https://atlas.microsoft.com/reverseGeocode", params=params)
            r.raise_for_status()
            data = r.json()
        features = data.get("features", [])
        if features:
            props = features[0].get("properties", {})
            addr = props.get("address", {})
            label = (
                addr.get("formattedAddress")
                or addr.get("addressLine")
                or props.get("name")
                or fallback
            )
            return {"label": label, "source": "azure-maps"}
    except Exception:
        pass
    return {"label": fallback, "source": "coordinates"}


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not sarvam_ready():
        raise HTTPException(503, "Sarvam API key is not configured in the server environment.")
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "No audio received.")
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(413, "Audio is too large.")

    headers = {"api-subscription-key": SARVAM_KEY}
    filename = file.filename or "citizen.wav"
    # The frontend now sends a standard PCM WAV. Keep a safe fallback for older clients.
    content_type = file.content_type or ("audio/wav" if filename.lower().endswith(".wav") else "audio/webm")

    def sarvam_error(response: httpx.Response) -> str:
        try:
            body = response.json()
            if isinstance(body, dict):
                err = body.get("error", body)
                if isinstance(err, dict):
                    return str(err.get("message") or err.get("detail") or err)[:500]
                return str(err)[:500]
        except Exception:
            pass
        return (response.text or f"HTTP {response.status_code}")[:500]

    # Primary path: Saaras v3 officially supports mode=translate, converting supported
    # Indian-language speech directly to English while auto-detecting language.
    primary_error = None
    try:
        async with httpx.AsyncClient(timeout=70) as client:
            r = await client.post(
                "https://api.sarvam.ai/speech-to-text",
                headers=headers,
                data={"model": SARVAM_STT_MODEL, "mode": "translate", "language_code": "unknown"},
                files={"file": (filename, raw, content_type)},
            )
            if r.status_code >= 400:
                raise ValueError(f"Sarvam STT {r.status_code}: {sarvam_error(r)}")
            result = r.json()
        english = (result.get("transcript") or "").strip()
        if not english:
            raise ValueError("Sarvam returned an empty transcript.")
        return {
            "english_text": english,
            "detected_language": result.get("language_code"),
            "language_probability": result.get("language_probability"),
            "model": f"{SARVAM_STT_MODEL}-translate",
        }
    except Exception as exc:
        primary_error = str(exc)

    # Fallback: Saaras v4 transcription, then Mayura auto-detect text translation to English.
    try:
        async with httpx.AsyncClient(timeout=70) as client:
            r = await client.post(
                "https://api.sarvam.ai/speech-to-text",
                headers=headers,
                data={"model": "saaras:v4", "language_code": "unknown"},
                files={"file": (filename, raw, content_type)},
            )
            if r.status_code >= 400:
                raise ValueError(f"Sarvam STT fallback {r.status_code}: {sarvam_error(r)}")
            stt = r.json()
        transcript = (stt.get("transcript") or "").strip()
        if not transcript:
            raise ValueError("Saaras v4 returned an empty transcript.")
        detected = stt.get("language_code") or "auto"

        # English needs no translation. Otherwise translate the detected text to English.
        if detected == "en-IN":
            english = transcript
        else:
            payload = {
                "input": transcript[:1000],
                "source_language_code": "auto",
                "target_language_code": "en-IN",
                "model": "mayura:v1",
            }
            async with httpx.AsyncClient(timeout=45) as client:
                tr = await client.post(
                    "https://api.sarvam.ai/translate",
                    headers={**headers, "Content-Type": "application/json"},
                    json=payload,
                )
                tr.raise_for_status()
                english = (tr.json().get("translated_text") or transcript).strip()
        return {
            "english_text": english,
            "detected_language": detected,
            "language_probability": stt.get("language_probability"),
            "model": "saaras:v4+mayura",
            "warning": f"Primary v3 translate path failed and fallback was used: {primary_error}"[:280],
        }
    except Exception as exc:
        raise HTTPException(502, f"Voice-to-English failed. Primary: {primary_error}; fallback: {exc}")


@app.post("/api/analyze")
async def analyze(body: AnalyzeBody):
    text = body.text.strip()
    photo: dict[str, Any] = {"available": False, "source": "no-photo", "observations": "No photo supplied."}

    if body.evidence_token:
        evidence = EVIDENCE / Path(body.evidence_token).name
        if evidence.exists():
            photo = await sarvam_photo_analyze(evidence)
            if not photo.get("available"):
                local = local_photo_cues(evidence)
                # Preserve the reason the multimodal call was unavailable while still using the photo.
                local["observations"] = f"{local.get('observations','')} Vision note: {photo.get('observations','')}"
                photo = local

    if sarvam_ready():
        try:
            result = await sarvam_analyze(text, body.location_label, photo)
            return {"analysis": result, "photo_analysis": photo, "live_ai": True}
        except Exception as exc:
            if not DEMO_FALLBACK:
                raise HTTPException(502, f"Sarvam combined analysis failed: {exc}")

    # Local fallback still combines text with the fact that a protected photo exists.
    if text:
        result = fallback_analysis(text)
    else:
        result = {
            "issue_type": normalize_issue(str(photo.get("issue_type", "Other"))),
            "severity": str(photo.get("severity", "Medium")),
            "department": DEPT[normalize_issue(str(photo.get("issue_type", "Other")))],
            "summary": str(photo.get("observations", "Photo received; live AI unavailable."))[:220],
            "reason": "Photo-only fallback used because live multimodal AI was unavailable.",
            "confidence": float(photo.get("confidence", 0.2) or 0.2),
            "ai_source": str(photo.get("source", "photo-fallback")),
        }
    return {"analysis": result, "photo_analysis": photo, "live_ai": False}


@app.post("/api/submit")
def submit(body: SubmitBody):
    evidence = EVIDENCE / Path(body.evidence_token).name
    if not evidence.exists():
        raise HTTPException(400, "Evidence has not passed privacy processing.")

    a = sanitize_ai(body.analysis, str(body.analysis.get("ai_source", "unknown")))
    c = conn()

    # Duplicate detection: same AI issue within 200m, open case, created during the last 72h.
    duplicate = None
    if body.latitude is not None and body.longitude is not None:
        since = (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat()
        candidates = c.execute(
            """SELECT * FROM reports
               WHERE issue_type=? AND status!='Resolved' AND created_at>=?
               AND latitude IS NOT NULL AND longitude IS NOT NULL""",
            (a["issue_type"], since)
        ).fetchall()
        for row in candidates:
            if haversine_m(
                body.latitude, body.longitude,
                float(row["latitude"]), float(row["longitude"])
            ) <= 200:
                duplicate = row
                break

    if duplicate:
        new_count = int(duplicate["duplicate_count"]) + 1
        score = priority_score(duplicate["severity"], new_count)
        c.execute(
            "UPDATE reports SET duplicate_count=?, priority_score=?, updated_at=? WHERE id=?",
            (new_count, score, now_iso(), duplicate["id"])
        )
        c.execute(
            "INSERT INTO notifications(report_id,message,created_at) VALUES(?,?,?)",
            (
                duplicate["id"],
                f"Your report matched existing case {duplicate['id']}. Community report count is now {new_count}.",
                now_iso()
            )
        )
        c.commit()
        row = c.execute("SELECT * FROM reports WHERE id=?", (duplicate["id"],)).fetchone()
        c.close()
        return {"report": dict(row), "duplicate": True}

    report_id = "IG-" + datetime.now().strftime("%y%m%d") + "-" + uuid.uuid4().hex[:5].upper()
    ts = now_iso()
    score = priority_score(a["severity"], 1)
    rel = f"uploads/evidence/{evidence.name}"
    c.execute(
        """INSERT INTO reports(
          id,created_at,updated_at,issue_type,severity,department,summary,reason,confidence,
          description,transcript,location_label,latitude,longitude,evidence_path,status,
          ai_source,duplicate_count,priority_score,redactions_json
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            report_id, ts, ts, a["issue_type"], a["severity"], a["department"],
            a["summary"] or body.description[:220], a["reason"], a["confidence"],
            body.description, body.transcript, body.location_label,
            body.latitude, body.longitude, rel, "Registered",
            a["ai_source"], 1, score, json.dumps(body.redactions)
        )
    )
    c.execute(
        "INSERT INTO notifications(report_id,message,created_at) VALUES(?,?,?)",
        (report_id, f"Complaint {report_id} registered and routed to {a['department']}.", ts)
    )
    c.commit()
    row = c.execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
    c.close()
    return {"report": dict(row), "duplicate": False}


@app.get("/api/reports")
def reports():
    c = conn()
    rows = c.execute("SELECT * FROM reports ORDER BY created_at DESC").fetchall()
    c.close()
    out = []
    for r in rows:
        d = dict(r)
        if d.get("evidence_path"): d["evidence_path"] = "/" + d["evidence_path"]
        if d.get("resolution_path"): d["resolution_path"] = "/" + d["resolution_path"]
        out.append(d)
    return {"reports": out}


@app.patch("/api/reports/{report_id}")
def update_report(report_id: str, body: StatusBody):
    allowed = ["Registered", "Assigned", "In Progress", "Resolved"]
    if body.status not in allowed:
        raise HTTPException(400, "Invalid status.")
    c = conn()
    row = c.execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
    if not row:
        c.close()
        raise HTTPException(404, "Report not found.")
    c.execute(
        "UPDATE reports SET status=?, officer_notes=?, updated_at=? WHERE id=?",
        (body.status, body.officer_notes[:2000], now_iso(), report_id)
    )
    c.execute(
        "INSERT INTO notifications(report_id,message,created_at) VALUES(?,?,?)",
        (report_id, f"{report_id} status updated to {body.status}.", now_iso())
    )
    c.commit()
    c.close()
    return {"ok": True}


@app.post("/api/reports/{report_id}/resolution")
async def upload_resolution(report_id: str, file: UploadFile = File(...)):
    raw = await file.read()
    redacted, counts = privacy_redact(raw)
    filename = save_jpeg(redacted, RESOLUTION, f"{report_id}-{uuid.uuid4().hex[:5]}")
    rel = f"uploads/resolution/{filename}"
    c = conn()
    if not c.execute("SELECT id FROM reports WHERE id=?", (report_id,)).fetchone():
        c.close()
        raise HTTPException(404, "Report not found.")
    c.execute(
        "UPDATE reports SET resolution_path=?, updated_at=? WHERE id=?",
        (rel, now_iso(), report_id)
    )
    c.commit()
    c.close()
    return {"ok": True, "url": "/" + rel, "redactions": counts}


@app.get("/api/notifications")
def notifications():
    c = conn()
    rows = c.execute("SELECT * FROM notifications ORDER BY id DESC LIMIT 50").fetchall()
    c.close()
    return {"notifications": [dict(r) for r in rows]}


@app.get("/api/analytics")
def analytics():
    c = conn()
    total = c.execute("SELECT COUNT(*) n FROM reports").fetchone()["n"]
    statuses = {r["status"]: r["n"] for r in c.execute("SELECT status,COUNT(*) n FROM reports GROUP BY status")}
    departments = {r["department"]: r["n"] for r in c.execute("SELECT department,COUNT(*) n FROM reports GROUP BY department")}
    severities = {r["severity"]: r["n"] for r in c.execute("SELECT severity,COUNT(*) n FROM reports GROUP BY severity")}
    top = [dict(r) for r in c.execute("SELECT * FROM reports ORDER BY priority_score DESC, created_at ASC LIMIT 10")]
    c.close()
    return {
        "total": total,
        "statuses": statuses,
        "departments": departments,
        "severities": severities,
        "top_priority": top
    }
