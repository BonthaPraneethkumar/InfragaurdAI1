from __future__ import annotations

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
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v4")
SARVAM_CHAT_MODEL = os.getenv("SARVAM_CHAT_MODEL", "sarvam-105b")
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


def privacy_redact(data: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Unsupported or corrupted image.")

    H0, W0 = img.shape[:2]
    if max(H0, W0) > 1800:
        s = 1800 / max(H0, W0)
        img = cv2.resize(img, (int(W0*s), int(H0*s)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    counts = {"faces": 0, "people": 0, "license_plates": 0, "qr_codes": 0}

    # Faces
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(32, 32))
    counts["faces"] = int(len(faces))
    for x, y, w, h in faces:
        blur_box(img, int(x), int(y), int(w), int(h), 10)

    # Full-body humans (conservative; useful when face is turned away).
    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        rects, weights = hog.detectMultiScale(img, winStride=(8,8), padding=(8,8), scale=1.06)
        for (x, y, w, h), wt in zip(rects, np.array(weights).flatten()):
            if float(wt) >= 0.65:
                counts["people"] += 1
                blur_box(img, int(x), int(y), int(w), int(h), 5)
    except Exception:
        pass

    # License plates. Haar cascade is heuristic; production needs stronger PII detection.
    try:
        plate_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
        )
        plates = plate_cascade.detectMultiScale(gray, 1.08, 4, minSize=(45, 15))
        counts["license_plates"] = int(len(plates))
        for x, y, w, h in plates:
            blur_box(img, int(x), int(y), int(w), int(h), 6)
    except Exception:
        pass

    # QR codes
    try:
        qr = cv2.QRCodeDetector()
        ok, decoded, points, _ = qr.detectAndDecodeMulti(img)
        if ok and points is not None:
            counts["qr_codes"] = len(points)
            for pts in points:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x, y = int(min(xs)), int(min(ys))
                w, h = int(max(xs)-x), int(max(ys)-y)
                blur_box(img, x, y, w, h, 8)
        else:
            text, pts, _ = qr.detectAndDecode(img)
            if pts is not None:
                counts["qr_codes"] = 1
                pts = np.array(pts).reshape(-1, 2)
                x, y = int(pts[:,0].min()), int(pts[:,1].min())
                w, h = int(pts[:,0].max()-x), int(pts[:,1].max()-y)
                blur_box(img, x, y, w, h, 8)
    except Exception:
        pass

    counts["total_redactions"] = sum(counts.values())
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


async def sarvam_analyze(text: str, location: str) -> dict[str, Any]:
    prompt = f"""
You are InfraGuard AI's civic infrastructure triage engine.
Citizen speech has already been translated to English.

Complaint: {text}
Location: {location or "GPS captured; readable address unavailable"}

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

Be concise. confidence must be 0 to 1.
"""
    headers = {"api-subscription-key": SARVAM_KEY, "Content-Type": "application/json"}
    payload = {
        "model": SARVAM_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": "Return only structured civic-triage JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 450
    }
    async with httpx.AsyncClient(timeout=50) as client:
        r = await client.post("https://api.sarvam.ai/v1/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return sanitize_ai(parse_json_from_text(content), SARVAM_CHAT_MODEL)


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
    text: str = Field(min_length=3, max_length=3000)
    location_label: str = ""


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
        "azure_maps": bool(AZURE_MAPS_KEY),
        "privacy": "Raw evidence is processed in memory. Only the redacted JPEG is persisted. Citizen photos are not sent to Sarvam."
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
    return {
        "evidence_token": filename,
        "redacted_url": f"/uploads/evidence/{filename}",
        "redactions": counts,
        "message": "Privacy processing complete. Only the redacted image was stored."
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
        raise HTTPException(503, "Sarvam API key is not configured.")
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "No audio received.")
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(413, "Audio is too large.")

    headers = {"api-subscription-key": SARVAM_KEY}
    data = {
        "model": SARVAM_STT_MODEL,
        "mode": "translate",
        "language_code": "unknown"
    }
    files = {
        "file": (
            file.filename or "citizen.webm",
            raw,
            file.content_type or "audio/webm"
        )
    }
    async with httpx.AsyncClient(timeout=70) as client:
        r = await client.post(
            "https://api.sarvam.ai/speech-to-text",
            headers=headers,
            data=data,
            files=files
        )
        r.raise_for_status()
        result = r.json()
    return {
        "english_text": (result.get("transcript") or "").strip(),
        "detected_language": result.get("language_code"),
        "language_probability": result.get("language_probability"),
        "model": SARVAM_STT_MODEL
    }


@app.post("/api/analyze")
async def analyze(body: AnalyzeBody):
    if sarvam_ready():
        try:
            return {"analysis": await sarvam_analyze(body.text, body.location_label), "live_ai": True}
        except Exception as e:
            if not DEMO_FALLBACK:
                raise HTTPException(502, f"Sarvam analysis failed: {e}")
    return {"analysis": fallback_analysis(body.text), "live_ai": False}


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
