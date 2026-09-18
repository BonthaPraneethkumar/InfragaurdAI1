# InfraGuard AI — Final Hackathon Product

This version preserves the supplied citizen-side home interface content and the supplied
InfraGuard AI logo. It adds the real workflow behind the interface.

## Mandatory workflow implemented

1. Citizen taps **Report an Issue**.
2. Browser requests camera + microphone permission and location permission on first use.
3. Real camera opens (back camera is preferred on mobile).
4. Citizen takes a real photo.
5. GPS location is automatically attached. If an Azure Maps key is configured, GPS is
   reverse-geocoded to a readable location.
6. The raw photo is processed in memory by the InfraGuard backend:
   - faces are blurred
   - detected human figures are blurred
   - detected license plates are blurred
   - QR codes are blurred
   - re-encoding removes EXIF/metadata
   Only the processed/redacted image is persisted.
7. Voice recording starts automatically after photo processing.
8. Sarvam Saaras v4 with `mode=translate` auto-detects a supported Indian language
   and returns English text.
9. Sarvam-105B analyzes the English complaint:
   - issue type
   - severity
   - responsible department
   - summary
   - reason
   - confidence
10. Duplicate detection checks for the same issue within 200 metres during the last 72 hours.
11. A case is created and routed to the officer dashboard.
12. Officer updates Registered -> Assigned -> In Progress -> Resolved.
13. Citizen sees status updates in My Reports / Notifications.
14. Analytics shows workload and severity.

## Important browser rule

A web app CANNOT silently bypass the first camera, microphone, or location permission prompt.
That permission is controlled by the browser/operating system. This product requests all
permissions immediately when the citizen taps Report an Issue. After the citizen grants
location permission once, location can be attached automatically without another prompt
on later reports (subject to browser/site settings).

## Real AI setup

Copy `.env.example` to `.env` and add:

SARVAM_API_KEY=...
AZURE_MAPS_KEY=...   # optional but recommended for readable automatic addresses

Do not put API keys in frontend JavaScript or GitHub.

## Start on Windows

Double-click `START_INFRAGUARD.bat`

or:

python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000

Then open http://127.0.0.1:8000

## Production note

The redaction layer in this one-day MVP is real but heuristic. It detects faces,
human figures, license plates and QR codes. No computer-vision system should be claimed
to guarantee detection of every possible sensitive item. Production deployment should
add a dedicated document/PII OCR redaction service, security testing and human review.
