# InfraGuard AI — Final Architecture

```mermaid
flowchart TD
  A[Citizen: exact supplied mobile home UI] --> B[Tap Report an Issue]
  B --> C[First-use permissions: Camera + Mic + Location]
  C --> D[Live back camera]
  D --> E[Capture real photo]
  E --> F[GPS captured automatically]
  F --> G[Azure Maps reverse geocoding - optional]
  E --> H[Privacy Redaction - OpenCV]
  H --> H1[Blur faces]
  H --> H2[Blur human figures]
  H --> H3[Blur license plates]
  H --> H4[Blur QR codes]
  H --> I[Persist redacted image only]
  I --> J[Auto-start voice recording]
  J --> K[Sarvam Saaras v4 - translate mode]
  K --> L[22 Indian languages + English -> English text]
  L --> M[Sarvam-105B]
  M --> N[Issue + Severity + Department + Summary + Confidence]
  N --> O[Duplicate detection: same issue within 200m / 72h]
  O --> P[FastAPI case orchestration]
  P --> Q[(SQLite MVP database)]
  Q --> R[Officer dashboard]
  R --> S[Registered -> Assigned -> In Progress -> Resolved]
  S --> T[Resolution evidence]
  T --> U[Citizen My Reports + Notifications]
  Q --> V[Operations analytics]
```

## Stack

- Citizen interface: HTML, CSS, JavaScript
- Camera / microphone: Web Media APIs
- GPS: Browser Geolocation API
- Reverse geocoding: Azure Maps (optional key)
- Privacy redaction: OpenCV
- Voice-to-English: Sarvam Saaras v4, `mode=translate`
- AI triage: Sarvam-105B
- Backend: Python FastAPI + Uvicorn
- MVP database: SQLite
- Evidence: local processed image storage
- Repository: GitHub
- Production target: Azure App Service / Container Apps + Blob Storage + PostgreSQL + Key Vault + Entra + Application Insights
