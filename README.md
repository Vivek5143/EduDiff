# EduDiff (AI-Powered Math Animation Generator)

EduDiff is a full-stack app that generates **short math explanation videos** using **Manim** on the backend and a **Next.js** UI on the frontend.

- **Frontend**: Next.js at `http://localhost:3000`
- **Backend**: Flask + Manim at `http://localhost:5001`
- **LLM**: Google Gemini via the supported **`google-genai`** SDK

---

## Features

- **Generate Manim videos from text prompts** (concepts + equations)
- **Docker Compose** setup (frontend + backend)
- **Video serving** from the backend (`/static/videos/...`)

---

## Repo Layout

```text
EduDiff/
  docker-compose.yaml
  backend/
    app.py
    Dockerfile.backend
    requirements.txt
    static/
    templates/
  frontend/
    Dockerfile.frontend
    package.json
    src/
```

---

## Quick Start (Docker — Recommended)

### 1) Create `backend/.env`

Create `EduDiff/backend/.env`:

```env
# Use either GOOGLE_API_KEY or GEMINI_API_KEY
GOOGLE_API_KEY=YOUR_KEY_HERE
# GEMINI_API_KEY=YOUR_KEY_HERE

GENAI_MODEL=gemini-2.5-flash
RENDER_QUALITY=low

PORT=5001
FLASK_ENV=production
```

### 2) Build + run

From the `EduDiff/` directory:

```bash
docker-compose up --build
```

### 3) Open the app

- **UI**: `http://localhost:3000`
- **Backend**: `http://localhost:5001`

---

## API

### POST `/generate`

Request JSON:

```json
{
  "concept": "Solve 3x - 5 = 10",
  "quality": "low"
}
```

Response includes:
- `video_url`: backend-served URL for the generated MP4
- `code`: generated Manim code
- `explanation`: short text explanation

### GET `/demos`

Returns demo GIF metadata used by the landing page.

---

## Run Without Docker (Local Dev)

### Backend

From `EduDiff/backend`:

```bash
pip install -r requirements.txt
python app.py
```

### Frontend

From `EduDiff/frontend`:

```bash
npm install
npm run dev
```

The frontend reads `NEXT_PUBLIC_API_URL` to call the backend. In Docker this is set automatically to `http://backend:5001`.

---

## Troubleshooting

### Docker build fails with dependency conflicts

The backend uses `google-genai`, which requires a newer `requests`. Keep the pinned range in `backend/requirements.txt`.

### “ffmpeg not found” warnings

The backend Docker image installs `ffmpeg`. If you changed Dockerfiles, rebuild:

```bash
docker-compose build backend
docker-compose up -d backend
```

### Backend starts but AI features don’t work

Confirm `backend/.env` contains a valid `GOOGLE_API_KEY` (or `GEMINI_API_KEY`).

---

## License

MIT (see `backend/LICENSE`).


