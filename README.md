# Chemical Eye

**Planetary Chemical Intelligence from Orbit**

Chemical Eye is a hyperspectral analysis platform that detects chemicals, minerals, and environmental signatures from satellite imagery using NASA EMIT data.

## Features

- 🔥 **Methane Detection** — Detect confirmed methane plumes using EMIT L2B data
- 🌾 **Spectral Analysis** — Full hyperspectral fingerprinting (coming soon)
- 🌱 **Vegetation Indices** — NDVI, nitrogen stress, and more
- 🪨 **Mineral Detection** — Clay, iron oxide, lithium signatures

## Quick Start

### 1. Install Dependencies

```bash
cd chemeye
pip install -e ".[dev]"
```

Or using requirements.txt:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your NASA EarthData credentials
```

Get NASA EarthData credentials at: https://urs.earthdata.nasa.gov/

### 3. Create an API Key

```bash
python -m chemeye.cli.create_key --email you@example.com --name "Dev Key"
```

Save the key that's printed — you won't be able to see it again!

### 4. Run the API

```bash
uvicorn src.chemeye.api.app:app --reload --port 8000
```

Visit:
- API Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### 5. Make a Detection Request

```bash
curl -X POST http://localhost:8000/v1/detect/methane \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "bbox": {
      "min_lon": -117.5,
      "min_lat": 35.2,
      "max_lon": -117.0,
      "max_lat": 35.7
    },
    "start_date": "2023-05-01",
    "end_date": "2023-08-30"
  }'
```

## Deploy to Modal

Chemical Eye can be deployed to Modal for serverless, scalable hosting.

### 1. Install Modal

```bash
pip install modal
modal setup
```

### 2. Create Secrets

In the Modal dashboard, create a secret called `chemeye-secrets` with:
- `NASA_EARTHDATA_USERNAME`
- `NASA_EARTHDATA_PASSWORD`
- `SECRET_KEY`
- `ADMIN_TOKEN`

### 3. Deploy

```bash
modal deploy modal_app.py
```

## Project Structure

```
chemeye/
├── src/chemeye/
│   ├── api/           # FastAPI application
│   │   ├── app.py     # Main app
│   │   ├── routes/    # API endpoints
│   │   ├── schemas.py # Pydantic models
│   │   └── deps.py    # Dependencies
│   ├── services/      # Core services
│   │   ├── emit.py    # NASA data access
│   │   ├── methane.py # Methane detection
│   │   └── indices.py # Spectral indices
│   ├── cli/           # Command-line tools
│   ├── config.py      # Configuration
│   ├── database.py    # SQLAlchemy models
│   └── auth.py        # API key auth
├── landing/           # Landing page
├── modal_app.py       # Modal deployment
├── pyproject.toml     # Project config
└── requirements.txt   # Dependencies
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/v1/status` | API status |
| POST | `/v1/detect/methane` | Detect methane plumes |

## License

MIT
