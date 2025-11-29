# ConfirmIT AI Microservice (FastAPI)

**Multi-Agent AI system for receipt verification and fraud detection**

This service implements the Multi-Agent Coordination Pattern (MCP) with specialized agents for vision analysis, forensic detection, metadata extraction, reputation checking, and reasoning.

## 🤖 Architecture

- **Framework:** FastAPI (Python 3.11+)
- **AI Model:** Google Gemini 2.0 Flash (via Generative AI SDK)
- **Agents:** Vision, Forensic, Metadata, Reputation, Reasoning
- **Pattern:** Multi-Agent Coordination Pattern (MCP)
- **Image Processing:** Pillow, OpenCV, scikit-image

## 🧠 Agent System

### 1. **Vision Agent** (`vision_agent.py`)
- OCR text extraction
- Visual anomaly detection
- Confidence scoring
- Uses: Gemini Vision API

### 2. **Forensic Agent** (`forensic_agent.py`)
- Error Level Analysis (ELA)
- Copy-Move detection (SIFT)
- Noise analysis
- Compression artifact detection

### 3. **Metadata Agent** (`metadata_agent.py`)
- EXIF metadata extraction
- Date/time consistency checks
- Software detection (editing tools)

### 4. **Reputation Agent** (`reputation_agent.py`)
- Account number extraction
- Fraud report lookup
- Verified business check
- Pattern matching for scam accounts

### 5. **Reasoning Agent** (`reasoning_agent.py`)
- Synthesizes all agent outputs
- Generates final verdict
- Creates user-friendly explanations
- Calculates trust score

### 6. **Orchestrator** (`orchestrator.py`)
- Coordinates all agents
- Parallel execution with timeouts
- Result aggregation
- Error handling

## 📁 Project Structure

```
ai-service/
├── app/
│   ├── main.py                 # FastAPI app
│   ├── config.py               # Configuration
│   ├── agents/                 # MCP agent system
│   │   ├── orchestrator.py     # Agent coordinator
│   │   ├── vision_agent.py     # Gemini Vision
│   │   ├── forensic_agent.py   # Image forensics
│   │   ├── metadata_agent.py   # EXIF analysis
│   │   ├── reputation_agent.py # Fraud checking
│   │   └── reasoning_agent.py  # Final synthesis
│   ├── routers/                # API endpoints
│   │   ├── receipts.py         # Receipt analysis
│   │   └── accounts.py         # Account checking
│   ├── services/               # External services
│   │   ├── gemini_service.py   # Gemini API client
│   │   ├── cloudinary_service.py
│   │   └── firebase_service.py
│   └── schemas/                # Pydantic models
│       ├── receipt.py
│       └── account.py
├── tests/                      # Unit tests
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- pip or pipenv
- Google Gemini API key
- Firebase credentials
- Cloudinary account

### Installation

1. **Create virtual environment:**

```bash
cd ai-service
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:
- Gemini API key (from Google AI Studio)
- Firebase service account credentials
- Cloudinary URL

4. **Run the development server:**

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

5. **View API documentation:**

Open `http://localhost:8000/docs` for interactive Swagger UI

## 📡 API Endpoints

### Receipt Analysis
- `POST /analyze-receipt` - Analyze receipt image
  ```json
  {
    "image_url": "https://cloudinary.com/...",
    "options": {
      "include_forensics": true,
      "check_reputation": true
    }
  }
  ```

### Account Checking
- `POST /check-account` - Check account reputation
  ```json
  {
    "account_hash": "sha256_hash",
    "bank_code": "058"
  }
  ```

### Health Check
- `GET /health` - Service health status

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_forensic_agent.py
```

## 🐳 Docker Deployment

```bash
# Build image
docker build -t confirmit-ai-service .

# Run container
docker run -p 8000:8000 --env-file .env confirmit-ai-service
```

## 🚀 Production Deployment (Google Cloud Run)

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/confirmit-ai-service
gcloud run deploy confirmit-ai-service \
  --image gcr.io/PROJECT_ID/confirmit-ai-service \
  --platform managed \
  --region us-central1 \
  --set-env-vars "$(cat .env | xargs)"
```

## 📊 Performance

- Receipt analysis: < 8 seconds
- Account check: < 3 seconds
- Agent execution: Parallel with 30s timeout
- Max concurrent requests: 100

## 🔒 Security

- API key authentication
- Rate limiting
- Input validation
- Secure credential storage
- No logging of sensitive data

## 🐛 Debugging

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
uvicorn app.main:app --reload --log-level debug
```

## 📝 Development Tips

### Add New Agent
1. Create agent file in `app/agents/`
2. Implement `analyze()` method
3. Register in `orchestrator.py`
4. Add tests in `tests/`

### Test Gemini API
```python
from app.services.gemini_service import GeminiService

service = GeminiService()
result = await service.analyze_image("https://image.url")
```

## 🤝 Contributing

1. Create feature branch
2. Make changes
3. Write tests
4. Submit PR

## 📄 License

MIT License

---

**Built with ❤️ for Codematics Hackathon 2025**
