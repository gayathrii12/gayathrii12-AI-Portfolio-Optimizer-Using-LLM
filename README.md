# Financial Returns Optimizer

[![Node.js](https://img.shields.io/badge/Node.js-18%2B-green.svg)](https://nodejs.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org/)
[![React](https://img.shields.io/badge/React-18.2-blue.svg)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-4.7-blue.svg)](https://typescriptlang.org/)

A comprehensive multi-agent financial planning system that processes 98+ years of historical market data through AI agents to generate personalized portfolio recommendations, investment projections, and rebalancing strategies.

## Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Install & Run](#install--run)
  - [macOS](#macos)
  - [Windows](#windows)
- [Environment Variables](#environment-variables)
- [Available Scripts](#available-scripts)
- [Testing](#testing)
- [Linting & Formatting](#linting--formatting)
- [Build & Deploy](#build--deploy)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Architecture Overview](#architecture-overview)
- [License](#license)

## Quick Start

**TL;DR** - Get up and running in 6 commands:

```bash
# Clone and install
git clone <repository-url>
cd financial-returns-optimizer
npm install --prefix src/frontend
pip install -r config/requirements.txt

# Start backend and frontend
./start_backend.sh    # Terminal 1 - Backend on :8000
./start_frontend.sh   # Terminal 2 - Frontend on :3000
```

## Prerequisites

### Required Software

- **Node.js**: 18+ LTS (detected from package.json engines)
- **Python**: 3.9+ (for backend ML agents)
- **Git**: Latest version
- **Package Manager**: npm (detected from package-lock.json)

### Platform-Specific Requirements

**macOS:**
- Xcode Command Line Tools: `xcode-select --install`
- Homebrew (recommended): `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

**Windows:**
- Microsoft Visual C++ Build Tools or Visual Studio 2019+
- Windows Subsystem for Linux (WSL2) - recommended for Python development
- PowerShell 5.1+ or PowerShell Core 7+

## Install & Run

### macOS

1. **Install Node.js using nvm (recommended):**
   ```bash
   # Install nvm
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
   source ~/.zshrc
   
   # Install and use Node.js 18
   nvm install 18
   nvm use 18
   ```

2. **Install Python dependencies:**
   ```bash
   # Using Homebrew (recommended)
   brew install python@3.9
   
   # Install backend dependencies
   pip install -r config/requirements.txt
   ```

3. **Install frontend dependencies:**
   ```bash
   cd src/frontend
   npm ci
   cd ../..
   ```

4. **Set up environment:**
   ```bash
   # Copy environment template
   cp .env.example .env.local
   # Edit .env.local with your configuration
   ```

5. **Start the application:**
   ```bash
   # Terminal 1 - Backend (Port 8000)
   ./start_backend.sh
   
   # Terminal 2 - Frontend (Port 3000)
   ./start_frontend.sh
   ```

6. **Open in browser:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000/docs

### Windows

1. **Install Node.js using nvm-windows:**
   ```powershell
   # Download and install nvm-windows from GitHub releases
   # https://github.com/coreybutler/nvm-windows/releases
   
   # Install Node.js 18
   nvm install 18.17.0
   nvm use 18.17.0
   ```

2. **Install Python:**
   ```powershell
   # Using winget (Windows 10+)
   winget install Python.Python.3.9
   
   # Or download from python.org
   # Install backend dependencies
   pip install -r config/requirements.txt
   ```

3. **Install frontend dependencies:**
   ```powershell
   cd src/frontend
   npm ci
   cd ../..
   ```

4. **Set up environment:**
   ```powershell
   # Copy environment template
   Copy-Item .env.example .env.local
   # Edit .env.local with your configuration
   ```

5. **Start the application:**
   ```powershell
   # Terminal 1 - Backend
   python src/backend/backend_api_with_agents.py
   
   # Terminal 2 - Frontend
   cd src/frontend
   npm start
   ```

**Alternative: Using WSL2 (Recommended)**
```bash
# Follow the macOS instructions within WSL2
wsl --install
# Then use the macOS setup steps
```

### Port Configuration

- **Frontend**: Default 3000, change with `PORT=3001 npm start`
- **Backend**: Default 8000, change in `src/backend/backend_api_with_agents.py`
- **Proxy**: Frontend proxies API calls to backend via `package.json` proxy setting

## Environment Variables

Create `.env.local` in the project root with these variables:

| Variable | Description | Default | Used In |
|----------|-------------|---------|---------|
| `REACT_APP_API_URL` | Backend API base URL | `http://localhost:8000` | Frontend services |
| `REACT_APP_AGENT_API_URL` | Agent API base URL | `http://localhost:8000` | Agent services |
| `PYTHON_ENV` | Python environment | `development` | Backend config |
| `LOG_LEVEL` | Logging level | `INFO` | Backend logging |
| `HISTORICAL_DATA_FILE` | Path to Excel data | `data/histretSP.xls` | Data processing |
| `ML_MODEL_PATH` | ML models directory | `saved_models/` | ML predictions |

**Example .env.local:**
```bash
# API Configuration
REACT_APP_API_URL=http://localhost:8000
REACT_APP_AGENT_API_URL=http://localhost:8000

# Backend Configuration
PYTHON_ENV=development
LOG_LEVEL=INFO
HISTORICAL_DATA_FILE=src/backend/data/histretSP.xls
ML_MODEL_PATH=saved_models/

# Optional: External Services
# SENTRY_DSN=your_sentry_dsn_here
# DATADOG_API_KEY=your_datadog_key_here
```

## Available Scripts

### Frontend Scripts (in `src/frontend/`)

| Script | Command | Description | Output |
|--------|---------|-------------|---------|
| `start` | `npm start` | Development server with hot reload | http://localhost:3000 |
| `build` | `npm run build` | Production build | `build/` directory |
| `test` | `npm test` | Run Jest tests in watch mode | Test results |
| `eject` | `npm run eject` | Eject from Create React App | ⚠️ Irreversible |

### Backend Scripts

| Script | Command | Description |
|--------|---------|-------------|
| Start Server | `python src/backend/backend_api_with_agents.py` | FastAPI server with agents |
| Run Tests | `pytest tests/` | Run Python test suite |
| Format Code | `black src/backend/` | Format Python code |
| Type Check | `mypy src/backend/` | Static type checking |

### Helper Scripts

| Script | Description |
|--------|-------------|
| `./start_backend.sh` | Start backend server (macOS/Linux) |
| `./start_frontend.sh` | Start frontend server (macOS/Linux) |
| `python test_api_response.py` | Test API endpoints |

## Testing

### Frontend Testing

**Run all tests:**
```bash
cd src/frontend
npm test
```

**Run specific test file:**
```bash
npm test -- --testPathPattern=PortfolioRecommendation
```

**Update snapshots:**
```bash
npm test -- --updateSnapshot
```

**Coverage report:**
```bash
npm test -- --coverage --watchAll=false
```

### Backend Testing

**Run all tests:**
```bash
pytest tests/
```

**Run with coverage:**
```bash
pytest tests/ --cov=src/backend --cov-report=html
```

**Run specific test:**
```bash
pytest tests/test_return_prediction_agent.py -v
```

## Linting & Formatting

### Frontend

**ESLint (configured via Create React App):**
```bash
cd src/frontend
npm run lint  # If configured
```

**Prettier (if configured):**
```bash
npm run format  # If configured
```

### Backend

**Black formatting:**
```bash
black src/backend/
```

**Flake8 linting:**
```bash
flake8 src/backend/
```

**MyPy type checking:**
```bash
mypy src/backend/
```

## Build & Deploy

### Frontend Production Build

```bash
cd src/frontend
npm run build
```

**Output:** `src/frontend/build/` directory with optimized static files.

**Environment Handling:**
- Development: `.env.local`
- Production: Environment variables injected at build time
- Docker: Use multi-stage builds for optimization

### Backend Deployment

**Production server:**
```bash
cd src/backend
uvicorn backend_api_with_agents:app --host 0.0.0.0 --port 8000
```

**Docker deployment:**
```dockerfile
# Example Dockerfile structure
FROM python:3.9-slim
COPY config/requirements.txt .
RUN pip install -r requirements.txt
COPY src/backend/ ./app/
CMD ["uvicorn", "app.backend_api_with_agents:app", "--host", "0.0.0.0"]
```

## Troubleshooting

### Jest "fsevents unavailable" Error

**Problem:** `fsevents unavailable (this watcher can only be used on Darwin)`

**Solutions:**

1. **Run tests without watch mode:**
   ```bash
   npm test -- --watchAll=false
   ```

2. **Install Watchman on macOS:**
   ```bash
   brew install watchman
   ```

3. **Set CI environment for non-interactive runs:**
   ```bash
   CI=true npm test
   ```

4. **Windows: Use polling mode:**
   ```bash
   npm test -- --watchAll=false
   # Or set CHOKIDAR_USEPOLLING=true
   ```

### Port Already in Use

**Frontend (3000):**
```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
# Or use different port
PORT=3001 npm start
```

**Backend (8000):**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
# Or change port in backend_api_with_agents.py
```

### Node-gyp Build Failures

**macOS:**
```bash
xcode-select --install
```

**Windows:**
```powershell
npm install --global windows-build-tools
# Or install Visual Studio Build Tools
```

**Linux:**
```bash
sudo apt-get install build-essential
```

### Python/Backend Issues

**Missing dependencies:**
```bash
pip install -r config/requirements.txt --upgrade
```

**Permission errors:**
```bash
# Use virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
pip install -r config/requirements.txt
```

### Proxy/SSL Issues

**Development proxy not working:**
- Check `package.json` proxy setting: `"proxy": "http://localhost:8000"`
- Restart frontend server after backend changes
- Use absolute URLs in development if needed

## Contributing

### Branch Naming
- Feature: `feature/portfolio-optimization`
- Bug fix: `fix/chart-rendering-issue`
- Hotfix: `hotfix/security-patch`

### Commit Style
Follow [Conventional Commits](https://conventionalcommits.org/):
```bash
feat: add portfolio rebalancing algorithm
fix: resolve chart rendering on mobile
docs: update API documentation
test: add unit tests for risk calculation
```

### Pre-commit Checks
Run these before committing:
```bash
# Frontend
cd src/frontend && npm test -- --watchAll=false
cd src/frontend && npm run build

# Backend
pytest tests/
black src/backend/
mypy src/backend/
```

## Architecture Overview

### System Components

- **Frontend**: React 18 + TypeScript + Styled Components
- **Backend**: FastAPI + Python 3.9 + LangChain agents
- **Data Processing**: Multi-agent pipeline with 98+ years of S&P 500 data
- **ML Models**: Scikit-learn for return predictions
- **Testing**: Jest + React Testing Library (frontend), pytest (backend)

### Key Features

- **Multi-Agent Architecture**: LangChain orchestration with specialized agents
- **Historical Data Analysis**: 98+ years of S&P 500 returns (1927-2024)
- **Risk-Based Portfolio Allocation**: Conservative, Moderate, Aggressive profiles
- **Investment Projections**: Lump sum and SIP (Systematic Investment Plan)
- **Real-time Monitoring**: System health and performance dashboards
- **ML-Enhanced Predictions**: Asset return forecasting with confidence scores

### Data Flow

1. **User Input** → Investment amount, risk profile, tenure
2. **Agent Pipeline** → Data cleaning, prediction, allocation
3. **Portfolio Generation** → Optimized asset allocation
4. **Visualization** → Charts, projections, recommendations

For detailed architecture documentation, see [docs/PROJECT_WORKFLOW.md](docs/PROJECT_WORKFLOW.md).

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

**Need Help?** 
- [Detailed Workflow Guide](docs/PROJECT_WORKFLOW.md)
- [Report Issues](https://github.com/your-repo/issues)
- [Discussions](https://github.com/your-repo/discussions)
