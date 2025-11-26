# Web App Backup Summary

This directory contains all the necessary files to run the PDE Solver Web App in a Docker container.

## Files Backed Up

### Application Code
- ✅ `app.py` - Streamlit web interface
- ✅ `multi_agent_orchestrator.py` - Multi-agent orchestrator
- ✅ `dispatcher_agent.py` - Dispatcher agent
- ✅ `pde_parser_agent.py` - PDE parser agent
- ✅ `pde_schema.py` - PDE parameters schema
- ✅ `conversation_memory.py` - Conversation memory management
- ✅ `fenics_mcp_server.py` - FEniCS MCP server

### Docker Configuration
- ✅ `Dockerfile` - Docker image configuration
- ✅ `docker-compose.yml` - Docker Compose configuration
- ✅ `.dockerignore` - Files to exclude from Docker build

### Dependencies
- ✅ `requirements.txt` - Python dependencies

### Documentation
- ✅ `README.md` - Main README
- ✅ `README_DOCKER.md` - Docker deployment instructions
- ✅ `README_MULTI_AGENT.md` - Multi-agent system documentation
- ✅ `README_WEB_INTERFACE.md` - Web interface documentation

### Configuration
- ✅ `.streamlit/` - Streamlit configuration directory
- ✅ `env.example` - Environment variables example

### Directories
- ✅ `data/` - Directory for generated data files (empty, will be created at runtime)
- ✅ `plots/` - Directory for generated visualization files (empty, will be created at runtime)

## Docker Deployment

### Quick Start

1. **Set up environment:**
   ```bash
   cd web_app
   cp env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

2. **Build and run:**
   ```bash
   docker-compose up -d
   ```

3. **Access the app:**
   Open http://localhost:8501 in your browser

### Manual Docker Build

```bash
cd web_app
docker build -t pde-solver-webapp .
docker run -d -p 8501:8501 -e OPENAI_API_KEY=your_key --name pde-solver pde-solver-webapp
```

## Files Not Included (By Design)

- `conversation_memory.json` - Generated at runtime
- `data/*.pkl` - Generated simulation data files
- `plots/*.html` - Generated visualization files
- `__pycache__/` - Python cache files
- `.env` - Should be created from env.example

## Notes

- All Python files are copied from the parent directory
- Directories for `data/` and `plots/` are created but empty (will be populated at runtime)
- The `.streamlit/` directory structure is preserved
- See `README_DOCKER.md` for detailed deployment instructions

