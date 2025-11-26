# PDE Solver Web App

A Streamlit-based web application for solving Partial Differential Equations (PDEs) using natural language input.

## Features

- **Multi-Agent System**: Uses specialized agents for parsing and solving PDE problems
- **Natural Language Interface**: Describe your PDE problem in plain English
- **Support for Multiple PDE Types**: Heat transfer and elasticity problems
- **1D, 2D, and 3D Simulations**: Support for different spatial dimensions
- **Interactive Visualizations**: Plotly-based interactive 3D visualizations
- **Conversation Memory**: Context-aware follow-up questions
- **LLM-Based Validation**: Intelligent validation of PDE queries

## Quick Start

### Using Docker (Recommended)

1. **Build the Docker image:**
   ```bash
   docker build -t pde-solver-webapp .
   ```

2. **Run with environment variable:**
   ```bash
   docker run -d -p 8501:8501 -e OPENAI_API_KEY=your_api_key --name pde-solver pde-solver-webapp
   ```

3. **Or use docker-compose:**
   ```bash
   # Create .env file with OPENAI_API_KEY=your_api_key
   docker-compose up -d
   ```

4. **Access the app:**
   Open http://localhost:8501 in your browser

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variable:**
   ```bash
   export OPENAI_API_KEY=your_api_key
   # Or create a .env file
   ```

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Configuration

Create a `.env` file in the `web_app` directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage Examples

### Heat Transfer

- "Solve 1D heat transfer in a 2 meter rod, left end at 20°C, right end at 0°C"
- "Simulate heat diffusion in a 1m x 1m plate with initial temperature 10°C"
- "Solve 2D heat equation on a 1m x 1m plate, boundary at 0°C, initial at 20°C"

### Elasticity

- "Solve 2D elasticity problem on a 1m x 1m plate with Young's modulus 210 GPa"
- "3D elasticity problem on a 1m x 0.2m x 0.2m cube with gravity"
- "1D bar elasticity with length 2m, Young's modulus 70 GPa (aluminum)"

### Follow-up Queries

- "Change the left boundary temperature to 50°C"
- "Add gravity to the elasticity problem"
- "Change Young's modulus to 70 GPa"

## Project Structure

```
web_app/
├── app.py                      # Streamlit web interface
├── multi_agent_orchestrator.py # Main orchestrator
├── dispatcher_agent.py         # Dispatcher agent
├── pde_parser_agent.py         # PDE parser agent
├── pde_schema.py              # PDE parameters schema
├── conversation_memory.py      # Conversation memory
├── fenics_mcp_server.py       # FEniCS MCP server
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose configuration
├── .dockerignore             # Docker ignore file
├── .env.example              # Environment variables example
├── README.md                 # This file
├── README_DOCKER.md          # Docker-specific documentation
├── data/                     # Generated data files
└── plots/                    # Generated visualization files
```

## Documentation

- See `README_DOCKER.md` for detailed Docker deployment instructions
- See `README_MULTI_AGENT.md` for multi-agent system documentation
- See `README_WEB_INTERFACE.md` for web interface details

## Troubleshooting

1. **FEniCS installation issues**: Make sure all system dependencies are installed (see Dockerfile)

2. **API Key errors**: Verify that `OPENAI_API_KEY` is set correctly

3. **Port conflicts**: Change the port in docker-compose.yml or when running docker run

4. **Memory issues**: Increase Docker memory allocation for complex simulations

## License

[Add your license here]

