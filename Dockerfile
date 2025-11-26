# Dockerfile for PDE Solver Web App
# Uses conda to install FEniCS (complex dependencies)
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Install system dependencies for FEniCS
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Ensure conda is initialized and PATH is set
ENV PATH="/opt/conda/bin:${PATH}"

# Install FEniCS via conda-forge (version 2019.1.0 to match fem-gpt environment)
RUN conda install -y -c conda-forge \
    fenics-dolfin=2019.1.0 \
    fenics-ffc=2019.1.0 \
    fenics-fiat=2019.1.0 \
    fenics-ufl=2019.1.0 \
    fenics-dijitso=2019.1.0 \
    && conda clean -afy

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies via pip (use conda's pip)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./
COPY *.md ./

# Create necessary directories
RUN mkdir -p plots data .streamlit

# Create .streamlit config if not exists
RUN echo "[general]\nemail = \"\"" > .streamlit/config.toml

# Verify FEniCS installation
RUN python -c "from dolfin import *; from fenics import *; print('FEniCS installed successfully')" || echo "FEniCS verification failed"

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
