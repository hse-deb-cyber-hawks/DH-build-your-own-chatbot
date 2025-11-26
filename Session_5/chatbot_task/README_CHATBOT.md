# Chatbot Solution

This project is a chatbot application designed to run on multiple platforms, including Jetson Nano, Linux, Windows, and macOS. The project includes a backend and frontend, as well as additional services like Chroma and Ollama for AI processing and management of the chatbot's tasks.

## Requirements

- [Docker](https://docs.docker.com/get-docker/) (Ensure Docker and Docker Compose are installed)
- GPU support for Ollama (if applicable)

## Quick Start

You can start the chatbot application by following the steps below based on your platform.

### Run Chatbot Application

#### On Jetson Nano:

To run the chatbot on a Jetson Nano, use the following commands:

```bash
cd /Session_5/chatbot_task/
docker compose -f docker-compose-jetson.yml up --build
```

#### On Linux, Windows, or macOS:

To run the chatbot on Linux, Windows, or macOS, use the following command:

```bash
docker compose -f docker-compose.yml up --build
```
or
```bash
docker compose -f Session_5/chatbot_task/docker-compose.yml up --build
```

This command will build and run the chatbot using the appropriate Docker Compose configuration for your system.

## Folder Structure

Below is the folder structure of the project and a detailed explanation of its contents:

```bash
.
├── app/
│   ├── Dockerfile                   # Dockerfile for building the chatbot service
│   ├── main.py                      # Streamlit app
│   ├── pyproject.toml               # Python project configuration for backend
│   ├── requirements-dev.lock        # Backend development dependency lock file
│   ├── requirements.lock            # Backend production dependency lock file
│   └── src/
│       ├── AI_Book.pdf              # Document for knowledge base
│       ├── chatbot.py               # Core logic of the chatbot, including AI-related operations
│       ├── __init__.py              # Marks the `src` directory as a Python module
├── docker-compose-jetson.yml        # Docker Compose configuration for Jetson Nano
├── docker-compose.yml               # Docker Compose configuration for Linux, Windows, and macOS
└── README_CHATBOT.md                # Instructions for running the chatbot application
```

### Explanation of Folders and Files:

- **app/**: Contains the application code and dependencies. The app contains the streamlit app and handles the core logic for the chatbot, including AI processing.
    - **Dockerfile**: Dockerfile for building the backend service.
    - **main.py**: Main entry point for the backend, likely initializing the chatbot and managing backend operations.
    - **pyproject.toml**: Python configuration file for backend dependencies.
    - **requirements-dev.lock**: Lock file for development dependencies of the backend.
    - **requirements.lock**: Lock file for production dependencies of the backend.
    - **src/**: Contains source code for the backend service.
        - **AI_Book.pdf**: Document to embedd and index into chroma vector store.
        - **chatbot.py**: Contains core chatbot logic, managing conversation flow, AI interaction, etc.
        - **__init__.py**: Initializes the `src` directory as a Python module.

- **docker-compose-jetson.yml**: Docker Compose configuration file for running the chatbot application on a Jetson Nano.

- **docker-compose.yml**: Docker Compose configuration file for running the application on Linux, Windows, or macOS.

- **run_ollama.sh**: Shell script to set up and run the Ollama service.

---

## Docker Compose Overview

The `docker-compose-jetson.yml` file orchestrates the different services of the chatbot, such as the frontend, backend, and AI-related services like Chroma and Ollama. Below is a breakdown of each service and how they are set up:

```yaml
version: '3'

services:

  chatbot:
    build:
      context: ./app
      dockerfile: Dockerfile
    ports:
      - "5001:5001"
      - "5678:5678"
    container_name: chatbot
    environment:
      - INDEX_DATA=0
      - MODEL_NAME=llama3.2:1B
      - EMBEDDING_MODEL=bge-m3
      - PULL_EMBEDDING_MODEL=0
      - OLLAMA_HOST_NAME=ollama
      - CHROMA_HOST_NAME=chroma
      - PDF_DOC_PATH=/app/src/AI_Book.pdf
    depends_on:
      chroma:
        condition: service_healthy
    networks:
      - app-network
    volumes:
      - ./app:/app

  chroma:
    image: chromadb/chroma:0.5.13
    volumes:
      - ../../container_cache/chroma:/chroma/chroma
    ports:
      - "8000:8000"
    container_name: chroma
    environment:
      - IS_PERSISTENT=TRUE
      - PERSIST_DIRECTORY=./chroma/chroma
    healthcheck: 
        test: curl localhost:8000/api/v1/heartbeat || exit 1
        interval: 10s
        retries: 2
        start_period: 5s
        timeout: 10s
    networks:
      - app-network

  ollama:
    image: makoit13/ollama:r36.4.0
    ports:
      - 11434:11434
    volumes:
      - ../../container_cache/ollama/models:/data/models/ollama/models
    container_name: ollama
    tty: true
    entrypoint: ["/bin/bash", "-c"]
    command: ["( ollama serve & sleep 5; ollama run llama3.2:1B )"]
    environment:
      - OLLAMA_KEEP_ALIVE=24h
      - OLLAMA_HOST=0.0.0.0
    networks:
      - app-network
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

networks:
  app-network:
    driver: bridge
```

### Explanation of Services

### 1. Chatbot
- Builds from `./app/Dockerfile`.
- Exposes ports `5001` and `5678`.
- Uses **Llama 3.2 (1B)** model with **bge-m3** embeddings.
- Depends on ChromaDB for vector storage.
- Mounts `./app` as a volume.

### 2. ChromaDB
- Uses `chroma:0.5.13` Docker image.
- Exposes port `8000`.
- Persistent storage enabled at `../../container_cache/chroma`.
- Health check on `/api/v1/heartbeat`.

### 3. Ollama (AI Model Execution)
- Uses `makoit13/ollama:r36.4.0` Docker image.
- Runs **Llama 3.2 (1B)** model.
- Exposes port `11434`.
- Utilizes **NVIDIA GPU** for acceleration.
- Persistent model storage at `../../container_cache/ollama/models`.

## Network Configuration
All services communicate through the `app-network` bridge network.

## Notes
- Ensure that an **NVIDIA GPU** is available for Ollama to function correctly.
- The chatbot uses `AI_Book.pdf` as input data.
- Modify the `MODEL_NAME` or `EMBEDDING_MODEL` as needed.
