# Select AI
Natural Language to SQL Query Generator for Teradata.

## Overview

Select AI is a natural language to SQL assistant that enables users to query databases using plain English. The application leverages the Code Llama 7B Instruct model to translate natural language questions into SQL queries and run it on Teradata.

Key features:

- Natural language to SQL translation
- Interactive web interface with real-time feedback
- Query execution against Teradata databases
- Batch processing mode for multiple queries

## Table of Contents

- [Project Structure](#project-structure)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Conda Environment](#conda-environment)
  - [GPU Configuration](#gpu-configuration)
  - [Download Model](#download-model)
  - [Database Setup](#database-setup)
- [Testing](#testing)
- [Usage](#usage)
  - [Web UI Mode](#web-ui-mode-default)
  - [Batch Mode](#batch-mode)
- [Docker Deployment](#docker-deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Project Structure
```
SelectAI/
├── app.py                   # Main application entry point
├── batch.py                 # Batch processing module
├── db.py                    # Database connection and operations
├── nl2sql.py                # Core NL-to-SQL conversion logic
├── prompt.txt               # Prompt template for the LLM
├── requirements.txt         # Python dependencies
├── config/                  # Configuration files
│   └── .env                 # Environment variables
├── models/                  # Storage for local LLM models
│   └── code-llama-7b-instruct/
├── scripts/                 # Utility scripts
│   ├── clear_cache.py       # Clears model cache
│   ├── download_model.py    # Downloads LLM model
│   └── td_init.sql          # Database initialization
├── static/                  # Web assets
│   ├── css/                 # Stylesheets
│   ├── img/                 # Images
│   └── js/                  # JavaScript files
├── templates/               # HTML templates
│   └── index.html           # Main application page
├── test/                    # Test modules
│   ├── end2end/             # Contains questions files for batch mode operation and Excel sheet outputs 
│   ├── test_db.py           # Database tests
│   ├── test_model.py        # Model tests
│   └── test_extract_sql.py  # SQL extraction tests
└── docs/                    # Project documents
    ├── dataset              # Contains data dictionary and sample data in CSV format.
    └── snapshots            # UI screenshots 
```

#### Important Files:

- GUI/frontend: `app.py`, `static\`, and `templates\`
- Backend: `db.py` and `nl2sql.py`
- Batch mode: `app.py` and `batch.py`
- Model folder: `models\code-llama-7b-instruct\`
- Prompt template: `prompt.txt`

## Setup for Development

#### Conda Environment

1. Create a new Conda environment:

    ```bash
	conda create --name select-ai 
    ```

2. Install pip:

    ```bash
	conda install pip
    ```

3. Install project dependencies:

    ```bash
	pip install -r requirements.txt
    ```

#### GPU Configuration

1. Download and install the Nvidia driver appropriate for your GPU
2. Install the CUDA toolkit:
   - Download from: https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local
   - Follow the installation instructions

3. Install CUDA deep learning package (cuDNN):
   - Download from: https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local
   - Extract and follow installation instructions

4. Set up PyTorch with CUDA support:
   ```bash
   # In your Conda environment
   pip uninstall torch torchvision torchaudio -y
   pip install torch --index-url https://download.pytorch.org/whl/cu126
   ```

5. Verify CUDA installation:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device count: {torch.cuda.device_count()}")
   print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
   ```

### Mac GPUs (Apple Silicon or Metal-compatible Intel)
1. Ensure PyTorch 2.0+ is installed:
   ```bash
   pip install --upgrade torch
   ```

#### Download Model

The application uses the Code Llama 7B Instruct model for NL-to-SQL conversion.

1. Use the download script:
   ```bash
   python scripts/download_model.py
   ```

2. The model will be downloaded to the `models/code-llama-7b-instruct/` directory.

#### Database Setup

1. Create a Teradata account on the Clearscape Analytics platform: https://clearscape.teradata.com/ 
2. Use the `scripts/td_init.sql` SQL script to create the database, the tables, and insert sample data.
3. Configure database credentials in `config/.env`:
   ```
   TD_HOST=your-teradata-host.com
   TD_NAME=your-database-name
   TD_USER=your-username
   TD_PASSWORD=your-password
   TD_PORT=1025
   ```

## How to Test

Run tests to verify the application components are working correctly:

#### Test Database Connection

```bash
python test/test_db.py
```

This test connects to the Teradata database, retrieves the schema, and executes a sample query to validate connectivity.

#### Test Model Integration

```bash
python test/test_model.py
```

This test verifies that:
- The model files are present
- The model loads correctly
- The model can generate responses to prompts

#### Test SQL Extraction

```bash
python test/test_extract_sql.py
```

This test validates the SQL extraction logic from model outputs.

## How to Run

The application can be run in two modes:

#### Web UI Mode (Default)

```bash
python app.py
```

This starts the web server on `http://localhost:5000`, where you can:
1. Connect to the database
2. Load the AI model
3. Enter natural language queries
4. Get translated SQL queries
5. Execute queries and view results

#### Batch Mode

```bash
python app.py --batch
```

This processes a batch of questions from the file specified in `QUESTIONS_PATH` in the `.env` file and outputs results to an Excel file.

## Containerization  

#### Docker Setup

1. Build the docker image:
   ```bash
   docker build --no-cache -t selectai .
   ```

2. Make sure the Clearscape environment is running.

3. Run the container with port mapping and model mounting:
   ```bash
   docker run --gpus all -p 5000:5000 -v "$(pwd)/models:/app/models" selectai
   ```
   Replace `$(pwd)` with the path to the models directory.

4. After starting, verify with:
   ```bash
   docker ps
   ```
   You should see a line like:
   ```
   0.0.0.0:5000->5000/tcp
   ```
   in the PORTS column.

5. Access the application:  
   Open your browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

#### Additional Requirements

The model should be downloaded to the `code-llama-7b-instruct` directory before building the Docker image or should be mounted as a volume as shown in the docker-compose.yml file.
