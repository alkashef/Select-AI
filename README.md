# Raven: Natural Language to SQL with LLMs

## Overview
Raven is a Python application that converts natural language questions into SQL queries using a locally hosted LLM (Code Llama 7B Instruct) and executes them on a PostgreSQL database. It features a Streamlit web interface for user interaction and supports database schema introspection, prompt customization, and sample data display.

---

## Project Setup

### 1. Install PostgreSQL Server
1. Download and install PostgreSQL from the [official website](https://www.postgresql.org/download/).
2. During installation, set a username and password for the database superuser (e.g., `postgres`).
3. After installation, ensure the PostgreSQL service is running.

### 2. Initialize the Database
1. Navigate to the `scripts` directory:
   ```bash
   cd scripts
   ```
2. Run the init.sql script to create the database and tables:
```bash
psql -U <username> -f init.sql
```
Replace `<username>` with your PostgreSQL username.

- Initialize tables using `scripts/init.sql` if needed.

### 3. Download the Model
- Download Code Llama 7B Instruct from Hugging Face (see their terms).
- Place the model files in `models/code-llama-7b-instruct/`.
- Use `scripts/download_model.py` for automated download if you have access.

### Testing the Database Connection
To test the database connection, run the following command:
```bash
python -m test.test_database
```

### 3. Test the Database Connection
1. Run the database test script:
   ```bash
   python -m test.test_database
   ```
2. The script will:
   - Connect to the database using the .env file.
   - List all tables in the database.
   - Fetch 10 sample records from each table.

### Downloading and Testing Models
1. Download the Code Llama Model
   1. Ensure you have access to the Code Llama model on Hugging Face:
      - Visit the Code Llama model page.
      - Request access and agree to the terms and conditions.
   2. Install the required dependencies:
      ```bash
      pip install transformers huggingface_hub
      ```
   3. Run the script to download the model:
      ```bash
      python scripts/download_model.py
      ```
   4. Verify the model is saved in the `code-llama-7b-instruct` directory.
2. Test the Downloaded Model
   1. Install additional dependencies:
      ```bash
      pip install torch>=2.0.0 transformers>=4.33.0
      ```
   2. Run the test script with a prompt:
      ```bash
      python -m test.test_model "Write a Python function to calculate factorial."
      ```
   3. The script will load the model and print the response to the given prompt.

### Testing the Model

Test the model's response to a prompt using:

```bash
python test/test_model.py "Your prompt here"
# Or from a file:
python test/test_model.py -f path/to/prompt_postgres.txt
```

You can also provide the prompt in a file:
```bash
python test/test_model.py -f path/to/prompt_postgres.txt
```

If no prompt is specified, the script will use a default example prompt.

### File Validation
Before loading the model, the script checks if all required files (`config.json`, `pytorch_model.bin`, `tokenizer.json`, `tokenizer_config.json`) exist in the `models/code-llama-7b-instruct` directory. If any file is missing, the script will raise an error and provide instructions to re-download the model using the `download_model.py` script.

### Downloading a Model for Offline Use

```bash
python scripts/download_model.py --repo_id <huggingface_repo_id> --save_path <local_path>
```

Both arguments are optional and default to CodeLlama-7b-Instruct-hf and `models/llama-2-7b`.

### Running the Streamlit App
To launch the Streamlit app for executing SQL queries:
```bash
streamlit run app.py
```

### Batch Command-Line Usage

You can generate SQL from a prompt via the command line:

```bash
python app_batch.py -p "List the top 5 customers by revenue"
```

### Project Structure
```
raven/
├── config/
│   └── .env                # Environment variables for database credentials
├── database.py             # Database connection and query execution
├── requirements.txt        # Python dependencies
├── scripts/
│   ├── init.sql            # SQL script for initializing the database
│   └── download_model.py   # Script to download AI models
├── test/
│   ├── __init__.py         # Makes the test directory a package
│   ├── test_database.py    # Script to test database functionality
│   └── test_model.py       # Script to test the downloaded model
└── README.md               # Project documentation
```

### Dependencies
The project uses the following Python libraries:

- `psycopg2>=2.9.0`: PostgreSQL database adapter for Python.
- `python-dotenv>=1.0.0`: For loading environment variables from .env files.
- `transformers>=4.33.0`: Hugging Face Transformers library.
- `torch>=2.0.0`: PyTorch for running AI models.

Install these dependencies using:
```bash
pip install -r requirements.txt
```

## Database Architecture

The application now uses an abstract base class design pattern for database connections:

- `BaseDatabase`: Abstract base class defining the common interface
- `Database`: PostgreSQL implementation
- `TDDatabase`: Teradata implementation

This design enables:
- Consistent interface across database types
- Easy addition of new database providers
- Context manager support (`with` statement)
- Standardized error handling

## Model Choice

The `code-llama-7b-instruct` model is a variant of the Code Llama series, which is fine-tuned specifically for **instruction-following tasks** in the domain of programming and code generation. It is particularly good at:

1. **Code Generation**: Generating code snippets or complete functions based on natural language prompts.
2. **Code Completion**: Completing partially written code or suggesting the next logical steps in a program.
3. **Code Explanation**: Explaining code snippets in natural language, making it useful for learning and debugging.
4. **Code Refactoring**: Suggesting improvements or refactoring existing code for better readability or performance.
5. **Multi-Language Support**: Supporting multiple programming languages, including Python, JavaScript, C++, and more.
6. **Instruction Following**: Responding to specific instructions related to coding tasks, such as "Write a function to calculate factorial" or "Explain this SQL query."

This model is ideal for developers, educators, and learners who need assistance with programming-related tasks.

The `code-llama-7b-instruct` model is proficient in SQL-related tasks, particularly when fine-tuned for instruction-following tasks. Its proficiency includes:

- **SQL Query Generation**: It can generate SQL queries from natural language prompts, making it useful for applications like natural language to SQL conversion.
- **SQL Query Explanation**: It can explain SQL queries in plain language, helping users understand complex queries.
- **SQL Query Debugging**: It can assist in identifying issues in SQL queries and suggest fixes.
- **SQL Query Optimization**: It can suggest improvements to SQL queries for better performance.

However, its proficiency depends on:

- The quality of the fine-tuning dataset (e.g., SQL-specific examples).
- The complexity of the SQL queries (e.g., simple SELECT statements vs. complex joins and subqueries).
- The clarity of the natural language prompt provided.

For advanced SQL tasks, additional fine-tuning or domain-specific training may be required.

## Model Loading

The application supports both GPU and CPU environments:

### GPU Mode
- Uses 8-bit quantization for memory efficiency
- Requires CUDA-compatible GPU
- Requires bitsandbytes library

### CPU Mode
- Uses optimized CPU loading without meta device
- Reduced precision is avoided on CPU
- Fallback mechanism for compatibility across PyTorch versions

### CPU Mode Configuration

When running on CPU, the application:
- Uses consistent float32 precision
- Implements memory optimization via offloading
- Requires more disk space for state dict offloading

### CUDA Compatibility

The application supports GPU acceleration with NVIDIA CUDA. If PyTorch can't detect your CUDA installation:

1. Run the CUDA compatibility fix script:
   ```bash
   python scripts/fix_cuda.py
   ```

### Known Issues
- Large language models may require significant memory (16GB+ RAM for CPU mode)
- CPU inference will be considerably slower than GPU

### Known Issues
- PyTorch path registration may conflict with Streamlit's file watcher
- Solution: File watcher is disabled by setting environment variables

## Prompt Template

The prompt template is stored in `prompt_postgres.txt` for consistency and easy maintenance. It is loaded dynamically by the `NL2SQL` class and supports parameterization with the following placeholders:
- `{limit}`: The maximum number of results to query
- `{schema}`: The database schema
- `{question}`: The natural language question

To modify the prompt, edit the `prompt_postgres.txt` file.

## References

- Using Natural Language to Query Postgres with Jacob
  https://www.youtube.com/watch?v=XNeTgVEzILg

- https://smith.langchain.com/hub/jacob/text-to-postgres-sql

- https://huggingface.co/docs/transformers/installation#offline-mode

- https://huggingface.co/meta-llama/CodeLlama-7b-Instruct-hf
- https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf
