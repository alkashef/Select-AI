import os
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from db import TeradataDatabase
from dotenv import load_dotenv
import re

class NL2SQL:
    """Natural Language to SQL converter using LLaMA model."""
    
    def __init__(self, db: TeradataDatabase, model_path: str = None):
        """
        Initialize NL2SQL converter with database connection and model.
        
        Args:
            db: Database connection instance
            model_path: Optional path to model, defaults to env value
        """
        # Load environment variables
        load_dotenv(os.path.join(os.path.dirname(__file__), 'config', '.env'))
        
        # Initialize configuration
        self.model_path = model_path or os.getenv('MODEL_PATH', "./models/code-llama-7b-instruct")
        self.max_length = int(os.getenv('MAX_LENGTH', 4096))
        self.input_context_length = int(os.getenv('INPUT_CONTEXT_LENGTH', 2048))
        self.token_generation_limit = int(os.getenv('TOKEN_GENERATION_LIMIT', 2048))
        self.result_limit = int(os.getenv('RESULT_LIMIT', 5))
        
        # Initialize model components
        self.db = db
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        
        # Load prompt template from file
        self.prompt_template = self._load_prompt_template()

        self._load_model()

    def _load_prompt_template(self) -> str:
        """
        Load the prompt template from the path specified in the PROMPT_PATH environment variable.
        """
        prompt_file_path = os.environ.get('PROMPT_PATH')
        if not prompt_file_path:
            raise ValueError("PROMPT_PATH environment variable is not set.")
        with open(prompt_file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def _load_model(self) -> None:
        """
        Load model and tokenizer with 4-bit quantization optimized for limited VRAM.
        
        Uses CPU-compatible configuration when CUDA is not available.
        """
        print(f"Loading model from: {self.model_path}")
        
        # Load tokenizer with padding token if not set
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Check CUDA availability
        if torch.cuda.is_available():
            # Optimized for 8GB VRAM with 4-bit quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,              # Use half precision
                device_map="auto",                      # Let the library handle memory mapping
                load_in_4bit=True,                      # Use 4-bit quantization (more memory efficient)
                bnb_4bit_compute_dtype=torch.float16,   # Compute dtype for 4-bit quantization
                bnb_4bit_use_double_quant=True,         # Use nested quantization for better memory efficiency
                bnb_4bit_quant_type="nf4",              # Use 4-bit NormalFloat (NF4) format
                max_memory={0: "6GiB", "cpu": "16GiB"}, # Memory allocation between GPU and CPU
                offload_folder="offload",               # Enable disk offloading for large components
                offload_state_dict=True,
                local_files_only=True
            )
            print("Model loaded on GPU with 4-bit quantization")
        else:
            # CPU configuration
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    offload_folder="offload",
                    offload_state_dict=True,
                    local_files_only=True
                )
                print("Model loaded successfully on CPU")
            except Exception as e:
                print(f"Error loading model: {e}")
                raise

    def _generate_prompt(self, question: str) -> str:
        """
        Generate the prompt by formatting the template with parameters.
        
        Args:
            question: The natural language question to convert to SQL.
        
        Returns:
            str: The formatted prompt.
        """
        schema = self.db.get_schema()
        return self.prompt_template.format(
            limit=self.result_limit,
            schema=schema,
            question=question
        )

    def _extract_sql_old(self, text: str) -> str:
        """
        Extract SQL query from model output text.
        
        Args:
            text: The full text output from the model
            
        Returns:
            str: Extracted and formatted SQL query
            
        Raises:
            ValueError: If SQL query cannot be extracted
        """
        START_MARKER = 'SQLQuery:'
        TEMPLATE_TEXT = '<SQL Query to run>'
        END = ';'
        
        try:
            # Find the template text position
            template_pos = text.find(TEMPLATE_TEXT)
            
            # Find the start of SQL query after the template text
            if template_pos != -1:
                search_start = template_pos + len(TEMPLATE_TEXT)
                start_pos = text.find(START_MARKER, search_start)
            else:
                # If template text not found, use the first occurrence of START_MARKER
                start_pos = text.find(START_MARKER)
                
            if start_pos == -1:
                raise ValueError("No SQL query found in output")
            
            sql_start = start_pos + len(START_MARKER)
            
            # Find the end of SQL query (first semicolon after sql_start)
            sql_end = text.find(END, sql_start)
            if sql_end == -1:
                # For Teradata: try appending a semicolon if extraction fails
                if isinstance(self.db, TeradataDatabase):
                    modified_text = text + ";"
                    # Try extraction again with modified text
                    return self._extract_sql(modified_text)
                raise ValueError("No query terminator found")
            
            # Extract and clean the SQL query
            sql_query = text[sql_start:sql_end + 1].strip()
            
            # Convert LIMIT clause to TOP clause
            limit_pattern = r'(?i)\s+LIMIT\s+(\d+)\s*;?$'
            limit_match = re.search(limit_pattern, sql_query)
            
            if limit_match:
                # Extract the limit value
                limit_value = limit_match.group(1)
                
                # Remove the LIMIT clause
                sql_query = re.sub(limit_pattern, ";", sql_query)
                
                # Add TOP clause only to the first SELECT statement
                first_select_pattern = r'(?i)SELECT\s+'
                # Find first occurrence of SELECT
                match = re.search(first_select_pattern, sql_query)
                if match:
                    pos = match.start()
                    # Split query into before and after first SELECT
                    before = sql_query[:pos]
                    after = sql_query[pos:]
                    # Replace only the first SELECT
                    after = re.sub(first_select_pattern, f"SELECT TOP {limit_value} ", after, count=1)
                    sql_query = before + after
            
            # Add newlines before specific keywords (case insensitive)
            keywords = ['from', 'inner join', 'left join', 'right join', 'full join', 'group', 'order', 'where', 'having']
            for keyword in keywords:
                # This regex adds newline before the keyword if not already preceded by one
                pattern = re.compile(f'(?<!\n)\\s+({keyword}\\s+)', re.IGNORECASE)
                sql_query = re.sub(pattern, f'\n\\1', sql_query)
            
            # Normalize whitespace but preserve newlines
            # 1. Replace consecutive spaces/tabs with a single space
            sql_query = re.sub(r'[ \t]+', ' ', sql_query)  # Only replace spaces and tabs, not newlines
            # 2. Trim spaces at the beginning of lines
            sql_query = re.sub(r'\n\s+', '\n', sql_query)
            
            return sql_query
        except Exception as e:
            print(f"Error extracting SQL: {str(e)}")
            raise ValueError(f"Failed to extract SQL query: {str(e)}")

    def _extract_sql(self, text: str) -> str:
        """
        Extract SQL query from model output.
        Extracts from the first 'SQLQuery:' that comes after '<SQL Query to run>' 
        to the first semicolon.
        
        Args:
            text: Raw text output from the model
            
        Returns:
            str: Extracted SQL query
            
        Raises:
            ValueError: If SQL query cannot be extracted
        """
        START_MARKER = 'SQLQuery:'
        TEMPLATE_TEXT = '<SQL Query to run>'
        END = ';'
        
        try:
            # Find the template text position
            template_pos = text.find(TEMPLATE_TEXT)
            
            # Find the start of SQL query after the template text
            if template_pos != -1:
                search_start = template_pos + len(TEMPLATE_TEXT)
                start_pos = text.find(START_MARKER, search_start)
            else:
                # If template text not found, use the first occurrence of START_MARKER
                start_pos = text.find(START_MARKER)
                
            if start_pos == -1:
                raise ValueError("No SQL query found in output")
            
            sql_start = start_pos + len(START_MARKER)
            
            # Find the end of SQL query (first semicolon after sql_start)
            sql_end = text.find(END, sql_start)
            if sql_end == -1:
                # For Teradata: try appending a semicolon if extraction fails
                if isinstance(self.db, TeradataDatabase):
                    modified_text = text + ";"
                    # Try extraction again with modified text
                    return self._extract_sql(modified_text)
                raise ValueError("No query terminator found")
            
            # Extract and clean the SQL query
            sql_query = text[sql_start:sql_end + 1].strip()
            
            # Validate that we didn't extract another template
            if TEMPLATE_TEXT in sql_query:
                raise ValueError("Extracted template placeholder instead of valid SQL")
            
            # Apply Teradata-specific post-processing if needed
            if isinstance(self.db, TeradataDatabase):
                sql_query = self._td_postprocessing(sql_query)
                
            return sql_query
            
        except Exception as e:
            raise ValueError(f"Failed to extract SQL query: {str(e)}")

    def _td_postprocessing(self, sql_query: str) -> str:
        """
        Apply Teradata-specific modifications to extracted SQL.
        
        Args:
            sql_query: The extracted SQL query
            
        Returns:
            str: Modified SQL query for Teradata
        """
        # Get database name for qualification
        db_name = self.db.db_name if hasattr(self.db, "db_name") else "raven"
        
        # Replace unqualified table names with qualified ones
        # This pattern finds table references after FROM or JOIN keywords
        table_pattern = r'(?i)(FROM|JOIN)\s+([a-zA-Z0-9_]+)(?!\s*\.|[a-zA-Z0-9_])'
        
        def qualify_table(match):
            keyword = match.group(1)
            table = match.group(2)
            return f"{keyword} {db_name}.{table}"  # Return the formatted string
        
        sql_query = re.sub(table_pattern, qualify_table, sql_query)
        
        # Remove LIMIT clause if present
        limit_pattern = r'(?i)\s+LIMIT\s+\d+\s*;?$'
        sql_query = re.sub(limit_pattern, ";", sql_query)
        
        # Ensure the query ends with a semicolon
        if not sql_query.strip().endswith(';'):
            sql_query += ";"
            
        return sql_query

    def nl2sql(self, prompt: str) -> tuple[str, str]:
        """
        Convert natural language to SQL using the model with memory-efficient settings.
        
        Args:
            prompt: Natural language query
        
        Returns:
            tuple[str, str]: (processed SQL query, raw SQL query)
        
        Raises:
            RuntimeError: If model is not loaded
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded.")

        formatted_prompt = self._generate_prompt(prompt)

        # Use INPUT_CONTEXT_LENGTH for input tokenization
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.input_context_length
        )
        
        # Move all input tensors to the same device
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Use TOKEN_GENERATION_LIMIT for output generation
        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=self.token_generation_limit,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,                     # Use greedy decoding to save memory
            temperature=1.0,                     # No temperature scaling 
            num_beams=1                          # No beam search to save memory
        )
        
        model_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        raw_sql = model_output[0]
        
        print(f"\nRaw SQL:\n{raw_sql}")
        sql_query = self._extract_sql(raw_sql)
        print(f"SQL after Post-processing:\n{sql_query}")
        
        return sql_query, raw_sql
