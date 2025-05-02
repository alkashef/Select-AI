import os
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from db import TeradataDatabase
from dotenv import load_dotenv
import re
from logger import logger  # Import the logger

class NL2SQL:
    def __init__(self, db: TeradataDatabase, model_path: str = None):
        load_dotenv(os.path.join(os.path.dirname(__file__), 'config', '.env'))
        
        self.model_path = model_path or os.getenv('MODEL_PATH', "./models/code-llama-7b-instruct")
        logger.info(f"Initializing NL2SQL with model path: {self.model_path}")
        
        self.max_length = int(os.getenv('MAX_LENGTH', 4096))
        self.input_context_length = int(os.getenv('INPUT_CONTEXT_LENGTH', 2048))
        self.token_generation_limit = int(os.getenv('TOKEN_GENERATION_LIMIT', 2048))
        self.result_limit = int(os.getenv('RESULT_LIMIT', 5))
        
        self.db = db
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        
        self.prompt_template = self._load_prompt_template()

        self._load_model()

    def _load_prompt_template(self) -> str:
        prompt_file_path = os.environ.get('PROMPT_PATH')
        logger.info(f"Loading prompt template from: {prompt_file_path}")
        if not prompt_file_path:
            logger.error("PROMPT_PATH environment variable is not set")
            raise ValueError("PROMPT_PATH environment variable is not set.")
        with open(prompt_file_path, 'r', encoding='utf-8') as file:
            template = file.read()
            logger.debug("Prompt template loaded successfully")
            return template

    def _load_model(self) -> None:
        logger.info(f"Loading tokenizer from: {self.model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.debug("Set pad_token to eos_token")
            
            logger.info("Tokenizer loaded successfully, proceeding with model loading")
            
            if torch.cuda.is_available():
                logger.info("Loading model on GPU")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    max_memory={0: "6GiB", "cpu": "16GiB"},
                    offload_folder="offload",
                    offload_state_dict=True,
                    local_files_only=True
                )
                logger.info("Model loaded successfully on GPU")
            else:
                logger.info("Loading model on CPU (GPU not available)")
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                        offload_folder="offload",
                        offload_state_dict=True,
                        local_files_only=True
                    )
                    logger.info("Model loaded successfully on CPU")
                except Exception as e:
                    logger.error(f"Error loading model: {e}")
                    raise
        except Exception as e:
            logger.error(f"Error in model loading process: {e}")
            raise

    def _generate_prompt(self, question: str) -> str:
        logger.debug(f"Generating prompt for question: {question[:50]}...")
        schema = self.db.get_schema()
        logger.debug(f"Retrieved database schema with {len(schema.split())} tokens")
        return self.prompt_template.format(
            limit=self.result_limit,
            schema=schema,
            question=question
        )

    def _extract_sql(self, text: str) -> str:
        logger.debug("Extracting SQL from model output")
        START_MARKER = 'SQLQuery:'
        TEMPLATE_TEXT = '<SQL Query to run>'
        END = ';'
        
        try:
            template_pos = text.find(TEMPLATE_TEXT)
            
            if template_pos != -1:
                search_start = template_pos + len(TEMPLATE_TEXT)
                start_pos = text.find(START_MARKER, search_start)
            else:
                start_pos = text.find(START_MARKER)
                
            if start_pos == -1:
                logger.error("No SQL query marker found in output")
                raise ValueError("No SQL query found in output")
            
            sql_start = start_pos + len(START_MARKER)
            
            sql_end = text.find(END, sql_start)
            if sql_end == -1:
                if isinstance(self.db, TeradataDatabase):
                    logger.debug("No semicolon found, appending one and retrying")
                    modified_text = text + ";"
                    return self._extract_sql(modified_text)
                logger.error("No query terminator found")
                raise ValueError("No query terminator found")
            
            sql_query = text[sql_start:sql_end + 1].strip()
            
            if TEMPLATE_TEXT in sql_query:
                logger.error("Extracted template placeholder instead of valid SQL")
                raise ValueError("Extracted template placeholder instead of valid SQL")
            
            if isinstance(self.db, TeradataDatabase):
                logger.debug("Running Teradata-specific post-processing")
                sql_query = self._td_postprocessing(sql_query)
                
            logger.debug(f"Successfully extracted SQL: {sql_query[:50]}...")
            return sql_query
            
        except Exception as e:
            logger.error(f"Failed to extract SQL query: {str(e)}")
            raise ValueError(f"Failed to extract SQL query: {str(e)}")

    def _td_postprocessing(self, sql_query: str) -> str:
        logger.debug("Applying Teradata post-processing to SQL query")
        db_name = self.db.db_name if hasattr(self.db, "db_name") else "raven"
        
        table_pattern = r'(?i)(FROM|JOIN)\s+([a-zA-Z0-9_]+)(?!\s*\.|[a-zA-Z0-9_])'
        
        def qualify_table(match):
            keyword = match.group(1)
            table = match.group(2)
            return f"{keyword} {db_name}.{table}"
        
        sql_query = re.sub(table_pattern, qualify_table, sql_query)
        
        limit_pattern = r'(?i)\s+LIMIT\s+\d+\s*;?$'
        sql_query = re.sub(limit_pattern, ";", sql_query)
        
        if not sql_query.strip().endswith(';'):
            sql_query += ";"
        
        logger.debug("Teradata post-processing completed")
        return sql_query

    def nl2sql(self, prompt: str) -> Tuple[str, str]:
        logger.info(f"Processing natural language query: {prompt[:50]}...")
        if not self.model or not self.tokenizer:
            logger.error("Model not loaded")
            raise RuntimeError("Model not loaded.")

        logger.debug("Generating formatted prompt")
        formatted_prompt = self._generate_prompt(prompt)

        logger.debug("Tokenizing input")
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.input_context_length
        )
        
        if torch.cuda.is_available():
            logger.debug("Moving inputs to GPU")
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        logger.info("Generating SQL with model")
        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=self.token_generation_limit,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,
            temperature=1.0,
            num_beams=1
        )
        
        logger.debug("Decoding model output")
        model_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        raw_sql = model_output[0]
        
        logger.debug("Extracting SQL query from raw output")
        sql_query = self._extract_sql(raw_sql)
        
        logger.info(f"Successfully generated SQL query: {sql_query[:50]}...")
        return sql_query, raw_sql
