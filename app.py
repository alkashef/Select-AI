import os
import time
import streamlit as st
from typing import Any, Dict, Optional, Tuple
from dotenv import load_dotenv
from db import TeradataDatabase
from nl2sql import NL2SQL
import torch
from logger import logger


def initialize_database() -> Optional[TeradataDatabase]:
    """
    Initialize database connection once during app startup.
    
    Returns:
        TeradataDatabase: Connected database instance or None if connection fails
    """
    try:
        logger.info("Initializing database connection at app startup")
        db = TeradataDatabase()    
        db.connect()
        # Cache the schema during initialization
        db.get_schema()
        logger.info("Database connection established and schema cached at startup")
        return db
    except Exception as e:
        logger.error(f"Failed to initialize database at startup: {str(e)}")
        return None


def get_log_contents() -> str:
    """Read and return the contents of the current log file.
    
    Returns:
        str: Contents of the log file or error message if file cannot be read
    """
    try:
        log_file_path = logger.log_file_path
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r') as f:
                return f.read()
        return "No log file found"
    except Exception as e:
        return f"Error reading log file: {str(e)}"


def perform_model_warmup(nl2sql_converter: NL2SQL) -> None:
    """
    Performs a warmup inference to initialize model weights and improve initial response time.
    
    Args:
        nl2sql_converter: The NL2SQL instance to warm up
    """
    warmup_query = "how many branches are in texas?"
    logger.info(f"Performing model warmup with query: '{warmup_query}'")
    
    with st.spinner("Warming up model..."):
        try:
            start_time = time.time()
            warmup_sql, _ = nl2sql_converter.nl2sql(warmup_query)
            end_time = time.time()
            warmup_time = end_time - start_time
            logger.info(f"Model warmup completed in {warmup_time:.2f} seconds. Generated SQL: {warmup_sql}")
        except Exception as e:
            logger.warning(f"Model warmup failed: {str(e)}. Continuing with initialization.")


def initialize_session_state() -> None:
    """Initialize the Streamlit session state variables if they don't exist."""
    if 'current_sql_query' not in st.session_state:
        st.session_state.current_sql_query = None
    if 'show_sql' not in st.session_state:
        st.session_state.show_sql = False


def setup_page_config() -> None:
    """Configure the Streamlit page settings."""
    st.set_page_config(page_title="Select AI", page_icon="🤖")
    
    env_path = os.path.join(os.path.dirname(__file__), 'config', '.env')
    load_dotenv(env_path)


def render_sidebar() -> None:
    """Render the sidebar content with logo and application description."""
    with st.sidebar:
        logo_path = os.path.join(os.path.dirname(__file__), "img", "td_new_trans.png")
        if os.path.exists(logo_path):
            st.image(logo_path)
        
        st.markdown("---")
        
        st.markdown("# Select AI")
        
        st.markdown("""
            <p style='color: grey;'>
            Select AI is a natural language to SQL assistant.
            </p><p style='color: grey;'>
            Enter your query in natural language, the AI will generate 
            the corresponding SQL query and execute it on Teradata.
            </p>
            """, unsafe_allow_html=True)
        
        st.markdown("---")


def check_database_connection() -> bool:
    """
    Check if the database connection is established.
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    if 'db' not in st.session_state:
        st.session_state.db = initialize_database()
    
    db = st.session_state.db
    db_status = db is not None
    
    st.sidebar.markdown("#### Connection Status")
    if db_status:
        st.sidebar.markdown(f"<span style='color:grey; font-size:small;'>Connected to {db.database} at {db.host}</span>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"<span style='color:grey; font-size:small;'>database connection failed</span>", unsafe_allow_html=True)
    
    if not db_status:
        st.error("Database connection failed. Please check your configuration and restart the application.")
    
    return db_status


def initialize_model() -> Optional[NL2SQL]:
    """
    Initialize the NL2SQL model if not already initialized.
    
    Returns:
        Optional[NL2SQL]: Initialized model or None if initialization fails
    """
    if 'nl2sql_converter' in st.session_state:
        return st.session_state.nl2sql_converter
        
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    
    try:          
        torch.cuda.empty_cache()
        device_type = "GPU" if torch.cuda.is_available() else "CPU"
        logger.log_model_loading("code-llama-7b-instruct", device_type, "started")
        
        nl2sql_converter = NL2SQL(st.session_state.db)
        
        # Call the separate warmup method
        perform_model_warmup(nl2sql_converter)
        
        st.session_state.nl2sql_converter = nl2sql_converter
        
        logger.log_model_loading("code-llama-7b-instruct", device_type, "completed")
        return nl2sql_converter
    except Exception as e:
        device_type = "GPU" if torch.cuda.is_available() else "CPU"
        logger.log_model_loading("code-llama-7b-instruct", device_type, "failed", 
                                {"error": str(e)})
        st.error(f"Failed to load model: {str(e)}")
        return None


def format_time(elapsed_seconds: float) -> str:
    """
    Format seconds into hours:minutes:seconds format.
    
    Args:
        elapsed_seconds: Time in seconds
        
    Returns:
        str: Formatted time string in HH:MM:SS format
    """
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def translate_query(nl2sql_converter: NL2SQL, natural_language_query: str) -> None:
    """
    Translate a natural language query to SQL.
    
    Args:
        nl2sql_converter: The model to use for translation
        natural_language_query: The natural language query to translate
    """
    if not natural_language_query.strip():
        return
        
    try:
        start_time = time.time()
        sql_query, _ = nl2sql_converter.nl2sql(natural_language_query)
        end_time = time.time()
        
        elapsed_seconds = end_time - start_time
        time_format = format_time(elapsed_seconds)
        
        # Store SQL query in session state for execution and display
        st.session_state.current_sql_query = sql_query
        st.session_state.show_sql = True
        
        # Log the translation time
        logger.info(f"NL2SQL translation completed in {time_format} - Query: {natural_language_query[:50]}...")
            
    except Exception as e:
        st.error(f"Error translating query: {str(e)}")
        logger.error(f"Translation error: {str(e)} - Query: {natural_language_query[:50]}...")


def display_sql_query() -> None:
    """Display the generated SQL query if available in session state."""
    if st.session_state.get('show_sql') and st.session_state.get('current_sql_query'):
        st.markdown("#### Generated SQL Query", unsafe_allow_html=True)
        sql_editor = st.text_area("", value=st.session_state.current_sql_query, height=200, label_visibility="collapsed")
        st.session_state.current_sql_query = sql_editor


def execute_sql_query() -> None:
    """Execute the current SQL query stored in the session state."""
    if not st.session_state.get('current_sql_query'):
        return
        
    try:
        sql_query = st.session_state.current_sql_query
        
        start_time = time.time()
        results = st.session_state.db.execute_query(sql_query)
        end_time = time.time()
        
        elapsed_seconds = end_time - start_time
        time_format = format_time(elapsed_seconds)
        
        logger.info(f"SQL execution completed in {time_format} - Query: {sql_query[:50]}...")
        
        if results is not None:
            st.markdown("#### Query Results")
            st.dataframe(results)
        else:
            st.info("Query executed successfully but returned no results.")
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        logger.error(f"SQL execution error: {str(e)} - Query: {sql_query[:50]}...")


def display_model_status() -> None:
    """
    Display model status in the sidebar, showing model name and whether it's loaded on GPU or CPU.
    """
    if 'nl2sql_converter' in st.session_state:
        model_name = "code-llama-7b-instruct"
        device_type = "GPU" if torch.cuda.is_available() else "CPU"
        
        st.sidebar.markdown("#### Model Status")
        st.sidebar.markdown(f"<span style='color:grey; font-size:small;'>Model {model_name} is loaded in {device_type}</span>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown("#### Model Status")
        st.sidebar.markdown(f"<span style='color:grey; font-size:small;'>model not loaded</span>", unsafe_allow_html=True)


def render_user_interface(nl2sql_converter: NL2SQL) -> None:
    """
    Render the main user interface components.
    
    Args:
        nl2sql_converter: The initialized NL2SQL model
    """
    st.markdown("#### Natural Language Query")
    natural_language_query = st.text_area("", height=200, label_visibility="collapsed")

    if st.button("Translate to SQL"):
        translate_query(nl2sql_converter, natural_language_query)
    
    display_sql_query()
    
    if st.session_state.get('current_sql_query'):
        if st.button("Execute Query"):
            execute_sql_query()


def main() -> None:
    """Main application entry point."""
    print(f"Application started. All messages are being logged to: {logger.log_file_path}")
    
    setup_page_config()
    initialize_session_state()
    render_sidebar()
    
    if not check_database_connection():
        return

    nl2sql_converter = initialize_model()
    display_model_status()
    
    if nl2sql_converter is None:
        return
        
    render_user_interface(nl2sql_converter)


if __name__ == "__main__":
    main()