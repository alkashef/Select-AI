import os
import time
import streamlit as st
from typing import Any
from dotenv import load_dotenv
from db import TeradataDatabase
from nl2sql import NL2SQL
import torch
from logger import logger  # Import the logger


def get_database() -> Any:
    """Get or create a database connection from the session state."""
    if 'db' not in st.session_state:
        try:
            db = TeradataDatabase()    
            db.connect()
            st.session_state.db = db
        except Exception:
            raise
    return st.session_state.db


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


def main():
    # Print a startup message with log file path
    print(f"Application started. All messages are being logged to: {logger.log_file_path}")
    
    st.set_page_config(page_title="Select AI", page_icon="🤖")
    
    env_path = os.path.join(os.path.dirname(__file__), 'config', '.env')
    load_dotenv(env_path)

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
    
    try:
        if 'db' not in st.session_state:
            db = get_database()
        else:
            db = st.session_state.db

        if 'nl2sql_converter' not in st.session_state:
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
            
            try:          
                torch.cuda.empty_cache()
                # Log the start of model loading
                device_type = "GPU" if torch.cuda.is_available() else "CPU"
                logger.log_model_loading("code-llama-7b-instruct", device_type, "started")
                
                nl2sql_converter = NL2SQL(db)
                st.session_state.nl2sql_converter = nl2sql_converter
                
                # Log the successful completion of model loading
                logger.log_model_loading("code-llama-7b-instruct", device_type, "completed")
            except Exception as e:
                # Log the failure of model loading
                device_type = "GPU" if torch.cuda.is_available() else "CPU"
                logger.log_model_loading("code-llama-7b-instruct", device_type, "failed", 
                                        {"error": str(e)})
                raise
        
        nl2sql_converter = st.session_state.nl2sql_converter

        st.markdown("#### Natural Language Query")
        natural_language_query = st.text_area("", height=200, label_visibility="collapsed")

        if st.button("Translate to SQL"):
            if natural_language_query.strip():
                try:
                    start_time = time.time()
                    sql_query, _ = nl2sql_converter.nl2sql(natural_language_query)
                    end_time = time.time()
                    
                    elapsed_seconds = end_time - start_time
                    hours, remainder = divmod(elapsed_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    time_format = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                    
                    # Store SQL query in session state for execution and display
                    st.session_state.current_sql_query = sql_query
                    st.session_state.show_sql = True
                    
                    # Log the translation time
                    logger.info(f"NL2SQL translation completed in {time_format} - Query: {natural_language_query[:50]}...")
                        
                except Exception as e:
                    if "connection" in str(e).lower():
                        st.session_state.pop('db', None)
                        db = get_database()
                    else:
                        st.error(f"Error translating query: {str(e)}")
                        # Log the error
                        logger.error(f"Translation error: {str(e)} - Query: {natural_language_query[:50]}...")
        
        # Always display SQL query if it exists in session state
        if st.session_state.get('show_sql') and st.session_state.get('current_sql_query'):
            st.markdown("#### Generated SQL Query", unsafe_allow_html=True)
            sql_editor = st.text_area("", value=st.session_state.current_sql_query, height=200, label_visibility="collapsed")
            st.session_state.current_sql_query = sql_editor 
            
        # Add Execute Query button
        if st.session_state.get('current_sql_query'):
            if st.button("Execute Query"):
                try:
                    sql_query = st.session_state.current_sql_query
                    
                    # Track execution time
                    start_time = time.time()
                    results = db.execute_query(sql_query)
                    end_time = time.time()
                    
                    # Calculate execution time
                    elapsed_seconds = end_time - start_time
                    hours, remainder = divmod(elapsed_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    time_format = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                    
                    # Log SQL execution time
                    logger.info(f"SQL execution completed in {time_format} - Query: {sql_query[:50]}...")
                    
                    if results is not None:
                        st.markdown("#### Query Results")
                        st.dataframe(results)
                        logger.info(f"Query execution completed in {time_format}")
                    else:
                        st.info("Query executed successfully but returned no results.")
                        logger.info(f"Query execution completed in {time_format}")
                except Exception as e:
                    if "connection" in str(e).lower():
                        st.session_state.pop('db', None)
                        db = get_database()
                        st.error("Database connection error. Please try again.")
                    else:
                        st.error(f"Error executing query: {str(e)}")
                        logger.error(f"SQL execution error: {str(e)} - Query: {sql_query[:50]}...")
        
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()