import argparse
import os
import time
import streamlit as st
from typing import Any
from dotenv import load_dotenv


def get_database() -> Any:
    """
    Get or create database connection from session state based on environment variable.
    
    Returns:
        Database instance based on the DB_TYPE environment variable
    """
    if 'db' not in st.session_state:
        try:
            # Get database type from environment variable (default to postgres)
            db_type = os.environ.get('DB_TYPE', 'postgres').lower()
            
            # Initialize the appropriate database class
            if db_type == 'postgres':
                from db_postgres import PostgresDatabase
                db = PostgresDatabase()
            elif db_type == 'teradata':
                from db_td import TeradataDatabase
                db = TeradataDatabase()
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
                
            db.connect()
            st.session_state.db = db
        except Exception:
            raise
    return st.session_state.db


def main():
    """Main function to run the Streamlit app."""
    # Set page title and icon
    st.set_page_config(page_title="Select AI", page_icon="🤖")
    
    # Load environment variables
    env_path = os.path.join(os.path.dirname(__file__), 'config', '.env')
    load_dotenv(env_path)

    # Create sidebar with logo, title, about section
    with st.sidebar:
        # Logo in sidebar
        logo_path = os.path.join(os.path.dirname(__file__), "static", "img", "teradata_logo-transparent.png")
        if os.path.exists(logo_path):
            st.image(logo_path, width=200)
        
        # Horizontal separator
        st.markdown("---")
        
        # Title in sidebar
        st.markdown("# Select AI")
        
        # About section in sidebar
        st.markdown("""
            <p style='color: grey;'>
            Select AI is a natural language to SQL assistant.
            </p><p style='color: grey;'>
            Enter your query in natural language, the AI will generate 
            the corresponding SQL query and execute it on Teradata.
            </p>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # Main content area
    try:
        # Ensure database connection exists
        if 'db' not in st.session_state:
            db = get_database()
        else:
            db = st.session_state.db

        # Initialize NL2SQL with database connection - only once per session
        if 'nl2sql_converter' not in st.session_state:
            # Set environment variable to avoid meta tensor issues in Streamlit
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
            
            try:
                # Import torch first to ensure proper initialization
                import torch
                torch.cuda.empty_cache()
                
                # Get engine choice from environment (default to huggingface)
                engine = os.environ.get('ENGINE', 'huggingface').lower()
                
                # Initialize NL2SQL based on selected engine
                if engine == "ollama":
                    from nl2sql_ollama import NL2SQLOllama
                    nl2sql_converter = NL2SQLOllama(db)
                else:  # default to huggingface
                    from nl2sql import NL2SQL
                    nl2sql_converter = NL2SQL(db)
                
                st.session_state.nl2sql_converter = nl2sql_converter
            except Exception:
                raise
        
        nl2sql_converter = st.session_state.nl2sql_converter

        # Input text area for natural language query
        st.markdown("#### Natural Language Query")
        natural_language_query = st.text_area("", height=200)

        # Button to execute the query
        if st.button("Translate to SQL"):
            if natural_language_query.strip():
                try:
                    start_time = time.time()
                    sql_query = nl2sql_converter.nl2sql(natural_language_query)
                    end_time = time.time()
                    
                    # Calculate elapsed time
                    elapsed_seconds = end_time - start_time
                    hours, remainder = divmod(elapsed_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    time_format = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                    
                    # Display results
                    st.markdown("#### Generated SQL Query")
                    st.code(sql_query, language='sql', line_numbers=True)
                    print(f"Conversion time: {time_format}")

                    # Execute the SQL query and fetch results
                    results = db.execute_query(sql_query)
                        
                    if results:
                        st.markdown("#### Query Results")
                        st.dataframe(results)
                        
                except Exception as e:
                    # Reset database connection on error
                    if "connection" in str(e).lower():
                        st.session_state.pop('db', None)
                        db = get_database()
    except Exception:
        pass

if __name__ == "__main__":
    main()