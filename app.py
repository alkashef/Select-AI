import os
import time
import streamlit as st
from typing import Any
from dotenv import load_dotenv
from db import TeradataDatabase
from nl2sql import NL2SQL
import torch

def get_database() -> Any:
    if 'db' not in st.session_state:
        try:
            db = TeradataDatabase()    
            db.connect()
            st.session_state.db = db
        except Exception:
            raise
    return st.session_state.db


def main():
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
                nl2sql_converter = NL2SQL(db)
                st.session_state.nl2sql_converter = nl2sql_converter
            except Exception:
                raise
        
        nl2sql_converter = st.session_state.nl2sql_converter

        st.markdown("#### Natural Language Query")
        natural_language_query = st.text_area("", height=200)

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
                    
                    # Display execution time
                    st.text(f"Translation time: {time_format}")
                        
                except Exception as e:
                    if "connection" in str(e).lower():
                        st.session_state.pop('db', None)
                        db = get_database()
                    else:
                        st.error(f"Error translating query: {str(e)}")
        
        # Always display SQL query if it exists in session state
        if st.session_state.get('show_sql') and st.session_state.get('current_sql_query'):
            st.markdown("#### Generated SQL Query")
            sql_editor = st.text_area("", value=st.session_state.current_sql_query, height=200)
            st.session_state.current_sql_query = sql_editor 
            
        # Add Execute Query button
        if st.session_state.get('current_sql_query'):
            if st.button("Execute Query"):
                try:
                    sql_query = st.session_state.current_sql_query
                    results = db.execute_query(sql_query)
                    
                    if results is not None:
                        st.markdown("#### Query Results")
                        st.dataframe(results)
                    else:
                        st.info("Query executed successfully but returned no results.")
                except Exception as e:
                    if "connection" in str(e).lower():
                        st.session_state.pop('db', None)
                        db = get_database()
                        st.error("Database connection error. Please try again.")
                    else:
                        st.error(f"Error executing query: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()