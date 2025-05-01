import os
from dotenv import load_dotenv
import teradataml as tdml
from teradataml import execute_sql
from typing import Any, List, Dict, Optional, Union


class TeradataDatabase():

    def __init__(self):
        """
        Initialize the TeradataDatabase class with connection parameters from the .env file.
        """
        # Load environment variables from the .env file in the config directory
        env_path = os.path.join(os.path.dirname(__file__), 'config', '.env')
        load_dotenv(env_path)

        self.host = os.getenv("TD_HOST")
        self.database = os.getenv("TD_NAME")
        self.user = os.getenv("TD_USER")
        self.password = os.getenv("TD_PASSWORD")
        self.port = os.getenv("TD_PORT", 1025)
        self.connection = None

    def connect(self) -> None:
        """
        Establish a connection to the Teradata database.
        
        Raises:
            tdml.OperationalError: If connection fails.
        """
        try:
            self.connection = tdml.create_context(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
            )
            print(f"Database connection established to {self.database} at {self.host}.")
        except tdml.OperationalError as e:
            print(f"Error connecting to the database: {e}")
            raise

    def disconnect(self) -> None:
        """
        Close the database connection.
        """
        if self.connection:
            tdml.remove_context()
            self.connection = None
            print("Database connection closed.")

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return the results with column names.
        
        Args:
            query: SQL query to execute
            
        Returns:
            List[Dict[str, Any]]: Query results with proper column names
            
        Raises:
            Exception: If query execution fails
        """
        if not self.connection:
            raise Exception("Database connection is not established. Call connect() first.")
        
        try:
            result = execute_sql(query)
            
            # Get column names from cursor description
            columns = [desc[0] for desc in result.description] if result.description else []
            
            # Convert results to a list of dictionaries with proper column names
            rows = result.fetchall()
            results = [
                {columns[i]: value for i, value in enumerate(row)}
                for row in rows
            ]
            
            return results
        except Exception as e:
            print(f"Query execution failed: {e}")
            raise Exception(f"Query execution failed: {str(e)}")

    def _get_sample_data(self, table: str) -> str:
        """
        Retrieve sample data from the specified table with compact formatting.

        Args:
            table: The name of the table to retrieve sample data from.

        Returns:
            str: A formatted string representation of the sample data.
        """
        query = f"SELECT * FROM {table} SAMPLE 3;"
        try:
            result = execute_sql(query)
            if result.rowcount == 0:
                return "No data available"
                
            rows = result.fetchall()
            columns = [col[0] for col in result.description]
            
            # Format columns row
            result = [f"columns: {', '.join(columns)}"]
            
            # Format each data row
            for i, row in enumerate(rows):
                # Convert all values to strings and handle None values
                row_values = [str(val) if val is not None else "NULL" for val in row]
                result.append(f"row{i+1}: {', '.join(row_values)}")
            
            return "\n".join(result)
        except tdml.OperationalError as e:
            print(f"Error retrieving sample data: {e}")
            return "Error retrieving sample data"

    def get_schema(self) -> str:
        """
        Extract the schema of the database in text format with sample data.

        Returns:
            str: A string representation of the database schema with sample data.
            
        Raises:
            Exception: If database connection is not established.
            tdml.OperationalError: If schema retrieval fails.
        """
        if not self.connection:
            raise Exception("Database connection is not established. Call connect() first.")

        query = """
        SELECT t.tablename, c.columnname, c.columntype
        FROM dbc.tablesv t
        JOIN dbc.columnsv c 
        ON t.tablename = c.tablename AND t.databasename = c.databasename
        WHERE t.databasename = 'raven'
        AND t.TableKind = 'T'
        ORDER BY t.tablename
        """

        # Mapping for Teradata types to more readable formats
        td_type_map = {
            'CV': 'String',
            'D': 'Numeric',
            'CF': 'Numeric',
            'I': 'Integer',
            'F': 'Float',
        }

        try:
            result = execute_sql(query)
            if result.rowcount == 0:
                print("No tables found in the database 'raven'.")
                raise
            rows = result.fetchall()
            schema = {}
            for row in rows:
                table_name = row[0]
                column_name = row[1]
                data_type = row[2]
                #print(table_name, column_name, data_type)
                if table_name not in schema:
                    schema[table_name] = []
                schema[table_name].append(f"{column_name} ({td_type_map[data_type.strip()]})")

            # Format schema as a text string with sample data
            schema_parts = []
            for table, columns in schema.items():
                table_schema = f"\n\nTable: {table}\nColumns:\n  - " + "\n  - ".join(columns)
                
                # Add sample data section
                sample_data = self._get_sample_data(table)
                table_schema += f"\n\nSample data:\n{sample_data}" if sample_data else "\n\nSample data: No data available"
                
                schema_parts.append(table_schema)
                
            return "\n".join(schema_parts)
        except tdml.OperationalError as e:
            print(f"Error retrieving schema: {e}")
            raise
    
    def _format_td_type(self, type_code: str, length: int, total_digits: int, fractional_digits: int) -> str:
        """
        Format Teradata data type.
        
        Args:
            type_code: Teradata type code.
            length: Column length.
            total_digits: Total digits for numeric types.
            fractional_digits: Fractional digits for numeric types.
            
        Returns:
            str: Formatted type string.
        """
        type_code = type_code.strip()
        if type_code == 'CV':
            return f"VARCHAR({length})"
        elif type_code == 'CF':
            return f"CHAR({length})"
        elif type_code == 'D':
            return "DATE"
        elif type_code == 'I':
            return "INTEGER"
        elif type_code == 'F':
            return f"DECIMAL({total_digits}, {fractional_digits})"
        else:
            return type_code

    def __enter__(self):
        """
        Enter the runtime context and establish a database connection.
        """
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the runtime context and close the database connection.
        """
        self.disconnect()
