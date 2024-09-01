from config import get_postgres_keys
import psycopg2
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote

def connect_to_postgres_sqlalchemy():
    """
    Connect to PostgreSQL using SQLAlchemy and return the connection engine.
    """
    pg_port, pg_username, pg_host, pg_database_name, postgres_superuser_password = get_postgres_keys()

    # URL-encode the password to handle special characters
    encoded_password = quote(postgres_superuser_password)
    
    try:
        # Create the connection string for SQLAlchemy, using the encoded password
        connection_string = f"postgresql://{pg_username}:{encoded_password}@{pg_host}:{pg_port}/{pg_database_name}"
        print(f"Connection string: {connection_string}")  # Debugging step
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        return None

def connect_to_postgres():
    """
    Connect to PostgreSQL and return connection and cursor.
    """
    pg_port, pg_username, pg_host, pg_database_name, postgres_superuser_password = get_postgres_keys()

    try:
        conn = psycopg2.connect(
            host=pg_host,
            database=pg_database_name,
            user=pg_username,
            password=postgres_superuser_password,
            port=pg_port
        )
        cursor = conn.cursor()
        return conn, cursor
    except Exception as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        return None, None
    
def load_data_from_db(query):
    """
    Load data from PostgreSQL database based on a query.
    Args:
        query (str): SQL query to load data.
    Returns:
        pd.DataFrame: Data from the SQL query.
    """
    conn, cursor = connect_to_postgres()
    
    if conn is None or cursor is None:
        return None
    
    try:
        # Load data into a DataFrame
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
    finally:
        # Close connection
        cursor.close()
        conn.close()

def save_df_to_db(df, table_name, mode='concat'):
    """
    Save a DataFrame to a PostgreSQL database table using SQLAlchemy. Allows replacing or concatenating.
    
    Args:
        df (pd.DataFrame): DataFrame to be saved.
        table_name (str): The name of the table in the database.
        mode (str): 'replace' to replace the table, 'concat' to append the data to the table.
    """
    engine = connect_to_postgres_sqlalchemy()
    
    if engine is None:
        return
    
    try:
        with engine.connect() as conn:
            # Check if the table exists
            check_query = text(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table_name}');")
            table_exists = conn.execute(check_query).scalar()

            if mode == 'replace':
                df.to_sql(table_name, engine, if_exists='replace', index=False)
                print(f"Table {table_name} has been replaced.")
            
            elif mode == 'concat':
                if table_exists:
                    # Load the existing table and concatenate with the new DataFrame
                    existing_df = pd.read_sql_table(table_name, engine)
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    combined_df.to_sql(table_name, engine, if_exists='replace', index=False)
                    print(f"Data has been concatenated to the existing table {table_name}.")
                else:
                    df.to_sql(table_name, engine, if_exists='replace', index=False)
                    print(f"Table {table_name} created and data saved.")

    except Exception as e:
        print(f"Error saving data: {e}")
        
def create_table_if_not_exists(cursor, table_name, df):
    """
    Create a table in PostgreSQL if it doesn't already exist, based on the DataFrame structure.
    
    Args:
        cursor: The cursor for the PostgreSQL connection.
        table_name (str): The name of the table to be created.
        df (pd.DataFrame): DataFrame to determine the structure of the table.
    """
    columns = []
    for col, dtype in df.dtypes.items():
        if pd.api.types.is_integer_dtype(dtype):
            sql_type = "INTEGER"
        elif pd.api.types.is_float_dtype(dtype):
            sql_type = "REAL"
        else:
            sql_type = "TEXT"  # Default to TEXT for non-numeric columns
        columns.append(f"{col} {sql_type}")
    
    columns_sql = ", ".join(columns)
    create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_sql});"
    
    cursor.execute(create_table_query)