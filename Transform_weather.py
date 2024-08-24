import pandas as pd
import psycopg2
from psycopg2 import sql
from config import get_postgres_keys

if __name__ == "__main__":
    pg_port, pg_username, pg_host, pg_database_name, postgres_superuser_password = get_postgres_keys()
    # Connect to PostgreSQL and fetch data ordered by DateTime
    conn = psycopg2.connect(
        host=pg_host,
        database=pg_database_name,
        user=pg_username,
        password=postgres_superuser_password,
        port=pg_port
    )
    cursor = conn.cursor()

    # Query data ordered by DateTime
    query = "SELECT * FROM weather ORDER BY datetime ASC"
    data = pd.read_sql_query(query, conn)

    # Strip the '°F' and convert temperature to integer
    data['temperature_int'] = data['temperature'].str.replace('°F', '').astype(int)

    # Calculate Δ in hourly temp (difference between current and previous rows)
    data['Δ in hourly temp'] = data['temperature_int'].diff()

    # Drop the temporary integer temperature column (optional)
    data.drop(columns=['temperature_int'], inplace=True)

    alter_table_query = """
        ALTER TABLE Weather
        ADD COLUMN IF NOT EXISTS temp_diff FLOAT;
    """

    cursor.execute(alter_table_query)
    conn.commit()
    for index, row in data.iterrows():
        update_query = sql.SQL("""
            UPDATE weather
            SET temp_diff = %s
            WHERE DateTime = %s
        """)
        cursor.execute(update_query, (row['Δ in hourly temp'], row['datetime']))

    # Display the data with the new column
    print(data[['datetime', 'temperature', 'Δ in hourly temp']])

    conn.commit()
    cursor.close()
    conn.close()

    # Display the data
    print(data)

