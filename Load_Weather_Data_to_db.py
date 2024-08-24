import os
import pandas as pd
import psycopg2
from datetime import datetime
from psycopg2 import sql
from config import get_postgres_keys
import warnings

# Function to display available folders and prompt the user to select one
def select_folder():
    # List all the folders in the current project directory
    folders = [f for f in os.listdir('.') if os.path.isdir(f)]
    
    # Display folders and prompt user to select one
    print("Select a folder containing the CSV data:")
    for i, folder in enumerate(folders, 1):
        print(f"{i}. {folder}")

    # Prompt user to select folder by number
    folder_choice = int(input("Enter the number of the folder: ")) - 1
    selected_folder = folders[folder_choice]
    
    # Parse the string after the second '_'
    location = '_'.join(selected_folder.split('_')[2:])
    
    print(f"Selected folder: {selected_folder}")
    print(f"Location parsed: {location}")
    
    return selected_folder, location

# Function to concatenate all CSV files and add a DateTime column
def concat_csv_and_add_location():
    selected_folder, location = select_folder()

    # Read all CSV files in the selected folder
    all_files = [f for f in os.listdir(selected_folder) if f.endswith('.csv')]
    all_data = []

    for file in all_files:
        file_path = os.path.join(selected_folder, file)
        df = pd.read_csv(file_path)
        
        # Suppress only the specific 'H' deprecated warning
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning, message="'H' is deprecated and will be removed in a future version, please use 'h' instead.")
            
            # Round the 'Time' column to the nearest hour
            df['Time_Rounded'] = pd.to_datetime(df['Time'], format='%I:%M %p').dt.round('H')
        df['DateTime'] = pd.to_datetime(df['Date']) + pd.to_timedelta(df['Time_Rounded'].dt.hour, unit='h')
        
        # Add the 'location' column
        df['Location'] = location

        # Append to the list of dataframes
        all_data.append(df)

    # Concatenate all the dataframes
    full_data = pd.concat(all_data, ignore_index=True)

    # Save to a new CSV (if needed) or send to PostgreSQL
    full_data.to_csv("concatenated_data.csv", index=False)
    print("Data has been concatenated and saved to concatenated_data.csv")

    return full_data


def create_table_if_not_exists(conn):
    cur = conn.cursor()

    # SQL statement to create the table
    create_table_query = """
    CREATE TABLE IF NOT EXISTS Weather (
        id SERIAL PRIMARY KEY,
        Time VARCHAR,
        Temperature VARCHAR,
        Dew_Point VARCHAR,
        Humidity VARCHAR,
        Wind VARCHAR,
        Wind_Speed VARCHAR,
        Wind_Gust VARCHAR,
        Pressure VARCHAR,
        Precipitation VARCHAR,
        Condition VARCHAR,
        Date DATE,
        Time_Rounded VARCHAR,
        DateTime TIMESTAMP,
        Location VARCHAR
    );
    """

    try:
        # Execute the create table query
        cur.execute(create_table_query)
        conn.commit()
        print("Table created successfully or already exists.")
    except Exception as e:
        print(f"Error creating table: {e}")
    finally:
        cur.close()

if __name__ == "__main__":
    # Read and concatenate CSV files, adding the 'location' column
    full_data = concat_csv_and_add_location()

    pg_port, pg_username, pg_host, pg_database_name, postgres_superuser_password = get_postgres_keys()
    # Connect to PostgreSQL database
    conn = psycopg2.connect(
        host=pg_host,
        database=pg_database_name,
        user=pg_username,
        password=postgres_superuser_password,
        port=pg_port
    )
    cursor = conn.cursor()

    # Create the table if it does not exist
    create_table_if_not_exists(conn)
    print(full_data)
    # Insert data into PostgreSQL
    for index, row in full_data.iterrows():
        insert_query = sql.SQL("""
            INSERT INTO Weather (Time, Temperature, Dew_Point, Humidity, Wind, Wind_Speed, Wind_Gust, Pressure, Precipitation, Condition, Date, Time_Rounded, DateTime, Location)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """)
        cursor.execute(insert_query, (row['Time'], row['Temperature'], row['Dew Point'],row['Humidity'], row['Wind'], row['Wind Speed'], row['Wind Gust'], row['Pressure'], row['Precip.'], row['Condition'], row['Date'], row['Time_Rounded'], row['DateTime'], row['Location']))

    # Commit and close the connection
    conn.commit()
    cursor.close()
    conn.close()

    print("Data has been successfully saved to the PostgreSQL database.")
