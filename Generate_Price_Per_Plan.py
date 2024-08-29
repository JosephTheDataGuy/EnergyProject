import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import connect, sql
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from Load_Energy_Plans import select_excel_file
from config import get_postgres_keys

# Function to calculate the cost based on the logic
def calculate_cost(row, kwh):
    # Apply default delivery charge and base charge if they are 0
    delivery_charge = row['Delivery Charge'] if row['Delivery Charge'] != 0 else 0.038264
    delivery_base_charge = row['Delivery Base Charge'] if row['Delivery Base Charge'] != 0 else 4.39
    
    # Initialize the cost with base charge if applicable
    cost = 0
    if row['Base Charge Qualifier'] < kwh < row['Base charge dequalifer']:
        cost += row['base charge']
    
    # Energy charge calculation based on usage
    if row['energy charge upcharge usage'] > 0 and kwh > row['energy charge upcharge usage']:
        cost += row['energy charge upcharge usage'] * row['energy charge']
        cost += (kwh - row['energy charge upcharge usage']) * row['secondary energy charge']
    else:
        cost += kwh * row['energy charge']
    
    # Add delivery charges
    cost += kwh * delivery_charge + delivery_base_charge
    
    # Apply bill credits if applicable
    if kwh >= row['1st Credit qualifier (kWh)']:
        cost -= row['Bill Credit']
        if kwh >= row['2nd Credit Qualifier (kWh)']:
            cost -= row['Bill Credit']
    if 'Credit dequalifier' in row and kwh > row['Credit dequalifier']:
        cost += row['Bill Credit']  # Add back the bill credit if dequalified

    return cost

# Function to load Excel file and filter rows where 'special' = 'N'
def load_energy_plans(file_path):
    df = pd.read_excel(file_path)
    df_filtered = df[df['special'] == 'N']  # Only process rows where 'special' is 'N'
    return df_filtered

# Function to create costs table in PostgreSQL using psycopg2.sql
def create_costs_table(cursor):
    # Define the table structure with SQL dynamic query
    query = sql.SQL("""
        CREATE TABLE IF NOT EXISTS energy_plan_costs (
            plan_url TEXT,
            kwh INT,
            cost NUMERIC
        )
    """)
    cursor.execute(query)

# Function to insert calculated costs into the PostgreSQL table
def insert_costs_into_db(cursor, energy_plans_df):
    for _, row in energy_plans_df.iterrows():
        plan_url = row['plan url']
        for kwh in range(1, 3001):  # kWh values from 1 to 3000
            cost = calculate_cost(row, kwh)
            insert_query = sql.SQL("""
                INSERT INTO energy_plan_costs (plan_url, kwh, cost)
                VALUES (%s, %s, %s)
            """)
            cursor.execute(insert_query, [plan_url, kwh, cost])

# Main script execution
def main():
    # Select Excel file
    file_path = select_excel_file()
    if not file_path:
        return
    
    # Load PostgreSQL keys and connect to the database
    pg_port, pg_username, pg_host, pg_database_name, postgres_superuser_password = get_postgres_keys()
    conn = psycopg2.connect(
        host=pg_host,
        database=pg_database_name,
        user=pg_username,
        password=postgres_superuser_password,
        port=pg_port
    )
    cursor = conn.cursor()

    # Load energy plans data
    energy_plans_df = load_energy_plans(file_path)

    # Create costs table
    create_costs_table(cursor)

    # Insert costs for the filtered energy plans
    insert_costs_into_db(cursor, energy_plans_df)

    # Commit and close connection
    conn.commit()
    cursor.close()
    conn.close()
    print("Costs have been successfully inserted into the database.")

if __name__ == "__main__":
    main()