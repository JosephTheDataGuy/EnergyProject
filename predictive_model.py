import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from Load_Energy_Plans import select_excel_file, load_excel_file
from database_functions import save_df_to_db, connect_to_postgres, load_data_from_db, create_table_if_not_exists, connect_to_postgres_sqlalchemy
from sqlalchemy import create_engine, text
from config import get_postgres_keys

def convert_period_start_to_datetime(df, column_name):
    """
    Convert 'Period Start' column from 'mm/dd/yyyy - hh:mm AM/PM' format to 'yyyy-mm-dd hh:mm:ss'.
    
    Args:
        df (pd.DataFrame): DataFrame containing the 'Period Start' column.
        column_name (str): The name of the column to convert.
    
    Returns:
        pd.DataFrame: DataFrame with a new column 'DateTime' in proper timestamp format.
    """
    # Convert 'Period Start' column to datetime
    df['DateTime'] = pd.to_datetime(df[column_name], format='%m/%d/%Y - %I:%M %p')
    return df

def clean_weather_data(df):
    """
    Clean weather data by converting string values to numeric where appropriate.
    """
    # Strip "°F" from temperature and dew_point, convert to integers
    df.loc[:, 'temperature'] = df['temperature'].str.replace('°F', '').astype(float)

    # Strip "°%" or "%" from humidity and convert to float
    df.loc[:, 'humidity'] = df['humidity'].str.replace(r'[°%]', '', regex=True).astype(float)

    return df

def create_predictive_model_summary_table():
    """
    Create a PostgreSQL table to store the summary of the predictive model's performance.
    """
    # Connect to your PostgreSQL database using SQLAlchemy
    engine = connect_to_postgres_sqlalchemy()

    # SQL query to create the table
    create_table_query = """
    CREATE TABLE IF NOT EXISTS predictive_model_summary (
        date DATE,
        actual_kwh FLOAT,
        predicted_kwh FLOAT,
        absolute_error FLOAT,
        cumulative_mae FLOAT
    );
    """

    # Execute the SQL query to create the table
    with engine.connect() as conn:
        conn.execute(text(create_table_query))
        print("Table 'predictive_model_summary' created successfully.")

def populate_aggregate_table(X_test, y_test, predictions, datetime_test):
    """
    Populate the PostgreSQL table with the test data, model predictions, and the 'DateTime' column.
    """
    # Create the summary DataFrame, including the 'DateTime' column
    summary_df = pd.DataFrame({
        'DateTime': datetime_test,  # Use the split 'DateTime' column here
        'actual_kwh': y_test,
        'predicted_kwh': predictions
    })
    
    # Calculate absolute error and cumulative MAE
    summary_df['absolute_error'] = abs(summary_df['actual_kwh'] - summary_df['predicted_kwh'])
    summary_df['cumulative_mae'] = summary_df['absolute_error'].expanding().mean()

    # Insert the data into the database
    engine = connect_to_postgres_sqlalchemy()
    summary_df[['DateTime', 'actual_kwh', 'predicted_kwh', 'absolute_error', 'cumulative_mae']].to_sql(
        'predictive_model_summary', engine, if_exists='replace', index=False
    )

def load_and_merge_data(usage_file, weather_query):
    """
    Load usage and weather data, then merge them on 'DateTime'.
    
    Args:
        usage_file (str): Path to the Excel file containing usage data.
        weather_query (str): SQL query to load weather data from the database.
    
    Returns:
        pd.DataFrame: Merged DataFrame of usage and weather data.
    """
    usage_df = load_excel_file(usage_file)
    
    # Convert 'Period Start' to proper datetime format
    usage_df = convert_period_start_to_datetime(usage_df, 'Period Start')
    
    weather_df = load_data_from_db(weather_query)
    weather_df = clean_weather_data(weather_df)  # Clean weather data
    
    # Merge usage and weather data on 'DateTime'
    merged_df = pd.merge(usage_df, weather_df, left_on='DateTime', right_on='datetime')
    return merged_df

def preprocess_data(merged_df):
    """
    Preprocess the merged data to filter appliance usage and add new features.
    
    Args:
        merged_df (pd.DataFrame): Merged DataFrame of usage and weather data.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame with added features and filtered data.
    """
    # Filter out appliance usage
    merged_df['appliance_used'] = merged_df['Usage (kWh)'].apply(lambda x: 1 if x >= 2.2 else 0)
    filtered_df = merged_df[merged_df['appliance_used'] == 0]

    # Feature engineering
    filtered_df['month'] = pd.to_datetime(filtered_df['DateTime']).dt.month
    filtered_df['hour'] = pd.to_datetime(filtered_df['DateTime']).dt.hour

    return filtered_df

def train_model(filtered_df):
    """
    Train a Random Forest model on the filtered data.
    
    Args:
        filtered_df (pd.DataFrame): Preprocessed and filtered data.
    
    Returns:
        model, X_test, y_test, predictions: Trained machine learning model and test data with predictions.
    """
    # Features and target variable
    x = filtered_df[['temperature', 'temp_diff', 'humidity', 'month', 'hour']]
    y = filtered_df['Usage (kWh)']

    # Retain 'datetime' for later use in the summary table
    datetime_col = filtered_df['DateTime']

    # Perform train-test split (datetime is not part of X, but is split separately)
    X_train, X_test, y_train, y_test, datetime_train, datetime_test = train_test_split(
        x, y, datetime_col, test_size=0.2, random_state=42
    )

    # Train model (Random Forest)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on test set
    predictions = model.predict(X_test)

    # Evaluate performance (Mean Absolute Error)
    mae = mean_absolute_error(y_test, predictions)
    print(f'MAE: {mae}')
    
    # Return model, X_test, y_test, and predictions
    return model, X_test, y_test, predictions, datetime_test

def predict_yearly_usage(model, location_query):
    """
    Predict yearly usage based on the full weather dataset using the trained model.
    
    Args:
        model: Trained machine learning model.
        location_query (str): SQL query to load full-year weather data.
    
    Returns:
        pd.DataFrame: DataFrame with predictions added.
    """
    full_year_weather = load_data_from_db(location_query)

    # Prepare full year features
    full_year_weather['month'] = pd.to_datetime(full_year_weather['datetime']).dt.month
    full_year_weather['hour'] = pd.to_datetime(full_year_weather['datetime']).dt.hour
    X_full_year = full_year_weather[['temperature', 'temp_diff', 'humidity', 'month', 'hour']]
    X_full_year = clean_weather_data(X_full_year)

    # Check if X_full_year is empty
    if X_full_year.empty:
        print("Error: No valid rows for prediction. Check your weather data and preprocessing steps.")
        return None

    # Predict full year usage
    full_year_predictions = model.predict(X_full_year)

    # Add predictions to DataFrame
    full_year_weather['Predicted_kWh'] = full_year_predictions

    return full_year_weather

def main():
    # Step 1: Select Excel file for usage data
    usage_file = select_excel_file()
    
    # Step 2: Define weather data query
    weather_query = "SELECT * FROM weather"
    
    # Step 3: Load and merge usage and weather data
    merged_df = load_and_merge_data(usage_file, weather_query)
    
    # Step 4: Preprocess data
    filtered_df = preprocess_data(merged_df)
    
    # Step 5: Train the Random Forest model
    model, X_test, y_test, predictions, datetime_test = train_model(filtered_df)
    
    # Step 6: Predict yearly usage based on weather data
    location_query = "SELECT * FROM weather WHERE location = 'tx_houston_KIAH'"
    full_year_weather_with_predictions = predict_yearly_usage(model, location_query)
    
    # Step 7: Save the predictions to the database
    save_df_to_db(full_year_weather_with_predictions, 'predicted_yearly_usage', mode='replace')

    # Step 8: Create and populate summary table for predictive model based on test data
    create_predictive_model_summary_table()
    populate_aggregate_table(X_test, y_test, predictions, datetime_test)


if __name__ == "__main__":
    main()
