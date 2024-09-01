import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from Load_Energy_Plans import select_excel_file, load_excel_file
from database_functions import save_df_to_db, connect_to_postgres, load_data_from_db, create_table_if_not_exists

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
    # df['dew_point'] = df['dew_point'].str.replace('°F', '').astype(float)

    # Strip "%" from humidity, convert to integers
    df.loc[:, 'humidity'] = df['humidity'].str.replace('°%', '').astype(float)

    # Strip "°mph" from wind_speed and wind_gust, convert to integers
    # df['wind_speed'] = df['wind_speed'].str.replace('°mph', '').astype(float)
    # df['wind_gust'] = df['wind_gust'].str.replace('°mph', '').astype(float)

    # Strip "°in" from pressure and precipitation, convert to floats
    # df['pressure'] = df['pressure'].str.replace('°in', '').astype(float)
    # df['precipitation'] = df['precipitation'].str.replace('°in', '').astype(float)

    return df

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
        model: Trained machine learning model.
    """
    # Features and target variable
    X = filtered_df[['temperature', 'temp_diff', 'humidity', 'month', 'hour']]
    y = filtered_df['Usage (kWh)']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model (Random Forest)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on test set
    predictions = model.predict(X_test)

    # Evaluate performance (Mean Absolute Error)
    mae = mean_absolute_error(y_test, predictions)
    print(f'MAE: {mae}')
    
    return model

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
    model = train_model(filtered_df)
    
    # Step 6: Predict yearly usage based on weather data
    location_query = "SELECT * FROM weather WHERE location = 'tx_houston_KIAH'"
    full_year_weather_with_predictions = predict_yearly_usage(model, location_query)
    
    # Step 7: Save the predictions to the database
    save_df_to_db(full_year_weather_with_predictions, 'predicted_yearly_usage', mode='replace')

if __name__ == "__main__":
    main()
