import pandas as pd
from datetime import datetime, timedelta
from playwright.sync_api import sync_playwright
from config import get_scrapingbee_api_key
import requests
from bs4 import BeautifulSoup
import os

def send_request(date):
    folder_name = 'individual_dates_tx_houston_KIAH'  # You can dynamically generate this based on URL if needed

    # Check if folder exists, if not, create it
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")

    api_key = get_scrapingbee_api_key()
    url = f'https://www.wunderground.com/history/daily/us/tx/houston/KIAH/date/{date}'
    
    response = requests.get(
        url='https://app.scrapingbee.com/api/v1/',
        params={
            'api_key': api_key,
            'url': url,
            'wait_for': 'table.mat-table',
        },
    )
    
    if response.status_code == 200:
        print('Data retrieved successfully')
        return response.content
    else:
        print('Failed to retrieve data')
        print(f'Status Code: {response.status_code}')
        return None


# Function to scrape data for a specific date
def scrape_data(date, html_content):
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the table element
    table = soup.find('table', {'class': 'mat-table'})
    if not table:
        print(f"No table found on {date}")
        return None

    # Scrape headers
    headers = [th.get_text(strip=True) for th in table.find('thead').find_all('th')]
    print(f"Headers: {headers}")

    # Add the 'Date' column to the headers
    headers.append('Date')

    # Scrape rows
    rows_data = []
    rows = table.find('tbody').find_all('tr')
    print(f"Number of rows found: {len(rows)}")

    for i, row in enumerate(rows):
        # Get all text from the <td> elements in each row
        row_data = [td.get_text(strip=True) for td in row.find_all('td')]
        row_data.append(date)  # Add the date to the end of the row
        print(f"Row {i+1}: {row_data}")  # Debugging: Inspect each row
        rows_data.append(row_data)

    # Save data to CSV
    if headers and rows_data:
        df = pd.DataFrame(rows_data, columns=headers)
        folder_name = 'individual_dates_tx_houston_KIAH'
        file_path = os.path.join(folder_name, f"weather_data_{date}.csv")
        df.to_csv(file_path, index=False)
        print(f"Data saved to weather_data_{date}.csv")
    else:
        print(f"No data available for {date}")

    return df

# Function to scrape data for a date range
def scrape_date_range(start_date, end_date):
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    delta = timedelta(days=1)
    all_data = []

    current_date = start
    while current_date <= end:
        formatted_date = current_date.strftime('%Y-%m-%d')
        print(f"Scraping data for {formatted_date}")
        
        # Call send_request to get the HTML content for the current date
        html_content = send_request(formatted_date)

        if html_content:
            # Pass the response content (HTML) to the scrape_data function
            df = scrape_data(formatted_date, html_content)
            if df is not None:
                all_data.append(df)
        else:
            print(f"Failed to retrieve data for {formatted_date}")
        
        # Move to the next date
        current_date += delta

    # Concatenate all data into one DataFrame
    if all_data:
        full_data = pd.concat(all_data, ignore_index=True)
    
        # Save the data to a CSV file
        full_data.to_csv(f"weather_data_{start_date}_to_{end_date}.csv", index=False)
        print(f"Data saved to weather_data_{start_date}_to_{end_date}.csv")
    else:
        print("No data was scraped.")

# Main script to run the scraping
if __name__ == "__main__":
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")
    scrape_date_range(start_date, end_date)
