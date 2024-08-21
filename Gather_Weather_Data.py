import pandas as pd
from datetime import datetime, timedelta
from playwright.sync_api import sync_playwright

# Function to scrape data for a specific date
def scrape_data(date):
    url = f"https://www.wunderground.com/history/daily/us/tx/houston/KIAH/date/{date}"
    
    # Initialize Playwright and open a browser
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Set headless=False for debugging
        page = browser.new_page()
        page.goto(url)

        # Wait for the table to load
        page.wait_for_selector('table.mat-table')

        # Debugging: Check if table exists
        table_exists = page.locator('table.mat-table').count() > 0
        if not table_exists:
            print(f"No table found on {date}")
            browser.close()
            return

        # Scrape headers
        headers = page.locator('table.mat-table thead th').all_inner_texts()
        print(f"Headers: {headers}")

        #Add the 'Date' columnn to the headers
        headers.append('Date')

        # Scrape rows
        rows_data = []
        rows = page.locator('table.mat-table tbody tr')
        row_count = rows.count()
        print(f"Number of rows found: {row_count}")

        for i in range(row_count):
            row = rows.nth(i).locator('td').all_inner_texts()
            row.append(date)
            print(f"Row {i+1}: {row}")  # Debugging: Inspect each row
            rows_data.append(row)

        # Save data to CSV
        if headers and rows_data:
            df = pd.DataFrame(rows_data, columns=headers)
            df.to_csv(f"weather_data_{date}.csv", index=False)
            print(f"Data saved to weather_data_{date}.csv")
        else:
            print(f"No data available for {date}")

        browser.close()
    
    return df

# Function to scrape data for a date range
def scrape_date_range(start_date, end_date):
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    delta = timedelta(days=1)
    all_data = []

    current_date = start
    while current_date <= end:
        print(f"Scraping data for {current_date.strftime('%Y-%m-%d')}")
        df = scrape_data(current_date.strftime('%Y-%m-%d'))
        all_data.append(df)
        current_date += delta

    # Concatenate all data into one DataFrame
    full_data = pd.concat(all_data, ignore_index=True)
    
    # Save the data to a CSV file
    full_data.to_csv(f"weather_data_{start_date}_to_{end_date}.csv", index=False)
    print(f"Data saved to weather_data_{start_date}_to_{end_date}.csv")

# Main script to run the scraping
if __name__ == "__main__":
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")
    scrape_date_range(start_date, end_date)
