import finnhub
import csv
import datetime
import yfinance as yf
from collections import defaultdict
from pprint import pprint

# Set up Finnhub client
finnhub_client = finnhub.Client(api_key="cohttk9r01qpcmnifpb0cohttk9r01qpcmnifpbg")

# Set up date range
end_date = datetime.datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.datetime.now() - datetime.timedelta(days=2*365)).strftime('%Y-%m-%d')

# Fetch company news   KO ACN SBUX FRSH
company_news = finnhub_client.company_news('FRSH', _from=start_date, to=end_date)

# Fetch AAPL close price history
aapl = yf.Ticker("FRSH")
aapl_history = aapl.history(start=start_date, end=end_date)
close_prices = aapl_history['Close'].tolist()

# Combine headlines, summaries, and closing prices for each date
date_data = defaultdict(lambda: {'headlines': [], 'summaries': [], 'close_prices': []})
for news_item, price in zip(company_news, close_prices):
    date = datetime.datetime.utcfromtimestamp(news_item['datetime']).strftime('%Y-%m-%d')
    date_data[date]['headlines'].append(news_item['headline'])
    date_data[date]['summaries'].append(news_item['summary'])
    date_data[date]['close_prices'].append(price)

# Calculate average closing price for each date
for date, data in date_data.items():
    avg_close_price = sum(data['close_prices']) / len(data['close_prices'])
    date_data[date]['close_price'] = avg_close_price

# Define CSV file name
csv_file = "company_news_aapl.csv"

# Write data to CSV file
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['date', 'headline', 'summary', 'close_price'])  # Write the header
    for date, data in date_data.items():
        writer.writerow([date, ", ".join(data['headlines']), ", ".join(data['summaries']), data['close_price']])

print("Data saved to", csv_file)

# print(finnhub_client.stock_symbols('US'))