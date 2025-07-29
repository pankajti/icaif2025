import yfinance as yf

def get_stock_data(ticker):

    historical_data = yf.download(ticker,period='1y')
    close_prices = historical_data['Close']
    return close_prices


if __name__ == '__main__':
    data = get_stock_data('AAPL')
    print(data.head())