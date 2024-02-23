from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime
from alpaca_trade_api import REST
from timedelta import Timedelta
from Analyze_Sentiments import analyze_news_sentiment

# Define Alpaca API credentials for paper trading
API_KEY = "YOUR_API_KEY" 
API_SECRET = "YOUR_API_SECRET" 
BASE_URL = "https://paper-api.alpaca.markets"

# Store the credentials in a dictionary for easy access
ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}

class SentimentBasedTrader(Strategy):
    # Initialization of the strategy with a default symbol and risk factor
    def initialize(self, symbol: str = "", risk_factor: float = 0.5):
        self.symbol = symbol
        self.sleeptime = "24H"
        self.last_deal = None
        self.risk_factor = risk_factor
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

    # Calculate the cash, last price, and quantity for order placement
    def calculate_order_details(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.risk_factor / last_price, 0)
        return cash, last_price, quantity

    # Fetch dates for sentiment analysis, specifically the last three days
    def fetch_dates_for_analysis(self):
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    # Analyze the sentiment of news articles for the specified symbol
    def analyze_sentiment(self):
        today, three_days_prior = self.fetch_dates_for_analysis()
        news = self.api.get_news(symbol=self.symbol, start=three_days_prior, end=today)
        news_headlines = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = analyze_news_sentiment(news_headlines)
        return probability, sentiment

    # The main trading logic to be executed on each trading iteration
    def on_trading_iteration(self):
        cash, last_price, quantity = self.calculate_order_details()
        probability, sentiment = self.analyze_sentiment()

        # Make trading decisions based on the analyzed sentiment
        if cash > last_price:
            if sentiment == "positive" and probability > .999:
                if self.last_deal == "sell":
                    self.sell_all()  # Clear any existing short positions before going long
                # Create a buy order with bracket parameters for taking profit and stopping loss
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type="bracket",
                    take_profit_price=last_price * 1.20,
                    stop_loss_price=last_price * 0.95
                )
                self.submit_order(order)  # Submit the created buy order
                self.last_deal = "buy"
                
            elif sentiment == "negative" and probability > .999:
                if self.last_deal == "buy":
                    self.sell_all()  # Clear any existing long positions before going short
                # Create a sell order with bracket parameters for taking profit and stopping loss
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "sell",
                    type="bracket",
                    take_profit_price=last_price * 0.80,
                    stop_loss_price=last_price * 1.05
                )
                self.submit_order(order)  # Submit the created sell order
                self.last_deal = "sell"

# Define the backtesting period
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 1, 31)

# Initialize the broker with the specified credentials
broker = Alpaca(ALPACA_CREDS)

# Initialize the strategy with a name, the broker, and specific parameters
strategy = SentimentBasedTrader(name='mlstrat', broker=broker,
                                parameters={"symbol": "SPY", "risk_factor": .5})

# Backtest the strategy using Yahoo data for historical prices
strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"symbol": "SPY", "risk_factor": .5}
)

# Uncomment the following lines to run the strategy with real or paper trading
# trader = Trader()
# trader.add_strategy(strategy)
# trader.run_all()
