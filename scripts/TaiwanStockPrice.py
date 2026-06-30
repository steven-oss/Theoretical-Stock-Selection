from FinMind.data import DataLoader
import os

api = DataLoader()
api.login_by_token(api_token=os.getenv("FINMIND_TOKEN"))

stock_data = api.taiwan_stock_daily(
    stock_id="2330", 
    start_date="2020-01-01", 
    end_date="2024-09-18"
)
print(stock_data)