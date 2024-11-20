import pandas as pd
import ast
import numpy as np
import os

# Step 1: Load the dataset
df = pd.read_csv("Trade.csv")

# Step 2: Handle missing data
df['Trade_History'] = df['Trade_History'].fillna('[]')

# Step 3: Safely parse 'Trade_History'
def safe_literal_eval(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []

df['Trade_History'] = df['Trade_History'].apply(safe_literal_eval)

# Step 4: Extract trade details
def extract_trade_details(trade_history):
    trade_data = []
    for trade in trade_history:
        trade_data.append({
            'time': trade.get('time', None),
            'symbol': trade.get('symbol', None),
            'side': trade.get('side', None),
            'price': float(trade.get('price', 0)),
            'quantity': float(trade.get('qty', 0)),
            'realizedProfit': float(trade.get('realizedProfit', 0)),
        })
    return trade_data

df['Parsed_Trade_History'] = df['Trade_History'].apply(extract_trade_details)

# Step 5: Financial metrics calculations
# ROI Calculation
def calculate_roi(trade_data):
    if len(trade_data) == 0:
        return 0
    roi = 0
    for trade in trade_data:
        if trade['quantity'] > 0:
            roi += trade['realizedProfit'] / trade['quantity']
    return roi

# PnL Calculation
def calculate_pnl(trade_data):
    return sum([trade['realizedProfit'] for trade in trade_data])

# Win Rate Calculation
def calculate_win_rate(trade_data):
    win_positions = len([trade for trade in trade_data if trade['realizedProfit'] > 0])
    total_positions = len(trade_data)
    return win_positions / total_positions if total_positions > 0 else 0

# Sharpe Ratio Calculation
def calculate_sharpe_ratio(trade_data):
    profits = [trade['realizedProfit'] for trade in trade_data if trade['quantity'] > 0]
    if len(profits) < 2:  # Not enough data for Sharpe Ratio
        return 0
    mean_profit = np.mean(profits)
    std_dev = np.std(profits)
    return mean_profit / std_dev if std_dev != 0 else 0

# Apply the metrics calculations
df['ROI'] = df['Parsed_Trade_History'].apply(calculate_roi)
df['PnL'] = df['Parsed_Trade_History'].apply(calculate_pnl)
df['Win_Rate'] = df['Parsed_Trade_History'].apply(calculate_win_rate)
df['Sharpe_Ratio'] = df['Parsed_Trade_History'].apply(calculate_sharpe_ratio)

# Step 6: Calculate rank and sort
df['Rank'] = df[['ROI', 'PnL', 'Win_Rate']].mean(axis=1)
df_sorted = df.sort_values(by='Rank', ascending=False)

# Step 7: Save top 20 accounts
top_20_accounts = df_sorted.head(20)

# Handle file overwrite case
if os.path.exists("top_20_accounts.csv"):
    os.remove("top_20_accounts.csv")

top_20_accounts.to_csv("top_20_accounts.csv", index=False)

# Display the top 20 accounts
print(top_20_accounts[['Port_IDs', 'Rank', 'ROI', 'PnL', 'Win_Rate']])
