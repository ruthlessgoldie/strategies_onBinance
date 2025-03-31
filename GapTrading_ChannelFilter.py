from binance.client import Client
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class GapStrategyBacktest:
    def __init__(self, symbol, timeframe):
        self.client = Client("", "")
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Strategy Parameters
        self.time_index = 5       # Exit after n periods
        self.initial_capital = 1000
        self.fixed_fractional = 0.01
        
        # Results tracking
        self.trades = []
        self.equity_curve = []

    def get_historical_data(self, start_date, end_date):
        klines = self.client.get_historical_klines(
            self.symbol, self.timeframe, start_date, end_date
        )
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignored'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        return df

    def identify_patterns(self, df):
        df = df.copy()
        
        # Gap Conditions with directional confirmation
        df['long_gap'] = (df['open'] > df['close'].shift(1)) & (df['close'] > df['open'])& (df['close'].shift(1) > df['open'].shift(1))
        df['short_gap'] = (df['open'] < df['close'].shift(1)) & (df['close'] < df['open'])& (df['close'].shift(1) < df['open'].shift(1))
        
        return df.dropna()

    def run_backtest(self, start_date, end_date):
        df = self.get_historical_data(start_date, end_date)
        df = self.identify_patterns(df)
        
        capital = self.initial_capital
        position = 0
        entry_price = 0
        entry_date = None
        position_size = 0
        
        for i in range(1, len(df)-1):
            current = df.iloc[i]
            next_bar = df.iloc[i+1]
            
            # Exit Logic
            if position != 0:
                days_in_trade = (current['timestamp'] - entry_date).days
                
                # Time-based exit
                if days_in_trade >= self.time_index:
                    exit_price = current['close']
                    self._close_position(position, entry_price, exit_price, entry_date, current['timestamp'], capital, 'time')
                    position = 0
                    entry_date = None  # Reset entry_date after closing position
            
            # Entry Logic
            if position == 0:
                if current['long_gap']:
                    position = 1
                    entry_price = next_bar['open']
                    position_size = (capital * self.fixed_fractional) / entry_price
                    entry_date = next_bar['timestamp']
                    self._enter_position(position, entry_price, entry_date, position_size)
                elif current['short_gap']:
                    position = -1
                    entry_price = next_bar['open']
                    position_size = (capital * self.fixed_fractional) / entry_price
                    entry_date = next_bar['timestamp']
                    self._enter_position(position, entry_price, entry_date, position_size)
            
            self.equity_curve.append({'date': current['timestamp'], 'equity': capital})
    
    def _enter_position(self, direction, entry_price, entry_date, position_size):
        self.trades.append({
            'type': 'long' if direction == 1 else 'short',
            'entry_date': entry_date,
            'entry_price': entry_price,
            'position_size': position_size
        })

    def _close_position(self, position, entry_price, exit_price, entry_date, exit_date, capital, exit_type):
        position_size = (capital * self.fixed_fractional) / entry_price
        profit = (exit_price - entry_price) * position_size * position
        capital += profit
        
        self.trades[-1].update({
            'exit_date': exit_date,
            'exit_price': exit_price,
            'profit': profit,
            'exit_type': exit_type
        })
        self.equity_curve[-1]['equity'] = capital  # Update last equity value

    def get_backtest_results(self):
        if not self.trades: return "No trades found."
        
        trades_df = pd.DataFrame(self.trades)
        equity = pd.DataFrame(self.equity_curve)['equity']
        
        stats = {
            'Total Trades': len(trades_df),
            'Win Rate': f"{len(trades_df[trades_df['profit'] > 0])/len(trades_df)*100:.1f}%",
            'Total Profit': f"${trades_df['profit'].sum():.2f}",
            'Max Drawdown': f"{self._calculate_max_drawdown(equity):.1f}%"
        }
        return stats

    def _calculate_max_drawdown(self, equity):
        peak = equity.expanding().max()
        return (1 - equity/peak).max() * 100

def main():
    backtest = GapStrategyBacktest('ETHUSDT', '4h')
    end = datetime.now()
    backtest.run_backtest((end - timedelta(days=365)).strftime("%d %b %Y %H:%M:%S"), end.strftime("%d %b %Y %H:%M:%S"))
    
    results = backtest.get_backtest_results()
    print("\nBacktest Results:")
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()