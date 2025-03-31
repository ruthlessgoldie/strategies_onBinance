from binance.client import Client
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class RossHookStrategy:
    def __init__(self, symbol='BTCUSDT', timeframe='1d'):
        """
        Initialize Ross Hook Trading Strategy
        """
        self.client = Client("", "")  # Empty credentials for data only
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Strategy Parameters
        self.min_setup = 10
        self.max_hook = 1.5  # [0.25, 2.00]
        self.target_index = 5.5  # [1.0, 10.0]
        self.atr_length = 20
        self.atr_stop = 6
        self.initial_capital = 1000
        self.fixed_fractional = 0.01
        
        # Results tracking
        self.trades = []
        self.equity_curve = []

    def get_historical_data(self, start_date, end_date):
        """Fetch historical data from Binance"""
        klines = self.client.get_historical_klines(
            self.symbol,
            self.timeframe,
            start_date,
            end_date
        )
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignored'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df

    def calculate_atr(self, df, length=20):
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.DataFrame([tr1, tr2, tr3]).max()
        atr = tr.rolling(window=length).mean()
        
        return atr

    def identify_patterns(self, df):
        """Identify 1-2-3 patterns and Ross Hooks"""
        df = df.copy()
        
        # Calculate swings for pattern identification
        df['higher_high'] = df['high'] > df['high'].shift(1)
        df['higher_low'] = df['low'] > df['low'].shift(1)
        df['lower_high'] = df['high'] < df['high'].shift(1)
        df['lower_low'] = df['low'] < df['low'].shift(1)
        
        # Bull Pattern identification
        df['bull_point1'] = (df['lower_low']) & (df['low'].rolling(self.min_setup).min() == df['low'])
        df['bull_point2'] = df['higher_high'] & (df['high'].rolling(self.min_setup).max() == df['high'])
        df['bull_point3'] = (df['low'] > df['low'].shift(self.min_setup)) & df['higher_low']
        
        # Bear Pattern identification
        df['bear_point1'] = (df['higher_high']) & (df['high'].rolling(self.min_setup).max() == df['high'])
        df['bear_point2'] = df['lower_low'] & (df['low'].rolling(self.min_setup).min() == df['low'])
        df['bear_point3'] = (df['high'] < df['high'].shift(self.min_setup)) & df['lower_high']
        
        # Ross Hook identification
        df['bull_breakout'] = (df['close'] > df['high'].shift(1)) & df['bull_point3'].shift(1)
        df['bull_hook'] = (df['low'] < df['low'].shift(1)) & df['bull_breakout'].shift(1)
        
        df['bear_breakout'] = (df['close'] < df['low'].shift(1)) & df['bear_point3'].shift(1)
        df['bear_hook'] = (df['high'] > df['high'].shift(1)) & df['bear_breakout'].shift(1)
        
        # Calculate ATR for stop loss
        df['atr'] = self.calculate_atr(df, self.atr_length)
        
        return df

    def run_backtest(self, start_date, end_date):
        """Run backtest over specified period"""
        # Get data and identify patterns
        df = self.get_historical_data(start_date, end_date)
        df = self.identify_patterns(df)
        
        capital = self.initial_capital
        position = 0  # 0: no position, 1: long, -1: short
        entry_price = 0
        entry_date = None
        position_size = 0
        
        for i in range(2, len(df)-1):
            current_bar = df.iloc[i]
            next_bar = df.iloc[i+1]
            
            # Exit existing position if needed
            if position != 0:
                # Target Exit
                target_level = (entry_price + (entry_price - current_bar['low']) * self.target_index 
                              if position == 1 else
                              entry_price - (current_bar['high'] - entry_price) * self.target_index)
                
                if ((position == 1 and current_bar['high'] >= target_level) or 
                    (position == -1 and current_bar['low'] <= target_level)):
                    exit_price = target_level
                    profit = (exit_price - entry_price) * position_size * position
                    capital += profit
                    
                    self.trades.append({
                        'type': 'long' if position == 1 else 'short',
                        'entry_date': entry_date,
                        'exit_date': current_bar['timestamp'],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': profit,
                        'exit_type': 'target'
                    })
                    position = 0
                
                # ATR Stop Loss
                elif ((position == 1 and 
                       current_bar['low'] <= (entry_price - current_bar['atr'] * self.atr_stop)) or
                      (position == -1 and 
                       current_bar['high'] >= (entry_price + current_bar['atr'] * self.atr_stop))):
                    exit_price = (entry_price - current_bar['atr'] * self.atr_stop if position == 1 
                                else entry_price + current_bar['atr'] * self.atr_stop)
                    profit = (exit_price - entry_price) * position_size * position
                    capital += profit
                    
                    self.trades.append({
                        'type': 'long' if position == 1 else 'short',
                        'entry_date': entry_date,
                        'exit_date': current_bar['timestamp'],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': profit,
                        'exit_type': 'stop_loss'
                    })
                    position = 0
            
            # Enter new position if conditions are met
            if position == 0:
                # Long setup
                if current_bar['bull_hook']:
                    position = 1
                    entry_price = next_bar['high']  # Buy stop above high
                    position_size = (capital * self.fixed_fractional) / entry_price
                    entry_date = next_bar['timestamp']
                
                # Short setup
                elif current_bar['bear_hook']:
                    position = -1
                    entry_price = next_bar['low']  # Sell stop below low
                    position_size = (capital * self.fixed_fractional) / entry_price
                    entry_date = next_bar['timestamp']
            
            self.equity_curve.append({
                'date': current_bar['timestamp'],
                'equity': capital
            })

    def get_backtest_results(self):
        """Calculate and return backtest statistics"""
        if not self.trades:
            return "No trades found in backtest period."
        
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Calculate statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['profit'] > 0])
        losing_trades = len(trades_df[trades_df['profit'] <= 0])
        win_rate = (winning_trades / total_trades) * 100
        
        total_profit = trades_df['profit'].sum()
        max_drawdown = self.calculate_max_drawdown(equity_df['equity'])
        
        return {
            'Total Trades': total_trades,
            'Winning Trades': winning_trades,
            'Losing Trades': losing_trades,
            'Win Rate': f"{win_rate:.2f}%",
            'Total Profit/Loss': f"${total_profit:,.2f}",
            'Max Drawdown': f"{max_drawdown:.2f}%",
            'Final Equity': f"${self.equity_curve[-1]['equity']:,.2f}",
            'Average Profit per Trade': f"${total_profit/total_trades:,.2f}"
        }

    def calculate_max_drawdown(self, equity_series):
        """Calculate maximum drawdown percentage"""
        rolling_max = equity_series.expanding(min_periods=1).max()
        drawdown = ((equity_series - rolling_max) / rolling_max) * 100
        return abs(drawdown.min())

def main():
    # Create backtest instance
    backtest = RossHookStrategy(symbol='ETHUSDT', timeframe='15m')
    
    # Set custom parameters if needed
    backtest.min_setup = 10
    backtest.max_hook = 3
    backtest.target_index = 10
    
    # Run 1-year backtest
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    backtest.run_backtest(
        start_date.strftime("%d %b %Y %H:%M:%S"),
        end_date.strftime("%d %b %Y %H:%M:%S")
    )
    
    # Print results
    results = backtest.get_backtest_results()
    print("\nBacktest Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()