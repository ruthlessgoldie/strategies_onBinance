from binance.client import Client
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class OpeningRangeBreakoutStrategy:
    def __init__(self, symbol, timeframe):
        """
        Initialize Opening Range Breakout Strategy with realistic trading costs
        """
        self.client = Client("", "")  # Empty credentials for data only
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Strategy Parameters from the specification
        self.noise_length = 10  # Length for Noise calculation
        self.stretch_multiple = 2  # Stretch multiplier
        self.nr_size = 2  # Narrow Range size (2-Bar Narrow Range)
        self.look_back = 20  # Look back period for NR pattern
        self.time_index = 1  # Time exit index (default 1 day)
        self.target_index = [1.0, 10.0]  # Target range
        self.target_step = 0.25  # Target step
        self.atr_length = 20  # ATR length for stop loss
        self.atr_stop = 6  # ATR multiplier for stop loss
        
        # Trading costs
        self.commission_rate = 0.0004  # Binance Futures maker/taker fee (0.04%)
        self.slippage_percent = 0.0005  # 0.05% slippage estimate
        self.spread_factor = 0.0002  # Estimated spread as percentage of price
        
        # Results tracking
        self.trades = []
        self.equity_curve = []
        self.trade_details = []  # For detailed trade analysis
        self.initial_capital = 1000
        self.fixed_fractional = 0.01

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
    
    def calculate_noise(self, df):
        """Calculate Noise: difference between open and extreme on each day"""
        df['Noise'] = np.abs(df['high'] - df['low']) - np.abs(df['open'] - df[['high', 'low']].min(axis=1))
        df['Average_Noise'] = df['Noise'].rolling(window=self.noise_length).mean()
        df['Stretch'] = df['Average_Noise'] * self.stretch_multiple
        return df
    
    def identify_nr_pattern(self, df):
        """Identify 2-Bar Narrow Range pattern within look back period"""
        df['Range'] = df['high'] - df['low']
        df['NR'] = df['Range'].rolling(window=2).min()
        
        # Check if current 2-bar range is narrowest in look_back period
        df['Is_NR'] = df['NR'] == df['Range'].rolling(window=self.look_back).min()
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
    
    def apply_opening_range_breakout(self, df):
        """Apply Opening Range Breakout strategy"""
        df = df.copy()
        
        # Calculate Noise and Stretch
        df = self.calculate_noise(df)
        df = self.identify_nr_pattern(df)
        df['atr'] = self.calculate_atr(df, self.atr_length)
        
        # Opening Range Breakout setup
        df['Long_Entry'] = df['open'] + df['Stretch']
        df['Short_Entry'] = df['open'] - df['Stretch']
        
        # Use a single target multiplier (midpoint of target_index range for simplicity)
        target_multiplier = np.mean(self.target_index)  # Use 5.5 as a default (midpoint of [1.0, 10.0])
        
        # Targets and Stops (using ATR-based targets and stops)
        df['Long_Target'] = df['open'] + (df['atr'] * target_multiplier)
        df['Short_Target'] = df['open'] - (df['atr'] * target_multiplier)
        df['Long_Stop'] = df['open'] - (df['atr'] * self.atr_stop)
        df['Short_Stop'] = df['open'] + (df['atr'] * self.atr_stop)
        
        return df

    def apply_market_friction(self, price, direction, is_entry=True):
        """
        Apply realistic market friction (spread, slippage, commission)
        """
        if (direction == 1 and is_entry) or (direction == -1 and not is_entry):
            price_after_spread = price * (1 + self.spread_factor)
            price_after_slippage = price_after_spread * (1 + self.slippage_percent)
        else:
            price_after_spread = price * (1 - self.spread_factor)
            price_after_slippage = price_after_spread * (1 - self.slippage_percent)
        
        commission = price_after_slippage * self.commission_rate
        return price_after_slippage, commission

    def run_backtest(self, start_date, end_date):
        """Run backtest over specified period with realistic trading costs"""
        df = self.get_historical_data(start_date, end_date)
        df = self.apply_opening_range_breakout(df)
        
        capital = self.initial_capital
        position = 0  # 0: no position, 1: long, -1: short
        entry_price = 0
        entry_date = None
        position_size = 0
        total_commission = 0
        
        for i in range(self.look_back, len(df)-1):
            current_bar = df.iloc[i]
            next_bar = df.iloc[i+1]
            
            # Exit existing position if needed
            if position != 0:
                days_in_trade = (current_bar['timestamp'] - entry_date).days
                exit_type = None
                exit_price_raw = 0
                
                # Time-based exit (n days at close)
                if days_in_trade >= self.time_index:
                    exit_price_raw = current_bar['close']
                    exit_type = 'time'
                
                # Target exit
                if position == 1 and current_bar['high'] >= current_bar['Long_Target']:
                    exit_price_raw = current_bar['Long_Target']
                    exit_type = 'target'
                elif position == -1 and current_bar['low'] <= current_bar['Short_Target']:
                    exit_price_raw = current_bar['Short_Target']
                    exit_type = 'target'
                
                # Stop loss exit (ATR based)
                if position == 1 and current_bar['low'] <= current_bar['Long_Stop']:
                    exit_price_raw = current_bar['Long_Stop']
                    exit_type = 'stop_loss'
                elif position == -1 and current_bar['high'] >= current_bar['Short_Stop']:
                    exit_price_raw = current_bar['Short_Stop']
                    exit_type = 'stop_loss'
                
                if exit_type:
                    exit_price_adjusted, exit_commission = self.apply_market_friction(
                        exit_price_raw, position, is_entry=False
                    )
                    
                    if position == 1:
                        profit = (exit_price_adjusted - entry_price) * position_size
                    else:  # position == -1
                        profit = (entry_price - exit_price_adjusted) * position_size
                    
                    commission_cost = exit_commission * position_size
                    total_commission += commission_cost
                    profit -= commission_cost
                    
                    capital += profit
                    
                    self.trades.append({
                        'type': 'long' if position == 1 else 'short',
                        'entry_date': entry_date,
                        'exit_date': current_bar['timestamp'],
                        'entry_price': entry_price,
                        'exit_price': exit_price_adjusted,
                        'position_size': position_size,
                        'commission': commission_cost,
                        'profit': profit,
                        'exit_type': exit_type
                    })
                    
                    position = 0
            
            # Enter new position if NR pattern and breakout conditions are met
            if position == 0 and current_bar['Is_NR']:
                # Long entry: Buy stop above open + Stretch
                if next_bar['high'] > current_bar['Long_Entry']:
                    position = 1
                    entry_price_raw = current_bar['Long_Entry']
                    entry_price_adjusted, entry_commission = self.apply_market_friction(
                        entry_price_raw, position, is_entry=True
                    )
                    entry_price = entry_price_adjusted
                    position_size = (capital * self.fixed_fractional) / entry_price
                    entry_date = next_bar['timestamp']
                    total_commission += entry_commission * position_size
                
                                # Short entry: Sell stop below open - Stretch
                elif next_bar['low'] < current_bar['Short_Entry']:
                    position = -1
                    entry_price_raw = current_bar['Short_Entry']
                    entry_price_adjusted, entry_commission = self.apply_market_friction(
                        entry_price_raw, position, is_entry=True
                    )
                    entry_price = entry_price_adjusted
                    position_size = (capital * self.fixed_fractional) / entry_price
                    entry_date = next_bar['timestamp']
                    total_commission += entry_commission * position_size
            
            self.equity_curve.append({
                'date': current_bar['timestamp'],
                'equity': capital,
                'total_commission': total_commission
            })

    def calculate_max_drawdown(self, equity_series):
        """Calculate maximum drawdown percentage"""
        rolling_max = equity_series.expanding(min_periods=1).max()
        drawdown = ((equity_series - rolling_max) / rolling_max) * 100
        return abs(drawdown.min())
    
    def calculate_sharpe_ratio(self, equity_series, risk_free_rate=0.02):
        """Calculate annualized Sharpe Ratio"""
        # Convert equity to returns
        returns = equity_series.pct_change().dropna()
        
        # Annualize
        days = len(returns)
        annual_factor = 365 / days  # Assuming 365 trading days per year
        
        excess_returns = returns - (risk_free_rate / 365)  # Daily risk-free rate
        sharpe = (excess_returns.mean() * annual_factor) / (returns.std() * np.sqrt(annual_factor))
        
        return sharpe

    def get_backtest_results(self):
        """Calculate and return backtest statistics with cost metrics"""
        if not self.trades:
            return "No trades found in backtest period."
        
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Calculate statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['profit'] > 0])
        losing_trades = len(trades_df[trades_df['profit'] <= 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_profit = trades_df['profit'].sum()
        total_commission = trades_df['commission'].sum()
        
        max_drawdown = self.calculate_max_drawdown(equity_df['equity'])
        
        # Calculate Sharpe Ratio
        sharpe_ratio = self.calculate_sharpe_ratio(pd.Series(equity_df['equity']))
        
        # Calculate profit factor
        gross_profit = trades_df[trades_df['profit'] > 0]['profit'].sum()
        gross_loss = abs(trades_df[trades_df['profit'] < 0]['profit'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Calculate average trade metrics
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        avg_win = trades_df[trades_df['profit'] > 0]['profit'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['profit'] < 0]['profit'].mean() if losing_trades > 0 else 0
        
        # Calculate by exit type
        exit_types = trades_df['exit_type'].value_counts().to_dict()
        
        return {
            'Performance Metrics': {
                'Total Trades': total_trades,
                'Winning Trades': winning_trades,
                'Losing Trades': losing_trades,
                'Win Rate': f"{win_rate:.2f}%",
                'Profit Factor': f"{profit_factor:.2f}",
                'Total Profit/Loss': f"${total_profit:,.2f}",
                'Total Commission Paid': f"${total_commission:,.2f}",
                'Net Profit %': f"{(total_profit / self.initial_capital) * 100:.2f}%",
                'Max Drawdown': f"{max_drawdown:.2f}%",
                'Sharpe Ratio': f"{sharpe_ratio:.2f}",
                'Final Equity': f"${self.equity_curve[-1]['equity']:,.2f}",
            },
            'Trade Metrics': {
                'Average Profit per Trade': f"${avg_profit:,.2f}",
                'Average Winning Trade': f"${avg_win:,.2f}",
                'Average Losing Trade': f"${avg_loss:,.2f}",
            },
            'Exit Types': {
                'Time-based Exits': exit_types.get('time', 0),
                'Target Exits': exit_types.get('target', 0),
                'Stop Loss Exits': exit_types.get('stop_loss', 0),
            },
            'Trading Costs': {
                'Commission Rate': f"{self.commission_rate*100:.3f}%",
                'Slippage Estimate': f"{self.slippage_percent*100:.3f}%",
                'Spread Factor': f"{self.spread_factor*100:.3f}%",
            }
        }
    
    def export_results(self, filename_prefix):
        """Export backtest results to CSV files"""
        # Export trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(f"{filename_prefix}_trades.csv", index=False)
        
        # Export equity curve
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.to_csv(f"{filename_prefix}_equity.csv", index=False)
        
        # Export detailed trade info
        if self.trade_details:
            details_df = pd.DataFrame(self.trade_details)
            details_df.to_csv(f"{filename_prefix}_trade_details.csv", index=False)

def main():
    # Create backtest instance
    backtest = OpeningRangeBreakoutStrategy(symbol='ETHUSDT', timeframe='5m')  # Using daily timeframe as per strategy
    
    # Run 6-month backtest
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30*6)
    
    print(f"Running backtest from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    backtest.run_backtest(
        start_date.strftime("%d %b %Y %H:%M:%S"),
        end_date.strftime("%d %b %Y %H:%M:%S")
    )
    
    # Print results
    results = backtest.get_backtest_results()
    print("\nBacktest Results:")
    
    for section, metrics in results.items():
        print(f"\n{section}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    
    # Export results to CSV
    backtest.export_results("ethusdt_orb_backtest")
    print("\nBacktest results exported to CSV files.")

if __name__ == "__main__":
    main()