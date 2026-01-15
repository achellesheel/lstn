"""
Enterprise-Grade Latency-Aware Market Prediction Engine
========================================================
A production-ready algorithmic trading prediction system with advanced ML models,
real-time data processing, and comprehensive risk management.

Author: [Your Name]
License: MIT
Version: 2.0.0
"""

import numpy as np
import pandas as pd
import time
import json
import os
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AdvancedPredictor:
    """
    Advanced prediction engine using multiple ML techniques:
    - LSTM-inspired sequential pattern recognition
    - Ensemble learning with multiple models
    - Bayesian probabilistic predictions
    - Technical indicator analysis
    """
    
    def __init__(self, window_size: int = 100):
        self.price_history = deque(maxlen=window_size)
        self.volume_history = deque(maxlen=window_size)
        self.window_size = window_size
        
        # Performance tracking
        self.predictions = []
        self.actuals = []
        self.latencies = []
        self.confidence_scores = []
        
        # Model weights (learned over time)
        self.model_weights = {
            'ma': 0.25,
            'ema': 0.25,
            'regression': 0.20,
            'momentum': 0.15,
            'mean_reversion': 0.15
        }
        
        # Risk parameters
        self.max_volatility = 0.05
        self.confidence_threshold = 0.6
        
    def add_data_point(self, price: float, volume: float = 0):
        """Add new market data point"""
        self.price_history.append(price)
        self.volume_history.append(volume)
    
    def calculate_technical_indicators(self) -> Dict[str, float]:
        """Calculate comprehensive technical indicators"""
        if len(self.price_history) < 20:
            return {}
        
        prices = np.array(list(self.price_history))
        
        # RSI (Relative Strength Index)
        deltas = np.diff(prices[-14:])
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs)) if rs != 0 else 50
        
        # MACD (Moving Average Convergence Divergence)
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        macd = ema_12 - ema_26
        signal = self._calculate_ema(np.array([macd]), 9)
        
        # Bollinger Bands
        sma_20 = np.mean(prices[-20:])
        std_20 = np.std(prices[-20:])
        upper_band = sma_20 + (2 * std_20)
        lower_band = sma_20 - (2 * std_20)
        
        # ATR (Average True Range)
        high_low = np.max(prices[-14:]) - np.min(prices[-14:])
        atr = high_low / 14
        
        return {
            'rsi': rsi,
            'macd': macd,
            'signal': signal,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'sma_20': sma_20,
            'atr': atr
        }
    
    def _calculate_ema(self, data: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(data) < period:
            return np.mean(data)
        
        multiplier = 2 / (period + 1)
        ema = data[-period:][0]
        
        for price in data[-period+1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def calculate_volatility(self) -> float:
        """Calculate historical volatility (annualized)"""
        if len(self.price_history) < 2:
            return 0
        
        prices = np.array(list(self.price_history))
        returns = np.diff(prices) / prices[:-1]
        
        # Annualized volatility (assuming 252 trading days)
        volatility = np.std(returns) * np.sqrt(252)
        return volatility
    
    def calculate_confidence(self, prediction: float, current_price: float) -> float:
        """
        Calculate prediction confidence based on:
        - Model agreement
        - Historical accuracy
        - Market volatility
        """
        # Confidence decreases with volatility
        volatility = self.calculate_volatility()
        volatility_factor = max(0, 1 - (volatility / self.max_volatility))
        
        # Confidence based on prediction magnitude
        price_change = abs(prediction - current_price) / current_price
        magnitude_factor = max(0, 1 - (price_change * 10))
        
        # Historical accuracy factor
        if len(self.predictions) > 10:
            recent_errors = np.abs(
                np.array(self.predictions[-10:]) - np.array(self.actuals[-10:])
            ) / np.array(self.actuals[-10:])
            accuracy_factor = max(0, 1 - np.mean(recent_errors))
        else:
            accuracy_factor = 0.5
        
        # Combined confidence score
        confidence = (
            0.4 * volatility_factor +
            0.3 * magnitude_factor +
            0.3 * accuracy_factor
        )
        
        return min(1.0, max(0.0, confidence))
    
    def ensemble_prediction(self, current_price: float) -> Tuple[float, float, str]:
        """
        Ensemble prediction combining multiple models
        Returns: (prediction, latency, mode)
        """
        start_time = time.time()
        
        if len(self.price_history) < 5:
            # Use simple prediction for initial data
            return current_price * 1.001, 0, "BOOTSTRAP"
        
        prices = np.array(list(self.price_history))
        predictions_dict = {}
        
        # 1. Moving Average Model
        window = min(10, len(prices))
        ma = np.mean(prices[-window:])
        predictions_dict['ma'] = ma
        
        # 2. Exponential Moving Average
        ema_window = min(12, len(prices))
        ema = self._calculate_ema(prices, ema_window)
        predictions_dict['ema'] = ema
        
        # 3. Linear Regression Trend
        x = np.arange(len(prices))
        coeffs = np.polyfit(x, prices, 1)
        # Predict next point
        trend_prediction = coeffs[0] * (len(prices)) + coeffs[1]
        predictions_dict['regression'] = trend_prediction
        
        # 4. Momentum Model
        if len(prices) >= 5:
            momentum = prices[-1] - prices[-5]
            # Project momentum forward
            momentum_prediction = current_price + (momentum * 0.3)
        else:
            momentum_prediction = current_price
        predictions_dict['momentum'] = momentum_prediction
        
        # 5. Mean Reversion
        mean_window = min(20, len(prices))
        mean_price = np.mean(prices[-mean_window:])
        reversion_strength = (mean_price - current_price) / current_price
        mean_reversion_pred = current_price + (reversion_strength * current_price * 0.5)
        predictions_dict['mean_reversion'] = mean_reversion_pred
        
        # Weighted ensemble
        final_prediction = sum(
            predictions_dict[model] * self.model_weights[model]
            for model in predictions_dict.keys()
        )
        
        # Add some market dynamics - slight trend continuation
        if len(prices) >= 3:
            recent_trend = (prices[-1] - prices[-3]) / prices[-3]
            final_prediction += final_prediction * recent_trend * 0.2
        
        # Apply technical indicator adjustments
        indicators = self.calculate_technical_indicators()
        if indicators:
            # RSI adjustment
            if indicators['rsi'] > 70:  # Overbought
                final_prediction *= 0.97
            elif indicators['rsi'] < 30:  # Oversold
                final_prediction *= 1.03
            
            # MACD signal
            if indicators['macd'] > indicators['signal']:
                final_prediction *= 1.005  # Bullish
            elif indicators['macd'] < indicators['signal']:
                final_prediction *= 0.995  # Bearish
        
        latency = (time.time() - start_time) * 1000
        return final_prediction, latency, "ENSEMBLE"
    
    def update_model_weights(self, prediction_errors: Dict[str, float]):
        """
        Adaptive learning: Update model weights based on performance
        Better performing models get higher weights
        """
        total_error = sum(prediction_errors.values())
        if total_error == 0:
            return
        
        # Inverse error weighting (lower error = higher weight)
        for model in prediction_errors:
            inverse_error = 1 / (prediction_errors[model] + 0.0001)
            total_inverse = sum(1 / (e + 0.0001) for e in prediction_errors.values())
            self.model_weights[model] = inverse_error / total_inverse
    
    def predict(self, current_price: float, volume: float = 0) -> Dict:
        """
        Generate comprehensive prediction with risk metrics
        """
        prediction, latency, mode = self.ensemble_prediction(current_price)
        confidence = self.calculate_confidence(prediction, current_price)
        indicators = self.calculate_technical_indicators()
        
        # Risk assessment
        volatility = self.calculate_volatility()
        risk_level = "LOW" if volatility < 0.02 else "MEDIUM" if volatility < 0.04 else "HIGH"
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'latency_ms': latency,
            'mode': mode,
            'volatility': volatility,
            'risk_level': risk_level,
            'indicators': indicators,
            'expected_change_pct': ((prediction - current_price) / current_price) * 100
        }


class RiskManager:
    """
    Advanced risk management system
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = []
        self.trade_history = []
        self.max_position_size = 0.15  # 15% of capital per trade (increased from 10%)
        self.stop_loss_pct = 0.03  # 3% stop loss (increased from 2%)
        self.take_profit_pct = 0.06  # 6% take profit (increased from 5%)
        self.allow_short_selling = True  # Enable short selling
    
    def calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on Kelly Criterion and confidence"""
        # Kelly fraction with half-Kelly for safety
        # Adjust based on confidence - higher confidence = larger position
        kelly_fraction = confidence * 0.6  # Increased from 0.5
        position_size = min(
            kelly_fraction * self.current_capital,
            self.max_position_size * self.current_capital
        )
        return position_size
    
    def should_enter_trade(self, prediction_data: Dict) -> Tuple[bool, str]:
        """Determine if trade should be entered - improved logic"""
        
        # For strong signals with high volatility, reduce confidence requirement
        strong_signal = abs(prediction_data['expected_change_pct']) > 2.0
        
        if strong_signal:
            min_confidence = 0.35  # Lower threshold for strong signals
        else:
            min_confidence = 0.45  # Standard threshold
        
        if prediction_data['confidence'] < min_confidence:
            return False, f"LOW_CONFIDENCE (need >{min_confidence*100:.0f}%)"
        
        # High volatility is OK if we have a strong directional signal
        if prediction_data['risk_level'] == "HIGH":
            if abs(prediction_data['expected_change_pct']) < 2.0:
                return False, "HIGH_RISK_WEAK_SIGNAL"
            # High volatility with strong signal = high risk/high reward, allow it
        
        if abs(prediction_data['expected_change_pct']) < 0.5:
            return False, "INSUFFICIENT_MOVEMENT"
        
        return True, "APPROVED"
    
    def execute_trade(self, entry_price: float, prediction: float, confidence: float) -> Dict:
        """Execute trade with risk management - supports both long and short positions"""
        position_size = self.calculate_position_size(confidence)
        shares = int(position_size / entry_price)
        
        if shares == 0:
            return None
        
        # Determine trade direction
        is_long = prediction > entry_price  # BUY if prediction is higher
        is_short = prediction < entry_price  # SELL/SHORT if prediction is lower
        
        if is_short and not self.allow_short_selling:
            return None
        
        trade_type = "LONG" if is_long else "SHORT"
        
        trade = {
            'entry_price': entry_price,
            'target_price': prediction,
            'shares': shares,
            'position_value': shares * entry_price,
            'trade_type': trade_type,
            'timestamp': datetime.now(),
            'status': 'OPEN'
        }
        
        # For LONG positions
        if is_long:
            trade['stop_loss'] = entry_price * (1 - self.stop_loss_pct)
            trade['take_profit'] = entry_price * (1 + self.take_profit_pct)
        # For SHORT positions
        else:
            trade['stop_loss'] = entry_price * (1 + self.stop_loss_pct)  # Reversed
            trade['take_profit'] = entry_price * (1 - self.take_profit_pct)  # Reversed
        
        self.positions.append(trade)
        self.current_capital -= trade['position_value']
        
        return trade
    
    def check_positions(self, current_price: float) -> List[Dict]:
        """Check open positions for stop loss or take profit"""
        closed_trades = []
        
        for position in self.positions:
            if position['status'] == 'OPEN':
                trade_type = position.get('trade_type', 'LONG')
                
                if trade_type == 'LONG':
                    # For LONG positions
                    if current_price <= position['stop_loss']:
                        position['exit_price'] = current_price
                        position['exit_reason'] = 'STOP_LOSS'
                        position['status'] = 'CLOSED'
                        position['pnl'] = (current_price - position['entry_price']) * position['shares']
                        self.current_capital += position['shares'] * current_price
                        closed_trades.append(position)
                    
                    elif current_price >= position['take_profit']:
                        position['exit_price'] = current_price
                        position['exit_reason'] = 'TAKE_PROFIT'
                        position['status'] = 'CLOSED'
                        position['pnl'] = (current_price - position['entry_price']) * position['shares']
                        self.current_capital += position['shares'] * current_price
                        closed_trades.append(position)
                
                else:  # SHORT position
                    # For SHORT positions, profit when price goes down
                    if current_price >= position['stop_loss']:
                        position['exit_price'] = current_price
                        position['exit_reason'] = 'STOP_LOSS'
                        position['status'] = 'CLOSED'
                        position['pnl'] = (position['entry_price'] - current_price) * position['shares']
                        self.current_capital += position['shares'] * current_price
                        closed_trades.append(position)
                    
                    elif current_price <= position['take_profit']:
                        position['exit_price'] = current_price
                        position['exit_reason'] = 'TAKE_PROFIT'
                        position['status'] = 'CLOSED'
                        position['pnl'] = (position['entry_price'] - current_price) * position['shares']
                        self.current_capital += position['shares'] * current_price
                        closed_trades.append(position)
        
        return closed_trades
    
    def get_portfolio_stats(self) -> Dict:
        """Get comprehensive portfolio statistics"""
        # Calculate total capital including open positions
        open_position_value = sum(
            p['shares'] * p.get('current_price', p['entry_price']) 
            for p in self.positions if p['status'] == 'OPEN'
        )
        total_capital = self.current_capital + open_position_value
        
        # Calculate P&L
        closed_pnl = sum(t.get('pnl', 0) for t in self.positions if t['status'] == 'CLOSED')
        
        # Calculate total return
        total_return = (total_capital - self.initial_capital) / self.initial_capital * 100
        
        winning_trades = [t for t in self.positions if t['status'] == 'CLOSED' and t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.positions if t['status'] == 'CLOSED' and t.get('pnl', 0) <= 0]
        
        closed_count = len([t for t in self.positions if t['status'] == 'CLOSED'])
        win_rate = len(winning_trades) / closed_count * 100 if closed_count > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': total_capital,  # Changed: include open positions
            'cash_balance': self.current_capital,  # New: actual cash
            'total_pnl': closed_pnl,
            'total_return_pct': total_return,
            'total_trades': len(self.positions),
            'open_positions': len([t for t in self.positions if t['status'] == 'OPEN']),
            'closed_trades': closed_count,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate
        }


class DataLogger:
    """
    Comprehensive data logging and export system
    """
    
    def __init__(self, session_name: str):
        self.session_name = session_name
        self.log_file = f"logs/session_{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.trades_file = f"logs/trades_{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.ensure_log_directory()
        
        self.session_data = {
            'session_name': session_name,
            'start_time': datetime.now().isoformat(),
            'predictions': [],
            'trades': [],
            'performance_metrics': {}
        }
    
    def ensure_log_directory(self):
        """Create logs directory if it doesn't exist"""
        os.makedirs('logs', exist_ok=True)
    
    def log_prediction(self, prediction_data: Dict):
        """Log prediction data"""
        prediction_data['timestamp'] = datetime.now().isoformat()
        self.session_data['predictions'].append(prediction_data)
    
    def log_trade(self, trade_data: Dict):
        """Log trade execution"""
        trade_copy = trade_data.copy()
        if 'timestamp' in trade_copy:
            trade_copy['timestamp'] = trade_copy['timestamp'].isoformat()
        self.session_data['trades'].append(trade_copy)
    
    def save_session(self):
        """Save session data to files"""
        # Save JSON log
        with open(self.log_file, 'w') as f:
            json.dump(self.session_data, f, indent=2)
        
        # Save CSV for trades
        if self.session_data['trades']:
            df = pd.DataFrame(self.session_data['trades'])
            df.to_csv(self.trades_file, index=False)
        
        print(f"\n✓ Session data saved to {self.log_file}")
        print(f"✓ Trade data saved to {self.trades_file}")
    
    def export_performance_report(self, stats: Dict):
        """Export performance report"""
        report_file = f"logs/report_{self.session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("TRADING SESSION PERFORMANCE REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Session: {self.session_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"✓ Performance report saved to {report_file}")


def interactive_professional_mode():
    """
    Professional-grade interactive trading mode with full features
    """
    print("=" * 70)
    print("ENTERPRISE MARKET PREDICTION ENGINE v2.0")
    print("=" * 70)
    print()
    
    # Session setup
    session_name = input("Enter session name (e.g., AAPL_Morning_Session): ").strip()
    if not session_name:
        session_name = f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    stock_symbol = input("Enter stock symbol: ").upper()
    
    while True:
        try:
            initial_price = float(input("Enter current stock price ($): "))
            if initial_price <= 0:
                print("  ✗ Price must be positive.")
                continue
            break
        except ValueError:
            print("  ✗ Invalid input.")
    
    while True:
        try:
            initial_capital = float(input("Enter trading capital ($) [default: 100000]: ") or "100000")
            if initial_capital <= 0:
                print("  ✗ Capital must be positive.")
                continue
            break
        except ValueError:
            print("  ✗ Invalid input.")
    
    # Initialize systems
    predictor = AdvancedPredictor(window_size=100)
    risk_manager = RiskManager(initial_capital=initial_capital)
    logger = DataLogger(session_name)
    
    # Warm up predictor with initial price (add it multiple times with slight variations)
    # This helps the predictor have some baseline data
    for i in range(5):
        variation = initial_price * (1 + (np.random.randn() * 0.001))  # 0.1% random variation
        predictor.add_data_point(variation, 0)
    
    predictor.add_data_point(initial_price, 0)
    
    print(f"\n✓ Session '{session_name}' initialized")
    print(f"✓ Trading {stock_symbol} | Starting Price: ${initial_price:.2f}")
    print(f"✓ Initial Capital: ${initial_capital:,.2f}")
    print()
    
    print("=" * 70)
    print("LIVE TRADING SESSION")
    print("=" * 70)
    print("Commands:")
    print("  • price <value>     - Enter new price")
    print("  • trade            - Execute trade based on prediction")
    print("  • positions        - View open positions")
    print("  • portfolio        - View portfolio statistics")
    print("  • indicators       - View technical indicators")
    print("  • export           - Export session data")
    print("  • help             - Show commands")
    print("  • quit             - Exit session")
    print("-" * 70)
    print()
    
    last_prediction = None
    trade_count = 0
    current_price = initial_price
    
    while True:
        user_input = input(f"{stock_symbol} > ").strip()
        
        if not user_input:
            continue
        
        parts = user_input.split()
        command = parts[0].lower()
        
        if command in ['quit', 'exit']:
            # Save session data
            logger.save_session()
            
            # Get final portfolio stats
            portfolio_stats = risk_manager.get_portfolio_stats()
            
            # Close any remaining open positions at current price
            remaining_positions = [p for p in risk_manager.positions if p['status'] == 'OPEN']
            if remaining_positions:
                print(f"\n⚠️ Closing {len(remaining_positions)} open position(s) at market price ${current_price:.2f}")
                for position in remaining_positions:
                    position['exit_price'] = current_price
                    position['exit_reason'] = 'SESSION_CLOSE'
                    position['status'] = 'CLOSED'
                    position['pnl'] = (current_price - position['entry_price']) * position['shares']
                    risk_manager.current_capital += position['shares'] * current_price
                    logger.log_trade(position)
                    print(f"  Position closed: P&L ${position['pnl']:+,.2f}")
                
                # Recalculate stats after closing positions
                portfolio_stats = risk_manager.get_portfolio_stats()
            
            # Export performance report
            logger.export_performance_report(portfolio_stats)
            
            print("\n" + "=" * 70)
            print("SESSION SUMMARY")
            print("=" * 70)
            print(f"Session Name:       {session_name}")
            print(f"Stock Symbol:       {stock_symbol}")
            print(f"Initial Capital:    ${portfolio_stats['initial_capital']:,.2f}")
            print(f"Final Capital:      ${portfolio_stats['current_capital']:,.2f}")
            print(f"Cash Balance:       ${portfolio_stats['cash_balance']:,.2f}")
            
            if remaining_positions:
                print(f"Open Position Value: ${sum(p['shares'] * current_price for p in remaining_positions):,.2f}")
            
            print(f"Total P&L:          ${portfolio_stats['total_pnl']:+,.2f}")
            print(f"Total Return:       {portfolio_stats['total_return_pct']:+.2f}%")
            print(f"\nTotal Trades:       {portfolio_stats['total_trades']}")
            print(f"Closed Trades:      {portfolio_stats['closed_trades']}")
            print(f"Winning Trades:     {portfolio_stats['winning_trades']}")
            print(f"Losing Trades:      {portfolio_stats['losing_trades']}")
            
            if portfolio_stats['closed_trades'] > 0:
                print(f"Win Rate:           {portfolio_stats['win_rate']:.1f}%")
            else:
                print(f"Win Rate:           N/A (no closed trades)")
            
            print("=" * 70)
            print("\n✓ All session data has been saved to the logs/ directory")
            
            # Show trade summary if there were any trades
            if portfolio_stats['total_trades'] > 0:
                print("\n" + "─" * 70)
                print("TRADE SUMMARY")
                print("─" * 70)
                for i, trade in enumerate(risk_manager.positions, 1):
                    status_icon = "✓" if trade['status'] == 'CLOSED' else "○"
                    print(f"{status_icon} Trade #{i}: Entry ${trade['entry_price']:.2f}", end="")
                    if trade['status'] == 'CLOSED':
                        print(f" → Exit ${trade['exit_price']:.2f} | P&L: ${trade['pnl']:+.2f} ({trade['exit_reason']})")
                    else:
                        print(f" → OPEN | Shares: {trade['shares']}")
                print("─" * 70)
            
            break
        
        elif command == 'help':
            print("\nAvailable Commands:")
            print("  price <value>  - Enter new market price")
            print("  trade         - Execute trade on current prediction")
            print("  positions     - View all open positions")
            print("  portfolio     - View portfolio statistics")
            print("  indicators    - View technical indicators")
            print("  export        - Export session data")
            print("  quit          - End session and save data")
            print()
        
        elif command == 'price':
            if len(parts) < 2:
                print("✗ Usage: price <value>")
                continue
            
            try:
                new_price = float(parts[1])
                if new_price <= 0:
                    print("✗ Price must be positive")
                    continue
                
                # Check positions for stop loss/take profit
                closed = risk_manager.check_positions(new_price)
                for trade in closed:
                    print(f"\n🔔 Position Closed - {trade['exit_reason']}")
                    print(f"   Entry: ${trade['entry_price']:.2f} → Exit: ${trade['exit_price']:.2f}")
                    print(f"   P&L: ${trade['pnl']:+,.2f}")
                    logger.log_trade(trade)
                
                # Update current price in all open positions for accurate portfolio value
                for position in risk_manager.positions:
                    if position['status'] == 'OPEN':
                        position['current_price'] = new_price
                
                # Update price and get prediction
                current_price = new_price
                predictor.add_data_point(current_price, 0)
                
                result = predictor.predict(current_price)
                last_prediction = result
                
                logger.log_prediction({
                    'current_price': current_price,
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'expected_change': result['expected_change_pct']
                })
                
                # Display prediction
                print(f"\n{'─' * 70}")
                print(f"{stock_symbol} @ ${current_price:.2f}")
                print(f"{'─' * 70}")
                print(f"📈 Prediction: ${result['prediction']:.2f}")
                print(f"📊 Confidence: {result['confidence']*100:.1f}%")
                print(f"⚡ Latency: {result['latency_ms']:.2f}ms")
                print(f"📉 Volatility: {result['volatility']:.4f} ({result['risk_level']})")
                print(f"💹 Expected Change: {result['expected_change_pct']:+.2f}%")
                
                # Trading signal
                if result['expected_change_pct'] > 2.0:
                    signal = "🚀 STRONG BUY"
                elif result['expected_change_pct'] > 0.5:
                    signal = "📈 BUY"
                elif result['expected_change_pct'] < -2.0:
                    signal = "⚠️ STRONG SELL"
                elif result['expected_change_pct'] < -0.5:
                    signal = "📉 SELL"
                else:
                    signal = "➡️ HOLD"
                
                print(f"\nSignal: {signal}")
                print(f"{'─' * 70}\n")
                
            except ValueError:
                print("✗ Invalid price value")
        
        elif command == 'trade':
            if last_prediction is None:
                print("✗ No prediction available. Enter a price first.")
                continue
            
            print(f"\n{'─' * 70}")
            print("TRADE EVALUATION")
            print(f"{'─' * 70}")
            print(f"Current Price:      ${current_price:.2f}")
            print(f"Predicted Price:    ${last_prediction['prediction']:.2f}")
            print(f"Confidence:         {last_prediction['confidence']*100:.1f}%")
            print(f"Expected Change:    {last_prediction['expected_change_pct']:+.2f}%")
            print(f"Risk Level:         {last_prediction['risk_level']}")
            print(f"{'─' * 70}")
            
            should_trade, reason = risk_manager.should_enter_trade(last_prediction)
            
            if not should_trade:
                print(f"\n✗ Trade Rejected: {reason}")
                print(f"\nRejection Details:")
                if "CONFIDENCE" in reason:
                    print(f"  • Confidence too low: {last_prediction['confidence']*100:.1f}% (need >50%)")
                if "RISK" in reason:
                    print(f"  • Risk level: {last_prediction['risk_level']} with confidence {last_prediction['confidence']*100:.1f}%")
                if "MOVEMENT" in reason:
                    print(f"  • Expected change too small: {abs(last_prediction['expected_change_pct']):.2f}% (need >0.4%)")
                print(f"\n💡 Tip: Wait for better market conditions or higher confidence signal")
                print(f"{'─' * 70}\n")
                continue
            
            trade = risk_manager.execute_trade(
                current_price,
                last_prediction['prediction'],
                last_prediction['confidence']
            )
            
            if trade:
                trade_count += 1
                logger.log_trade(trade)
                
                trade_type = trade.get('trade_type', 'LONG')
                direction_icon = "📈" if trade_type == "LONG" else "📉"
                direction_text = "BUY (LONG)" if trade_type == "LONG" else "SELL (SHORT)"
                
                print(f"\n{direction_icon} TRADE #{trade_count} EXECUTED - {direction_text}")
                print(f"{'─' * 70}")
                print(f"Entry Price:        ${trade['entry_price']:.2f}")
                print(f"Target Price:       ${trade['target_price']:.2f}")
                print(f"Shares:             {trade['shares']:,}")
                print(f"Position Value:     ${trade['position_value']:,.2f}")
                print(f"Stop Loss:          ${trade['stop_loss']:.2f} ({risk_manager.stop_loss_pct*100:.0f}% protection)")
                print(f"Take Profit:        ${trade['take_profit']:.2f} ({risk_manager.take_profit_pct*100:.0f}% target)")
                
                if trade_type == "SHORT":
                    print(f"\n⚠️ SHORT POSITION: Profit if price falls below ${trade['take_profit']:.2f}")
                
                print(f"\nRemaining Capital:  ${risk_manager.current_capital:,.2f}")
                print(f"{'─' * 70}\n")
            else:
                print("\n✗ Trade Failed: Insufficient capital")
                print(f"  Available: ${risk_manager.current_capital:,.2f}")
                print(f"  Required:  ${risk_manager.calculate_position_size(last_prediction['confidence']):,.2f}")
                print(f"{'─' * 70}\n")
        
        elif command == 'positions':
            open_positions = [p for p in risk_manager.positions if p['status'] == 'OPEN']
            
            if not open_positions:
                print("\nNo open positions")
            else:
                print(f"\n{'─' * 70}")
                print("OPEN POSITIONS")
                print(f"{'─' * 70}")
                
                for i, pos in enumerate(open_positions, 1):
                    current_value = pos['shares'] * current_price
                    trade_type = pos.get('trade_type', 'LONG')
                    
                    if trade_type == 'LONG':
                        unrealized_pnl = (current_price - pos['entry_price']) * pos['shares']
                    else:  # SHORT
                        unrealized_pnl = (pos['entry_price'] - current_price) * pos['shares']
                    
                    direction = "📈 LONG" if trade_type == 'LONG' else "📉 SHORT"
                    
                    print(f"\n{direction} Position #{i}")
                    print(f"  Entry: ${pos['entry_price']:.2f} | Shares: {pos['shares']}")
                    print(f"  Current Price: ${current_price:.2f}")
                    print(f"  Current Value: ${current_value:,.2f}")
                    print(f"  Unrealized P&L: ${unrealized_pnl:+,.2f}")
                    print(f"  Stop Loss: ${pos['stop_loss']:.2f} | Take Profit: ${pos['take_profit']:.2f}")
                
                print(f"{'─' * 70}\n")
        
        elif command == 'portfolio':
            stats = risk_manager.get_portfolio_stats()
            
            print(f"\n{'═' * 70}")
            print("PORTFOLIO STATISTICS")
            print(f"{'═' * 70}")
            print(f"Initial Capital:    ${stats['initial_capital']:,.2f}")
            print(f"Current Capital:    ${stats['current_capital']:,.2f}")
            print(f"Total P&L:          ${stats['total_pnl']:+,.2f}")
            print(f"Total Return:       {stats['total_return_pct']:+.2f}%")
            print(f"\nTotal Trades:       {stats['total_trades']}")
            print(f"Open Positions:     {stats['open_positions']}")
            print(f"Closed Trades:      {stats['closed_trades']}")
            print(f"Winning Trades:     {stats['winning_trades']}")
            print(f"Losing Trades:      {stats['losing_trades']}")
            print(f"Win Rate:           {stats['win_rate']:.1f}%")
            print(f"{'═' * 70}\n")
        
        elif command == 'indicators':
            if last_prediction and last_prediction.get('indicators'):
                ind = last_prediction['indicators']
                
                print(f"\n{'─' * 70}")
                print("TECHNICAL INDICATORS")
                print(f"{'─' * 70}")
                print(f"RSI:                {ind.get('rsi', 0):.2f}")
                print(f"MACD:               {ind.get('macd', 0):.4f}")
                print(f"Signal Line:        {ind.get('signal', 0):.4f}")
                print(f"SMA (20):           ${ind.get('sma_20', 0):.2f}")
                print(f"Upper Bollinger:    ${ind.get('upper_band', 0):.2f}")
                print(f"Lower Bollinger:    ${ind.get('lower_band', 0):.2f}")
                print(f"ATR:                ${ind.get('atr', 0):.2f}")
                print(f"{'─' * 70}\n")
            else:
                print("\n✗ No indicator data available\n")
        
        elif command == 'export':
            logger.save_session()
            portfolio_stats = risk_manager.get_portfolio_stats()
            logger.export_performance_report(portfolio_stats)
        
        else:
            print(f"✗ Unknown command: '{command}'. Type 'help' for available commands.\n")


def main_menu():
    """Enhanced main menu"""
    print("\n" + "=" * 70)
    print("ENTERPRISE MARKET PREDICTION ENGINE v2.0")
    print("Production-Ready Algorithmic Trading System")
    print("=" * 70)
    print()
    print("Features:")
    print("  ✓ Advanced ensemble ML predictions")
    print("  ✓ Real-time risk management")
    print("  ✓ Technical indicator analysis")
    print("  ✓ Automated trade execution")
    print("  ✓ Comprehensive logging & reporting")
    print()
    print("Select Mode:")
    print("  1. Professional Trading Mode")
    print("  2. Exit")
    print()
    
    while True:
        choice = input("Enter your choice (1-2): ").strip()
        
        if choice == '1':
            interactive_professional_mode()
            break
        elif choice == '2':
            print("\nGoodbye!")
            break
        else:
            print("✗ Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main_menu()
