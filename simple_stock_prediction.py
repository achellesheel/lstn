import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class LatencyAwarePredictor:
    """
    Market prediction engine that dynamically balances latency vs accuracy.
    Three prediction modes:
    - FAST: Low latency (~10ms), lower accuracy
    - BALANCED: Medium latency (~50ms), medium accuracy  
    - ACCURATE: High latency (~200ms), higher accuracy
    """
    
    def __init__(self, window_size=50):
        # Store historical prices for pattern analysis
        self.price_history = deque(maxlen=window_size)
        self.window_size = window_size
        
        # Track prediction performance metrics
        self.predictions = []
        self.actuals = []
        self.latencies = []
        
        # Dynamic mode selection parameters
        self.volatility_threshold_high = 0.02  # 2% volatility
        self.volatility_threshold_low = 0.005  # 0.5% volatility
        
    def add_price(self, price):
        """Add new price point to history"""
        self.price_history.append(price)
    
    def calculate_volatility(self):
        """Calculate recent market volatility (standard deviation of returns)"""
        if len(self.price_history) < 2:
            return 0
        
        prices = np.array(self.price_history)
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)
    
    def fast_prediction(self, current_price):
        """
        FAST mode: Simple moving average prediction
        Latency: ~10ms
        Uses only recent few prices for quick calculation
        """
        start_time = time.time()
        
        if len(self.price_history) < 5:
            prediction = current_price
        else:
            # Simple 5-period moving average
            recent_prices = list(self.price_history)[-5:]
            prediction = np.mean(recent_prices)
        
        latency = (time.time() - start_time) * 1000  # Convert to ms
        return prediction, latency, "FAST"
    
    def balanced_prediction(self, current_price):
        """
        BALANCED mode: Weighted moving average with trend
        Latency: ~50ms
        Uses more data points with exponential weighting
        """
        start_time = time.time()
        
        if len(self.price_history) < 10:
            prediction = current_price
        else:
            prices = np.array(list(self.price_history)[-20:])
            
            # Exponential weights (more recent = more weight)
            weights = np.exp(np.linspace(-1, 0, len(prices)))
            weights = weights / weights.sum()
            
            # Weighted average
            weighted_avg = np.sum(prices * weights)
            
            # Add trend component
            if len(prices) >= 5:
                recent_trend = (prices[-1] - prices[-5]) / 5
                prediction = weighted_avg + recent_trend
            else:
                prediction = weighted_avg
        
        latency = (time.time() - start_time) * 1000
        return prediction, latency, "BALANCED"
    
    def accurate_prediction(self, current_price):
        """
        ACCURATE mode: Advanced statistical model
        Latency: ~200ms
        Uses full history with multiple indicators
        """
        start_time = time.time()
        
        if len(self.price_history) < 20:
            prediction = current_price
        else:
            prices = np.array(list(self.price_history))
            
            # 1. Exponential Moving Average (EMA)
            alpha = 0.3
            ema = prices[0]
            for price in prices[1:]:
                ema = alpha * price + (1 - alpha) * ema
            
            # 2. Linear regression trend
            x = np.arange(len(prices))
            coeffs = np.polyfit(x, prices, 1)
            trend = coeffs[0]
            
            # 3. Momentum indicator
            if len(prices) >= 10:
                momentum = prices[-1] - prices[-10]
            else:
                momentum = 0
            
            # 4. Mean reversion component
            mean_price = np.mean(prices[-30:]) if len(prices) >= 30 else np.mean(prices)
            reversion = (mean_price - current_price) * 0.1
            
            # Combine all components with weights
            prediction = (
                0.4 * ema +              # 40% EMA
                0.3 * (current_price + trend) +  # 30% trend continuation
                0.2 * (current_price + momentum * 0.1) +  # 20% momentum
                0.1 * reversion          # 10% mean reversion
            )
        
        latency = (time.time() - start_time) * 1000
        return prediction, latency, "ACCURATE"
    
    def dynamic_mode_selection(self):
        """
        Automatically select prediction mode based on market conditions
        High volatility -> FAST (need quick decisions)
        Low volatility -> ACCURATE (can afford to wait)
        Medium volatility -> BALANCED
        """
        volatility = self.calculate_volatility()
        
        if volatility > self.volatility_threshold_high:
            return "FAST"
        elif volatility < self.volatility_threshold_low:
            return "ACCURATE"
        else:
            return "BALANCED"
    
    def predict(self, current_price, mode="AUTO"):
        """
        Generate prediction based on selected mode
        mode: "AUTO", "FAST", "BALANCED", or "ACCURATE"
        """
        # Auto mode selection based on market conditions
        if mode == "AUTO":
            mode = self.dynamic_mode_selection()
        
        # Route to appropriate prediction method
        if mode == "FAST":
            return self.fast_prediction(current_price)
        elif mode == "BALANCED":
            return self.balanced_prediction(current_price)
        else:  # ACCURATE
            return self.accurate_prediction(current_price)
    
    def evaluate_prediction(self, prediction, actual):
        """Track prediction accuracy"""
        self.predictions.append(prediction)
        self.actuals.append(actual)
        
        if len(self.predictions) > 1:
            # Calculate Mean Absolute Percentage Error (MAPE)
            errors = np.abs(np.array(self.predictions) - np.array(self.actuals))
            mape = np.mean(errors / np.array(self.actuals)) * 100
            return mape
        return 0
    
    def get_performance_stats(self):
        """Get overall performance statistics"""
        if len(self.predictions) == 0:
            return None
        
        errors = np.abs(np.array(self.predictions) - np.array(self.actuals))
        mape = np.mean(errors / np.array(self.actuals)) * 100
        avg_latency = np.mean(self.latencies) if self.latencies else 0
        
        return {
            'MAPE': mape,
            'Average_Latency_ms': avg_latency,
            'Total_Predictions': len(self.predictions)
        }


def generate_realistic_stock_prices(base_price=100, n_points=100, volatility=0.02):
    """
    Generate realistic stock price movements using Geometric Brownian Motion
    base_price: Starting stock price
    n_points: Number of price points to generate
    volatility: Market volatility (standard deviation)
    """
    prices = [base_price]
    dt = 1/252  # Daily time step (252 trading days/year)
    mu = 0.0002  # Drift (small positive trend)
    
    for _ in range(n_points - 1):
        # Random shock from normal distribution
        shock = np.random.normal(mu * dt, volatility * np.sqrt(dt))
        new_price = prices[-1] * (1 + shock)
        prices.append(new_price)
    
    return prices


def run_market_simulation():
    """
    Run complete market prediction simulation
    Demonstrates latency vs accuracy tradeoff
    """
    print("=" * 70)
    print("LATENCY-AWARE MARKET PREDICTION ENGINE")
    print("=" * 70)
    print()
    
    # Initialize predictor
    predictor = LatencyAwarePredictor(window_size=50)
    
    # Generate realistic stock prices
    print("Generating realistic stock price movements...")
    stock_prices = generate_realistic_stock_prices(
        base_price=150.0,
        n_points=100,
        volatility=0.015
    )
    
    print(f"✓ Generated {len(stock_prices)} price points")
    print(f"✓ Starting price: ${stock_prices[0]:.2f}")
    print(f"✓ Ending price: ${stock_prices[-1]:.2f}")
    print()
    
    # Warm up predictor with initial prices
    warmup_size = 20
    for price in stock_prices[:warmup_size]:
        predictor.add_price(price)
    
    print("Running predictions with different modes...")
    print("-" * 70)
    
    # Track results for each mode
    mode_results = {
        'FAST': {'predictions': [], 'latencies': [], 'errors': []},
        'BALANCED': {'predictions': [], 'latencies': [], 'errors': []},
        'ACCURATE': {'predictions': [], 'latencies': [], 'errors': []},
        'AUTO': {'predictions': [], 'latencies': [], 'errors': [], 'modes_used': []}
    }
    
    # Run predictions on remaining data
    test_prices = stock_prices[warmup_size:]
    
    for i, actual_price in enumerate(test_prices[:-1]):
        current_price = actual_price
        next_price = test_prices[i + 1]
        
        # Test all modes
        for mode in ['FAST', 'BALANCED', 'ACCURATE', 'AUTO']:
            pred, latency, used_mode = predictor.predict(current_price, mode=mode)
            error = abs(pred - next_price) / next_price * 100
            
            mode_results[mode]['predictions'].append(pred)
            mode_results[mode]['latencies'].append(latency)
            mode_results[mode]['errors'].append(error)
            
            if mode == 'AUTO':
                mode_results[mode]['modes_used'].append(used_mode)
        
        # Add actual price to history for next iteration
        predictor.add_price(current_price)
        
        # Print sample prediction every 20 steps
        if i % 20 == 0:
            print(f"\nStep {i+1}:")
            print(f"  Current Price: ${current_price:.2f}")
            print(f"  Next Actual:   ${next_price:.2f}")
            print(f"  Volatility:    {predictor.calculate_volatility():.4f}")
            print(f"  Predictions:")
            for mode in ['FAST', 'BALANCED', 'ACCURATE']:
                pred = mode_results[mode]['predictions'][-1]
                lat = mode_results[mode]['latencies'][-1]
                err = mode_results[mode]['errors'][-1]
                print(f"    {mode:10s}: ${pred:7.2f} | Latency: {lat:6.2f}ms | Error: {err:5.2f}%")
    
    # Final performance summary
    print("\n" + "=" * 70)
    print("FINAL PERFORMANCE SUMMARY")
    print("=" * 70)
    print()
    
    print(f"{'Mode':<12} {'Avg Error':<12} {'Avg Latency':<15} {'Accuracy':<12}")
    print("-" * 70)
    
    for mode in ['FAST', 'BALANCED', 'ACCURATE', 'AUTO']:
        avg_error = np.mean(mode_results[mode]['errors'])
        avg_latency = np.mean(mode_results[mode]['latencies'])
        accuracy = 100 - avg_error
        
        print(f"{mode:<12} {avg_error:>6.3f}%     {avg_latency:>8.3f} ms      {accuracy:>6.2f}%")
    
    # AUTO mode breakdown
    if mode_results['AUTO']['modes_used']:
        print("\nAUTO Mode Breakdown:")
        mode_counts = pd.Series(mode_results['AUTO']['modes_used']).value_counts()
        for mode, count in mode_counts.items():
            percentage = count / len(mode_results['AUTO']['modes_used']) * 100
            print(f"  {mode}: {count} times ({percentage:.1f}%)")
    
    # Latency vs Accuracy Tradeoff Analysis
    print("\n" + "=" * 70)
    print("LATENCY vs ACCURACY TRADEOFF ANALYSIS")
    print("=" * 70)
    
    fast_speedup = np.mean(mode_results['ACCURATE']['latencies']) / np.mean(mode_results['FAST']['latencies'])
    fast_accuracy_loss = np.mean(mode_results['FAST']['errors']) - np.mean(mode_results['ACCURATE']['errors'])
    
    balanced_speedup = np.mean(mode_results['ACCURATE']['latencies']) / np.mean(mode_results['BALANCED']['latencies'])
    balanced_accuracy_loss = np.mean(mode_results['BALANCED']['errors']) - np.mean(mode_results['ACCURATE']['errors'])
    
    print(f"\nFAST mode:")
    print(f"  ✓ {fast_speedup:.1f}x faster than ACCURATE mode")
    print(f"  ✗ {abs(fast_accuracy_loss):.3f}% {'higher' if fast_accuracy_loss > 0 else 'lower'} error")
    
    print(f"\nBALANCED mode:")
    print(f"  ✓ {balanced_speedup:.1f}x faster than ACCURATE mode")
    print(f"  ✗ {abs(balanced_accuracy_loss):.3f}% {'higher' if balanced_accuracy_loss > 0 else 'lower'} error")
    
    print(f"\nRecommendation:")
    if predictor.calculate_volatility() > 0.02:
        print("  → High volatility detected: Use FAST mode for quick reactions")
    elif predictor.calculate_volatility() < 0.005:
        print("  → Low volatility detected: Use ACCURATE mode for best predictions")
    else:
        print("  → Medium volatility: BALANCED mode offers best tradeoff")
    print("  → AUTO mode adapts dynamically to market conditions")
    
    print("\n" + "=" * 70)


def interactive_trader_mode():
    """
    Interactive mode where traders can input real-time prices
    and get predictions with latency information
    """
    print("=" * 70)
    print("INTERACTIVE STOCK TRADING PREDICTION ENGINE")
    print("=" * 70)
    print()
    print("Welcome! This engine provides real-time predictions with latency tracking.")
    print()
    
    # Get initial setup from trader
    print("Initial Setup:")
    print("-" * 70)
    
    stock_symbol = input("Enter stock symbol (e.g., AAPL, TSLA): ").upper()
    
    while True:
        try:
            initial_price = float(input("Enter current stock price ($): "))
            if initial_price <= 0:
                print("  ✗ Price must be positive. Try again.")
                continue
            break
        except ValueError:
            print("  ✗ Invalid input. Please enter a number.")
    
    # Initialize predictor
    predictor = LatencyAwarePredictor(window_size=50)
    predictor.add_price(initial_price)
    
    print(f"\n✓ Initialized predictor for {stock_symbol} at ${initial_price:.2f}")
    print()
    
    # Ask for prediction mode preference
    print("Prediction Modes:")
    print("  1. FAST     - ~10ms latency, quick decisions")
    print("  2. BALANCED - ~50ms latency, good tradeoff")
    print("  3. ACCURATE - ~200ms latency, best accuracy")
    print("  4. AUTO     - Adaptive mode based on volatility")
    print()
    
    mode_map = {'1': 'FAST', '2': 'BALANCED', '3': 'ACCURATE', '4': 'AUTO'}
    while True:
        mode_choice = input("Select default mode (1-4) or press Enter for AUTO: ").strip()
        if mode_choice == '':
            mode_choice = '4'
        if mode_choice in mode_map:
            default_mode = mode_map[mode_choice]
            break
        print("  ✗ Invalid choice. Please enter 1-4.")
    
    print(f"\n✓ Using {default_mode} mode by default")
    print()
    print("=" * 70)
    print("LIVE TRADING SESSION")
    print("=" * 70)
    print("Commands:")
    print("  • Enter price: Type the current price and press Enter")
    print("  • 'mode': Change prediction mode")
    print("  • 'stats': View performance statistics")
    print("  • 'help': Show available commands")
    print("  • 'quit': Exit the session")
    print("-" * 70)
    print()
    
    session_predictions = []
    session_actuals = []
    last_prediction = None
    trade_count = 0
    
    while True:
        # Get user input
        user_input = input(f"{stock_symbol} > ").strip().lower()
        
        # Handle commands
        if user_input == 'quit' or user_input == 'exit':
            print("\nExiting trading session...")
            break
        
        elif user_input == 'help':
            print("\nAvailable Commands:")
            print("  • Enter a number: Record new price and get prediction")
            print("  • mode: Change prediction mode (FAST/BALANCED/ACCURATE/AUTO)")
            print("  • stats: View session performance statistics")
            print("  • help: Show this help message")
            print("  • quit/exit: End the trading session")
            print()
            continue
        
        elif user_input == 'mode':
            print("\nSelect new prediction mode:")
            print("  1. FAST")
            print("  2. BALANCED")
            print("  3. ACCURATE")
            print("  4. AUTO")
            mode_choice = input("Enter choice (1-4): ").strip()
            if mode_choice in mode_map:
                default_mode = mode_map[mode_choice]
                print(f"✓ Mode changed to {default_mode}")
            else:
                print("✗ Invalid choice. Mode unchanged.")
            print()
            continue
        
        elif user_input == 'stats':
            print("\n" + "=" * 70)
            print("SESSION PERFORMANCE STATISTICS")
            print("=" * 70)
            
            if len(session_predictions) > 0:
                errors = np.abs(np.array(session_predictions) - np.array(session_actuals))
                mape = np.mean(errors / np.array(session_actuals)) * 100
                
                print(f"\nTotal Predictions: {len(session_predictions)}")
                print(f"Average Error: {mape:.3f}%")
                print(f"Current Volatility: {predictor.calculate_volatility():.4f}")
                print(f"Price Range: ${min(predictor.price_history):.2f} - ${max(predictor.price_history):.2f}")
                
                if len(session_predictions) >= 5:
                    recent_errors = errors[-5:]
                    recent_mape = np.mean(recent_errors / np.array(session_actuals[-5:])) * 100
                    print(f"Recent 5 Predictions Error: {recent_mape:.3f}%")
            else:
                print("\nNo predictions made yet. Enter prices to start trading.")
            
            print("=" * 70)
            print()
            continue
        
        # Try to parse as price input
        try:
            current_price = float(user_input)
            
            if current_price <= 0:
                print("✗ Price must be positive.")
                continue
            
            trade_count += 1
            
            # If we had a previous prediction, evaluate it
            if last_prediction is not None:
                pred_price = last_prediction['price']
                error = abs(pred_price - current_price) / current_price * 100
                session_predictions.append(pred_price)
                session_actuals.append(current_price)
                
                print(f"\n📊 Prediction Evaluation:")
                print(f"  Predicted: ${pred_price:.2f}")
                print(f"  Actual:    ${current_price:.2f}")
                print(f"  Error:     {error:.3f}%")
                
                if error < 0.5:
                    print("  ✓ Excellent prediction!")
                elif error < 1.0:
                    print("  ✓ Good prediction")
                elif error < 2.0:
                    print("  ⚠ Moderate prediction")
                else:
                    print("  ✗ Poor prediction")
            
            # Add current price to history
            predictor.add_price(current_price)
            
            # Make new prediction
            prediction, latency, used_mode = predictor.predict(current_price, mode=default_mode)
            
            # Calculate market metrics
            volatility = predictor.calculate_volatility()
            
            if len(predictor.price_history) >= 2:
                price_change = (current_price - list(predictor.price_history)[-2]) / list(predictor.price_history)[-2] * 100
            else:
                price_change = 0
            
            # Display prediction results
            print(f"\n{'─' * 70}")
            print(f"Trade #{trade_count} | {stock_symbol} @ ${current_price:.2f}")
            print(f"{'─' * 70}")
            print(f"Mode: {used_mode} | Latency: {latency:.2f}ms")
            print(f"Volatility: {volatility:.4f} | Price Change: {price_change:+.2f}%")
            print()
            print(f"📈 NEXT PRICE PREDICTION: ${prediction:.2f}")
            
            # Provide trading recommendation
            price_diff = prediction - current_price
            price_diff_pct = (price_diff / current_price) * 100
            
            print(f"Expected Change: {price_diff:+.2f} ({price_diff_pct:+.2f}%)")
            print()
            
            if abs(price_diff_pct) < 0.3:
                recommendation = "HOLD - Minimal movement expected"
                signal = "➡️"
            elif price_diff_pct > 1.0:
                recommendation = "STRONG BUY - Significant upward movement expected"
                signal = "🚀"
            elif price_diff_pct > 0.3:
                recommendation = "BUY - Upward movement expected"
                signal = "📈"
            elif price_diff_pct < -1.0:
                recommendation = "STRONG SELL - Significant downward movement expected"
                signal = "⚠️"
            elif price_diff_pct < -0.3:
                recommendation = "SELL - Downward movement expected"
                signal = "📉"
            else:
                recommendation = "HOLD - Minimal movement expected"
                signal = "➡️"
            
            print(f"{signal} {recommendation}")
            print(f"{'─' * 70}")
            print()
            
            # Store prediction for next evaluation
            last_prediction = {
                'price': prediction,
                'mode': used_mode,
                'latency': latency
            }
            
        except ValueError:
            print(f"✗ Unknown command: '{user_input}'. Type 'help' for available commands.")
            print()
    
    # Final session summary
    if len(session_predictions) > 0:
        print("\n" + "=" * 70)
        print("FINAL SESSION SUMMARY")
        print("=" * 70)
        
        errors = np.abs(np.array(session_predictions) - np.array(session_actuals))
        mape = np.mean(errors / np.array(session_actuals)) * 100
        
        print(f"\nStock Symbol: {stock_symbol}")
        print(f"Total Trades: {trade_count}")
        print(f"Predictions Made: {len(session_predictions)}")
        print(f"Average Prediction Error: {mape:.3f}%")
        print(f"Prediction Accuracy: {100 - mape:.2f}%")
        print(f"Final Price: ${current_price:.2f}")
        print(f"Session Volatility: {predictor.calculate_volatility():.4f}")
        
        print("\n" + "=" * 70)
    
    print("\nThank you for using the Latency-Aware Market Prediction Engine!")


def main_menu():
    """Main menu for selecting operation mode"""
    print("\n" + "=" * 70)
    print("LATENCY-AWARE MARKET PREDICTION ENGINE")
    print("=" * 70)
    print()
    print("Select Mode:")
    print("  1. Interactive Trader Mode - Input real-time prices")
    print("  2. Demo Mode - Run automated simulation")
    print("  3. Exit")
    print()
    
    while True:
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            interactive_trader_mode()
            break
        elif choice == '2':
            run_market_simulation()
            break
        elif choice == '3':
            print("\nGoodbye!")
            break
        else:
            print("✗ Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main_menu()
