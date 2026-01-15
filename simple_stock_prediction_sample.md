======================================================================
LATENCY-AWARE MARKET PREDICTION ENGINE
======================================================================

Select Mode:
  1. Interactive Trader Mode - Input real-time prices
  2. Demo Mode - Run automated simulation
  3. Exit

Enter your choice (1-3):  aapl
✗ Invalid choice. Please enter 1, 2, or 3.
Enter your choice (1-3):  1
======================================================================
INTERACTIVE STOCK TRADING PREDICTION ENGINE
======================================================================

Welcome! This engine provides real-time predictions with latency tracking.

Initial Setup:
----------------------------------------------------------------------
Enter stock symbol (e.g., AAPL, TSLA):  tsla
Enter current stock price ($):  5

✓ Initialized predictor for TSLA at $5.00

Prediction Modes:
  1. FAST     - ~10ms latency, quick decisions
  2. BALANCED - ~50ms latency, good tradeoff
  3. ACCURATE - ~200ms latency, best accuracy
  4. AUTO     - Adaptive mode based on volatility

Select default mode (1-4) or press Enter for AUTO:  1

✓ Using FAST mode by default

======================================================================
LIVE TRADING SESSION
======================================================================
Commands:
  • Enter price: Type the current price and press Enter
  • 'mode': Change prediction mode
  • 'stats': View performance statistics
  • 'help': Show available commands
  • 'quit': Exit the session
----------------------------------------------------------------------

TSLA >  5

──────────────────────────────────────────────────────────────────────
Trade #1 | TSLA @ $5.00
──────────────────────────────────────────────────────────────────────
Mode: FAST | Latency: 0.00ms
Volatility: 0.0000 | Price Change: +0.00%

📈 NEXT PRICE PREDICTION: $5.00
Expected Change: +0.00 (+0.00%)

➡️ HOLD - Minimal movement expected
──────────────────────────────────────────────────────────────────────

TSLA >  6

📊 Prediction Evaluation:
  Predicted: $5.00
  Actual:    $6.00
  Error:     16.667%
  ✗ Poor prediction

──────────────────────────────────────────────────────────────────────
Trade #2 | TSLA @ $6.00
──────────────────────────────────────────────────────────────────────
Mode: FAST | Latency: 0.00ms
Volatility: 0.1000 | Price Change: +20.00%

📈 NEXT PRICE PREDICTION: $6.00
Expected Change: +0.00 (+0.00%)

➡️ HOLD - Minimal movement expected
──────────────────────────────────────────────────────────────────────

TSLA >  16

📊 Prediction Evaluation:
  Predicted: $6.00
  Actual:    $16.00
  Error:     62.500%
  ✗ Poor prediction

──────────────────────────────────────────────────────────────────────
Trade #3 | TSLA @ $16.00
──────────────────────────────────────────────────────────────────────
Mode: FAST | Latency: 0.00ms
Volatility: 0.7430 | Price Change: +166.67%

📈 NEXT PRICE PREDICTION: $16.00
Expected Change: +0.00 (+0.00%)

➡️ HOLD - Minimal movement expected
──────────────────────────────────────────────────────────────────────

TSLA >  1

📊 Prediction Evaluation:
  Predicted: $16.00
  Actual:    $1.00
  Error:     1500.000%
  ✗ Poor prediction

──────────────────────────────────────────────────────────────────────
Trade #4 | TSLA @ $1.00
──────────────────────────────────────────────────────────────────────
Mode: FAST | Latency: 0.06ms
Volatility: 0.9329 | Price Change: -93.75%

📈 NEXT PRICE PREDICTION: $6.60
Expected Change: +5.60 (+560.00%)

🚀 STRONG BUY - Significant upward movement expected
──────────────────────────────────────────────────────────────────────

TSLA > 
↑↓ for history. Search history with c-↑/c-↓

Click to add a cell.


TSLA >  
✗ Unknown command: ''. Type 'help' for available commands.

TSLA >  help

Available Commands:
  • Enter a number: Record new price and get prediction
  • mode: Change prediction mode (FAST/BALANCED/ACCURATE/AUTO)
  • stats: View session performance statistics
  • help: Show this help message
  • quit/exit: End the trading session

TSLA >  quit

Exiting trading session...

======================================================================
FINAL SESSION SUMMARY
======================================================================

Stock Symbol: TSLA
Total Trades: 4
Predictions Made: 3
Average Prediction Error: 526.389%
Prediction Accuracy: -426.39%
Final Price: $1.00
Session Volatility: 0.9329

======================================================================

Thank you for using the Latency-Aware Market Prediction Engine!

