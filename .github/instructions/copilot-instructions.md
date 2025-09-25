# AI Coding Agent Instructions for ACS-ML Trading System

## Project Overview
This is a **black-box indicator optimization system** that treats the rented "Dynamic Forex28 Navigator" MT4 indicator as an opaque signal source for machine learning models. The system uses MT4 exclusively for trading and data collection.

## Architecture & Data Flow
```
MT4 Terminal → [Rented Indicator] → MT4 EA/Script Data Export → Python ML Pipeline → Trading Signals → MT4 Expert Advisor
```

**Critical constraint**: The core indicator is rented and cannot be modified or inspected - treat it as a pure signal source.

## Project Structure (Planned)
- `data/raw/` - Direct MT4 exports (OHLCV + indicator signals via CSV/files)
- `data/processed/` - ML-ready datasets with engineered features
- `scripts/data_collector.py` - Process MT4 exported data files
- `scripts/model_trainer.py` - ML pipeline (feature selection + model training)
- `mql4/Experts/` - Expert Advisors for data collection and signal execution
- `mql4/Scripts/` - Data extraction scripts that read indicator values

## Development Workflows

### Data Collection Phase
1. **Run MT4 with rented indicator** - Dynamic Forex28 Navigator must be active
2. **MT4 EA/Script extraction** - Create EA to export indicator buffer values + OHLCV
3. **File-based communication** - Use CSV exports since direct Python-MT4 integration is complex
4. **Accumulate 2-3 months minimum** - Required for robust ML training

### Indicator Signal Format (Key Data Points)
- **Currency strength values** (8 currencies: USD, EUR, GBP, JPY, CHF, CAD, AUD, NZD)
- **BUY/SELL Momentum arrows** - Strong momentum signals for individual currencies
- **Dual Momentum signals** - When base+quote currencies have opposing strength
- **Dynamic Market Fibonacci levels** - MFib 23, 100, 161, 261 (unique to this indicator)
- **Outer Range warnings** - Overbought/oversold at ±100 levels  
- **HOOK alerts** - Reversal signals when strength retreats from extremes
- **X-STOP signals** - Market exhaustion warnings
- **Market activity status** - 5 possible states including volatility warnings

### ML Training Phase
1. **Feature engineering** - Combine indicator signals with technical indicators
2. **Feature selection** - Auto-discover complementary indicators
3. **Model validation** - Use walk-forward analysis for time series data
4. **Signal filtering** - Train model to validate indicator signals

### Deployment Phase
1. **MQL4 Expert Advisor** - Reads real-time indicator output
2. **File-based integration** - EA reads ML predictions from CSV/files (no direct HTTP)
3. **Risk management** - Implement position sizing and stop-loss logic

## Critical Integration Points

### MT4 ↔ Python Communication
- **Data export**: Use MT4 EA to write indicator values + OHLCV to CSV files
- **Signal delivery**: Python writes trading signals to CSV files, EA reads them
- **Real-time constraints**: Indicator updates every tick - optimize for file I/O speed

### Black-Box Signal Processing
- **Never attempt to reverse-engineer** the rented indicator
- **Focus on signal validation** - when are the signals most reliable?
- **Combine with market context** - time of day, volatility, trend strength

## Project-Specific Patterns

### Time Series Considerations
- **No look-ahead bias** - Ensure all features use only past data
- **Walk-forward validation** - Split by time, not random sampling
- **Market regime awareness** - Model performance varies by market conditions

### MQL4/Python Integration
```mql4
// Pattern: Export indicator data from MT4 EA
int handle = FileOpen("indicator_signals.csv", FILE_WRITE|FILE_CSV);
FileWrite(handle, TimeToString(Time[0]), iCustom(NULL,0,"Dynamic_Forex28_Navigator",0,0), Close[0]);
FileClose(handle);

// Pattern: Read ML predictions in MT4 EA  
int signal_handle = FileOpen("ml_signals.csv", FILE_READ|FILE_CSV);
string signal = FileReadString(signal_handle);
FileClose(signal_handle);
```

### Feature Engineering Strategy
- **Indicator signal + market context** - Don't use indicator signals in isolation
- **Multi-timeframe analysis** - Combine signals across different periods
- **Volatility normalization** - Adjust signals based on market volatility

## External Dependencies
- **MetaTrader 4** - Primary platform, must be running
- **Dynamic Forex28 Navigator** - Core rented indicator (cannot be shared)
- **Python ML stack** - pandas, scikit-learn, tensorflow/pytorch
- **MT4/Python bridge** - Research MetaTrader4 package compatibility

## Development Guidelines
- **Backtest everything** - No live trading without extensive backtesting
- **Version control data pipelines** - Track feature engineering changes
- **Monitor indicator rental** - Ensure rented indicator remains active
- **Respect broker limitations** - Some brokers restrict API access/WebRequests

## Getting Started Checklist
1. ✅ Verify MT4 installation and rented indicator access
2. ⚠️ Create planned directory structure (`data/`, `scripts/`, `mql4/`)
3. ⚠️ Implement `data_collector.py` - Start with basic OHLCV + indicator export
4. ⚠️ Set up Python environment with ML dependencies
5. ⚠️ Build initial data collection pipeline before ML work

---
*This project is in early development - most implementation files don't exist yet. Focus on data collection infrastructure first.*