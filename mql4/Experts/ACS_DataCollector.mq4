//+------------------------------------------------------------------+
//|                                            ACS_DataCollector.mq4 |
//|                                        ACS-ML Trading System     |
//|                       Data Collection EA for Dynamic Forex28    |
//+------------------------------------------------------------------+
#property copyright "ACS-ML Trading System"
#property link      ""
#property version   "1.00"
#property strict

//--- Input parameters
input int ExportIntervalSeconds = 60;        // Export interval in seconds
input string ExportPath = "data\\mt4_exports\\";  // Export directory path
input bool ExportIndicatorData = true;       // Export indicator signals
input bool ExportPriceData = true;          // Export OHLCV data
input bool EnableLogging = true;            // Enable detailed logging

//--- Global variables
datetime lastExportTime = 0;
string indicatorName = "Dynamic_Forex28_Navigator";
string currencies[8] = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"};

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("ACS Data Collector EA initialized");
    Print("Export interval: ", ExportIntervalSeconds, " seconds");
    Print("Export path: ", ExportPath);
    
    // Create export directory if it doesn't exist
    CreateDirectory(ExportPath);
    
    // Verify that Dynamic Forex28 Navigator is available
    if (!IsIndicatorAvailable())
    {
        Print("ERROR: Dynamic Forex28 Navigator indicator not found!");
        Print("Please ensure the indicator is installed and running on this chart.");
        return INIT_FAILED;
    }
    
    // Initialize export timer
    lastExportTime = TimeCurrent();
    
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("ACS Data Collector EA stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // Check if it's time to export data
    if (TimeCurrent() - lastExportTime >= ExportIntervalSeconds)
    {
        ExportData();
        lastExportTime = TimeCurrent();
    }
}

//+------------------------------------------------------------------+
//| Main data export function                                        |
//+------------------------------------------------------------------+
void ExportData()
{
    if (EnableLogging)
        Print("Starting data export at: ", TimeToString(TimeCurrent()));
    
    if (ExportIndicatorData)
        ExportIndicatorSignals();
    
    if (ExportPriceData)
        ExportOHLCVData();
    
    if (EnableLogging)
        Print("Data export completed");
}

//+------------------------------------------------------------------+
//| Export Dynamic Forex28 Navigator indicator data                  |
//+------------------------------------------------------------------+
void ExportIndicatorSignals()
{
    string filename = ExportPath + "indicator_signals_" + Symbol() + "_" + 
                     StringSubstr(TimeToString(TimeCurrent(), TIME_DATE), 0, 10) + ".csv";
    
    int handle = FileOpen(filename, FILE_WRITE | FILE_CSV | FILE_COMMON);
    
    if (handle == INVALID_HANDLE)
    {
        Print("ERROR: Failed to open indicator export file: ", filename);
        return;
    }
    
    // Write CSV header (first time only)
    if (FileSize(handle) == 0)
    {
        WriteIndicatorHeader(handle);
    }
    
    // Write current indicator values
    WriteIndicatorData(handle);
    
    FileClose(handle);
    
    if (EnableLogging)
        Print("Indicator data exported to: ", filename);
}

//+------------------------------------------------------------------+
//| Export OHLCV price data                                          |
//+------------------------------------------------------------------+
void ExportOHLCVData()
{
    string filename = ExportPath + "ohlcv_" + Symbol() + "_" + 
                     StringSubstr(TimeToString(TimeCurrent(), TIME_DATE), 0, 10) + ".csv";
    
    int handle = FileOpen(filename, FILE_WRITE | FILE_CSV | FILE_COMMON);
    
    if (handle == INVALID_HANDLE)
    {
        Print("ERROR: Failed to open OHLCV export file: ", filename);
        return;
    }
    
    // Write CSV header (first time only)
    if (FileSize(handle) == 0)
    {
        FileWrite(handle, "timestamp", "open", "high", "low", "close", "volume", "symbol");
    }
    
    // Write current bar data
    FileWrite(handle, 
              TimeToString(Time[0], TIME_DATE | TIME_MINUTES),
              DoubleToString(Open[0], Digits),
              DoubleToString(High[0], Digits),
              DoubleToString(Low[0], Digits),
              DoubleToString(Close[0], Digits),
              IntegerToString(Volume[0]),
              Symbol());
    
    FileClose(handle);
    
    if (EnableLogging && TimeCurrent() % 300 == 0)  // Log every 5 minutes to avoid spam
        Print("OHLCV data exported to: ", filename);
}

//+------------------------------------------------------------------+
//| Write indicator CSV header                                        |
//+------------------------------------------------------------------+
void WriteIndicatorHeader(int handle)
{
    string header = "timestamp";
    
    // Add currency strength columns
    for (int i = 0; i < 8; i++)
    {
        header += "," + currencies[i] + "_strength";
    }
    
    // Add indicator signal columns
    header += ",buy_momentum_currency,buy_momentum_value";
    header += ",sell_momentum_currency,sell_momentum_value";
    header += ",market_activity";
    header += ",dual_momentum_pairs";
    header += ",outer_range_warning";
    header += ",hook_alert,xstop_signal";
    header += ",mfib_23,mfib_100,mfib_161,mfib_261";
    
    FileWriteString(handle, header + "\n");
}

//+------------------------------------------------------------------+
//| Write current indicator data                                      |
//+------------------------------------------------------------------+
void WriteIndicatorData(int handle)
{
    string dataLine = TimeToString(Time[0], TIME_DATE | TIME_MINUTES);
    
    // Extract currency strength values from indicator
    // Note: These buffer indices are estimates and need to be adjusted
    // based on the actual Dynamic Forex28 Navigator buffer structure
    for (int i = 0; i < 8; i++)
    {
        double strength = iCustom(NULL, 0, indicatorName, i, 0);
        dataLine += "," + DoubleToString(strength, 2);
    }
    
    // Extract additional signals (buffer indices need verification)
    double buyMomentum = iCustom(NULL, 0, indicatorName, 8, 0);      // Buy momentum value
    double sellMomentum = iCustom(NULL, 0, indicatorName, 9, 0);     // Sell momentum value
    double marketActivity = iCustom(NULL, 0, indicatorName, 10, 0);  // Market activity state
    
    dataLine += ",BUY," + DoubleToString(buyMomentum, 2);
    dataLine += ",SELL," + DoubleToString(sellMomentum, 2);
    dataLine += "," + DoubleToString(marketActivity, 0);
    
    // Placeholder for additional signals
    dataLine += ",NONE";  // dual_momentum_pairs
    dataLine += ",0";     // outer_range_warning
    dataLine += ",0,0";   // hook_alert, xstop_signal
    
    // Market Fibonacci levels (need correct buffer indices)
    double mfib23 = iCustom(NULL, 0, indicatorName, 11, 0);
    double mfib100 = iCustom(NULL, 0, indicatorName, 12, 0);
    double mfib161 = iCustom(NULL, 0, indicatorName, 13, 0);
    double mfib261 = iCustom(NULL, 0, indicatorName, 14, 0);
    
    dataLine += "," + DoubleToString(mfib23, 2);
    dataLine += "," + DoubleToString(mfib100, 2);
    dataLine += "," + DoubleToString(mfib161, 2);
    dataLine += "," + DoubleToString(mfib261, 2);
    
    FileWriteString(handle, dataLine + "\n");
}

//+------------------------------------------------------------------+
//| Check if Dynamic Forex28 Navigator indicator is available        |
//+------------------------------------------------------------------+
bool IsIndicatorAvailable()
{
    // Try to read from indicator buffer
    double testValue = iCustom(NULL, 0, indicatorName, 0, 0);
    
    // Check if we got a valid value (not empty value)
    if (testValue != EMPTY_VALUE && testValue != 0.0)
    {
        return true;
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| Create directory if it doesn't exist                             |
//+------------------------------------------------------------------+
void CreateDirectory(string path)
{
    // MT4 will create directory automatically when writing files
    // This is a placeholder for directory creation logic
    Print("Export directory set to: ", path);
}

//+------------------------------------------------------------------+
//| Timer function (alternative to tick-based export)               |
//+------------------------------------------------------------------+
void OnTimer()
{
    ExportData();
}

//+------------------------------------------------------------------+