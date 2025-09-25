//+------------------------------------------------------------------+
//|                                       ACS_ML_DataCollector.mq4  |
//|                                    Copyright 2025, ACS-ML Team |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, ACS-ML Team"
#property link      ""
#property version   "1.0"
#property strict

// Input parameters
input int DataCollectionIntervalMinutes = 60;  // Data collection interval in minutes
input string DataExportPath = "ACS_ML\\";       // Subfolder for exported data
input bool ExportIndicatorSignals = true;       // Export Dynamic Forex28 signals
input bool ExportPriceData = true;              // Export OHLCV data
input string IndicatorName = "Dynamic_Forex28_Navigator"; // Exact indicator name

// Global variables
datetime lastCollectionTime = 0;
int fileHandle = -1;
string currentPair;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("ACS-ML Data Collector EA initialized");
    Print("Collection interval: ", DataCollectionIntervalMinutes, " minutes");
    Print("Export path: ", DataExportPath);
    
    currentPair = Symbol();
    
    // Create export directory
    if(!FileIsExist(DataExportPath, FILE_COMMON))
    {
        if(!FileCreateFolder(DataExportPath, FILE_COMMON))
        {
            Print("Failed to create export directory: ", DataExportPath);
            return INIT_FAILED;
        }
    }
    
    Print("Data collector ready for pair: ", currentPair);
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                               |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    if(fileHandle >= 0)
    {
        FileClose(fileHandle);
        fileHandle = -1;
    }
    Print("ACS-ML Data Collector EA deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                            |
//+------------------------------------------------------------------+
void OnTick()
{
    // Check if it's time to collect data
    if(TimeCurrent() - lastCollectionTime >= DataCollectionIntervalMinutes * 60)
    {
        CollectData();
        lastCollectionTime = TimeCurrent();
    }
}

//+------------------------------------------------------------------+
//| Data collection function                                        |
//+------------------------------------------------------------------+
void CollectData()
{
    if(ExportIndicatorSignals)
    {
        ExportIndicatorData();
    }
    
    if(ExportPriceData)
    {
        ExportPriceData();
    }
}

//+------------------------------------------------------------------+
//| Export Dynamic Forex28 Navigator indicator signals             |
//+------------------------------------------------------------------+
void ExportIndicatorData()
{
    string filename = DataExportPath + currentPair + "_signals_" + TimeToString(TimeCurrent(), TIME_DATE) + ".csv";
    
    // Open file for append (create header if new)
    bool isNewFile = !FileIsExist(filename, FILE_COMMON);
    fileHandle = FileOpen(filename, FILE_WRITE|FILE_CSV|FILE_COMMON|FILE_READ);
    
    if(fileHandle < 0)
    {
        Print("Failed to open indicator signals file: ", filename);
        return;
    }
    
    // Write header for new file
    if(isNewFile)
    {
        WriteIndicatorHeader();
    }
    
    // Move to end of file
    FileSeek(fileHandle, 0, SEEK_END);
    
    // Collect current indicator values
    WriteIndicatorData();
    
    FileClose(fileHandle);
    fileHandle = -1;
    
    Print("Exported indicator signals to: ", filename);
}

//+------------------------------------------------------------------+
//| Write indicator data header                                     |
//+------------------------------------------------------------------+
void WriteIndicatorHeader()
{
    FileWrite(fileHandle, 
        "timestamp",
        "USD_strength", "EUR_strength", "GBP_strength", "JPY_strength",
        "CHF_strength", "CAD_strength", "AUD_strength", "NZD_strength",
        "market_activity",
        "buy_momentum_currency", "buy_momentum_value",
        "sell_momentum_currency", "sell_momentum_value",
        "dual_momentum_signal",
        "outer_range_warning",
        "hook_alert",
        "xstop_signal",
        "mfib_23", "mfib_100", "mfib_161", "mfib_261"
    );
}

//+------------------------------------------------------------------+
//| Write current indicator data                                    |
//+------------------------------------------------------------------+
void WriteIndicatorData()
{
    // Get indicator values using iCustom
    // Note: Buffer indices need to be determined by examining the indicator
    // These are placeholder values - adjust based on actual indicator buffers
    
    double usd_strength = iCustom(NULL, 0, IndicatorName, 0, 1);  // Buffer 0 = USD
    double eur_strength = iCustom(NULL, 0, IndicatorName, 1, 1);  // Buffer 1 = EUR
    double gbp_strength = iCustom(NULL, 0, IndicatorName, 2, 1);  // Buffer 2 = GBP
    double jpy_strength = iCustom(NULL, 0, IndicatorName, 3, 1);  // Buffer 3 = JPY
    double chf_strength = iCustom(NULL, 0, IndicatorName, 4, 1);  // Buffer 4 = CHF
    double cad_strength = iCustom(NULL, 0, IndicatorName, 5, 1);  // Buffer 5 = CAD
    double aud_strength = iCustom(NULL, 0, IndicatorName, 6, 1);  // Buffer 6 = AUD
    double nzd_strength = iCustom(NULL, 0, IndicatorName, 7, 1);  // Buffer 7 = NZD
    
    // Market activity and signals (these buffer indices need to be confirmed)
    double market_activity = iCustom(NULL, 0, IndicatorName, 8, 1);
    double momentum_signal = iCustom(NULL, 0, IndicatorName, 9, 1);
    double dual_momentum = iCustom(NULL, 0, IndicatorName, 10, 1);
    
    // Additional signal buffers
    double outer_range = iCustom(NULL, 0, IndicatorName, 11, 1);
    double hook_alert = iCustom(NULL, 0, IndicatorName, 12, 1);
    double xstop_signal = iCustom(NULL, 0, IndicatorName, 13, 1);
    
    // Find strongest buy/sell momentum
    string buy_currency = "N/A";
    double buy_value = 0;
    string sell_currency = "N/A";
    double sell_value = 0;
    
    // Simple logic to find strongest currencies (enhance based on indicator logic)
    string currencies[8] = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"};
    double strengths[8] = {usd_strength, eur_strength, gbp_strength, jpy_strength, 
                          chf_strength, cad_strength, aud_strength, nzd_strength};
    
    for(int i = 0; i < 8; i++)
    {
        if(strengths[i] > buy_value)
        {
            buy_value = strengths[i];
            buy_currency = currencies[i];
        }
        if(strengths[i] < sell_value)
        {
            sell_value = strengths[i];
            sell_currency = currencies[i];
        }
    }
    
    // Write data row
    FileWrite(fileHandle,
        TimeToString(Time[1], TIME_DATE|TIME_MINUTES),  // Previous bar timestamp
        DoubleToString(usd_strength, 2),
        DoubleToString(eur_strength, 2),
        DoubleToString(gbp_strength, 2),
        DoubleToString(jpy_strength, 2),
        DoubleToString(chf_strength, 2),
        DoubleToString(cad_strength, 2),
        DoubleToString(aud_strength, 2),
        DoubleToString(nzd_strength, 2),
        DoubleToString(market_activity, 0),
        buy_currency,
        DoubleToString(buy_value, 2),
        sell_currency,
        DoubleToString(sell_value, 2),
        DoubleToString(dual_momentum, 0),
        DoubleToString(outer_range, 0),
        DoubleToString(hook_alert, 0),
        DoubleToString(xstop_signal, 0),
        "23", "100", "161", "261"  // Static MFib levels for now
    );
}

//+------------------------------------------------------------------+
//| Export OHLCV price data                                        |
//+------------------------------------------------------------------+
void ExportPriceData()
{
    string filename = DataExportPath + currentPair + "_prices_" + TimeToString(TimeCurrent(), TIME_DATE) + ".csv";
    
    bool isNewFile = !FileIsExist(filename, FILE_COMMON);
    fileHandle = FileOpen(filename, FILE_WRITE|FILE_CSV|FILE_COMMON|FILE_READ);
    
    if(fileHandle < 0)
    {
        Print("Failed to open price data file: ", filename);
        return;
    }
    
    // Write header for new file
    if(isNewFile)
    {
        FileWrite(fileHandle, "timestamp", "open", "high", "low", "close", "volume");
    }
    
    // Move to end of file
    FileSeek(fileHandle, 0, SEEK_END);
    
    // Write current bar data (previous completed bar)
    FileWrite(fileHandle,
        TimeToString(Time[1], TIME_DATE|TIME_MINUTES),
        DoubleToString(Open[1], Digits),
        DoubleToString(High[1], Digits),
        DoubleToString(Low[1], Digits),
        DoubleToString(Close[1], Digits),
        IntegerToString(Volume[1])
    );
    
    FileClose(fileHandle);
    fileHandle = -1;
    
    Print("Exported price data to: ", filename);
}