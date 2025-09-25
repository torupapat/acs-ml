# MQL5 Black-Box Indicator Optimization with Machine Learning

## Project Description

This project explores a novel approach to optimizing a rented MQL5 indicator, "Dynamic Forex28 Navigator" using machine learning. Without access to the indicator's source code, we treat it as a "black box" and use its output signals as key features for a predictive model. The goal is to build a robust trading strategy by intelligently filtering the indicator's signals and identifying the best complementary indicators.

This repository contains the necessary code for data collection, feature engineering, model training, and integration with the MetaTrader 5 platform.

## Key Features

    Black-Box Optimization: Uses the output of a rented MQL5 indicator as a primary feature for an ML model.

    Intelligent Signal Filtering: A machine learning model learns to identify profitable trading opportunities by validating the indicator's signals against market conditions.

    Complementary Indicator Discovery: Employs feature selection techniques to automatically find and select the most effective technical indicators to pair with the rented one.

    MQL5 Integration: Provides a pipeline to collect data from MetaTrader 4 and send trading signals back to the platform.

## Project Structure

.
├── data/
│   ├── raw/                 # Raw data from MQL5 (e.g., price, indicator outputs)
│   └── processed/           # Processed and cleaned data for ML model training
├── mql5/
│   ├── Experts/             # MQL5 Expert Advisor to run the strategy
│   └── Indicators/          # (Optional) Custom indicators for data collection
├── scripts/
│   ├── data_collector.py    # Python script to collect data from MT5
│   └── model_trainer.py     # Script to train and evaluate the ML model
└── README.md                # This file

# Getting Started

## Prerequisites

    MetaTrader 4 (MT4): You must have the MT4 terminal installed and running.

    Rented Indicator: You need to have the "Dynamic Forex28" indicator rented and installed on your MT5 terminal.

    Python 3.x: A compatible Python environment is required for the ML model and data processing.

    Python Libraries:

        pandas

        scikit-learn

        tensorflow or pytorch (or another ML library of your choice)

        MetaTrader4 (I don't know if there's a specific library for MT4, futher research may be needed)


## Usage

    Data Collection:

        ***The indicator running on MT4 is rented and cannot be shared. You must rent it yourself from the MQL5 Market.***

        Run the MT4 Expert Advisor to start collecting historical data and live indicator outputs. The script will save the data in the data/raw directory.

        Allow the script to run for a sufficient period to gather enough data for training (at least several weeks to a few months for robust results).

    Model Training:

        Run the scripts/model_trainer.py to preprocess the data and train your ML model.

        The model will learn the most effective combination of the "Dynamic Forex28" outputs and other features for predicting price movements.

        The trained model and any derived trading rules will be saved.

    Deployment (MQL5 Expert Advisor):

        Implement the trained model's logic in the mql5/Experts/YourStrategyEA.mq5 file. This EA will:

            Read the real-time output of the "Dynamic Forex28" indicator.

            Pass this data to your model (e.g., by calling a Python script via WebRequest).

            Execute trades based on the signal returned by the ML model.

## Contributing

We welcome contributions! If you have suggestions for new features, bug fixes, or improvements, please feel free to open an issue or submit a pull request.

## References

    MetaTrader 4 Documentation: https://www.mql5.com/en/docs

    Dynamic Forex28 Navigator: https://www.mql5.com/en/market/product/122172

    Dynamic Forex28 User Guide: https://www.mql5.com/en/blogs/post/758844

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.