#!/usr/bin/env python3
"""
ML Model Trainer for ACS-ML Trading System

This script trains machine learning models on Dynamic Forex28 Navigator signals
to validate and filter the indicator's trading signals.

Features:
- Walk-forward analysis for time series data
- Multiple ML models (RandomForest, XGBoost, LSTM)
- Feature engineering with technical indicators
- Signal validation and filtering
- Model performance evaluation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import pandas_ta as ta  # Technical Analysis library

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_trainer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ACSMLTrainer:
    """Machine Learning trainer for ACS-ML trading system"""
    
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """Initialize the ML trainer with configuration"""
        self.config = self._load_config(config_path)
        self.processed_dir = Path(self.config['paths']['processed_data'])
        self.models_dir = Path(self.config['paths']['models'])
        
        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # ML components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        
        # Configuration
        self.currencies = self.config['indicators']['currencies']
        self.lookback_periods = self.config['ml']['lookback_periods']
        self.technical_indicators = self.config['ml']['technical_indicators']
        
        logger.info("ACS-ML Trainer initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file {config_path} not found")
            raise
    
    def load_processed_data(self) -> pd.DataFrame:
        """Load all processed data files and combine them"""
        logger.info("Loading processed data files...")
        
        # Find all processed CSV files
        csv_files = list(self.processed_dir.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} processed data files")
        
        if not csv_files:
            raise FileNotFoundError("No processed data files found. Run data collector first.")
        
        # Load and combine all CSV files
        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                dfs.append(df)
                logger.info(f"Loaded {len(df)} rows from {csv_file.name}")
            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {e}")
        
        if not dfs:
            raise ValueError("No valid data files could be loaded")
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, axis=0)
        combined_df = combined_df.sort_index()  # Sort by timestamp
        combined_df = combined_df.drop_duplicates()  # Remove duplicates
        
        logger.info(f"Combined dataset: {len(combined_df)} rows, {len(combined_df.columns)} columns")
        return combined_df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from the raw data"""
        logger.info("Engineering features...")
        
        feature_df = df.copy()
        
        # Technical indicators using TA library
        if 'close' in df.columns:
            # RSI
            if 'RSI' in self.technical_indicators:
                feature_df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            
            # MACD
            if 'MACD' in self.technical_indicators:
                macd = ta.trend.MACD(df['close'])
                feature_df['macd'] = macd.macd()
                feature_df['macd_signal'] = macd.macd_signal()
                feature_df['macd_diff'] = macd.macd_diff()
            
            # Bollinger Bands
            if 'Bollinger_Bands' in self.technical_indicators:
                bb = ta.volatility.BollingerBands(df['close'])
                feature_df['bb_high'] = bb.bollinger_hband()
                feature_df['bb_low'] = bb.bollinger_lband()
                feature_df['bb_mid'] = bb.bollinger_mavg()
            
            # ATR
            if 'ATR' in self.technical_indicators and 'high' in df.columns and 'low' in df.columns:
                feature_df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            
            # Moving averages
            if 'SMA' in self.technical_indicators:
                for period in self.lookback_periods:
                    feature_df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            
            if 'EMA' in self.technical_indicators:
                for period in self.lookback_periods:
                    feature_df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Currency strength derivatives
        for currency in self.currencies:
            strength_col = f'{currency}_strength'
            if strength_col in df.columns:
                # Momentum (rate of change)
                for period in [5, 10, 20]:
                    feature_df[f'{currency}_momentum_{period}'] = df[strength_col].pct_change(period)
                
                # Moving averages of strength
                feature_df[f'{currency}_strength_sma_10'] = df[strength_col].rolling(10).mean()
        
        # Time-based features
        feature_df['hour'] = feature_df.index.hour
        feature_df['day_of_week'] = feature_df.index.dayofweek
        feature_df['is_london_session'] = ((feature_df['hour'] >= 8) & (feature_df['hour'] <= 16)).astype(int)
        feature_df['is_ny_session'] = ((feature_df['hour'] >= 13) & (feature_df['hour'] <= 21)).astype(int)
        
        logger.info(f"Feature engineering complete. Features: {len(feature_df.columns)}")
        return feature_df
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.Series:
        """Create target variable based on future price movement"""
        logger.info("Creating target variable...")
        
        if 'close' not in df.columns:
            raise ValueError("Close price required for target variable creation")
        
        # Simple future return prediction
        future_periods = 10  # Predict 10 periods ahead
        future_returns = df['close'].shift(-future_periods) / df['close'] - 1
        
        # Classify into buy (1), hold (0), sell (-1)
        target = pd.Series(0, index=df.index)  # Default: hold
        target[future_returns > 0.001] = 1      # Buy if return > 0.1%
        target[future_returns < -0.001] = -1    # Sell if return < -0.1%
        
        logger.info(f"Target distribution: {target.value_counts().to_dict()}")
        return target
    
    def prepare_training_data(self, feature_df: pd.DataFrame, target: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for ML training"""
        logger.info("Preparing training data...")
        
        # Remove rows with NaN values
        combined = pd.concat([feature_df, target.rename('target')], axis=1)
        combined = combined.dropna()
        
        # Split features and target
        X = combined.drop('target', axis=1)
        y = combined['target']
        
        # Select numeric columns only
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_columns]
        
        logger.info(f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        return X.values, y.values
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the ML model using walk-forward analysis"""
        logger.info("Training ML model with walk-forward analysis...")
        
        # Use TimeSeriesSplit for proper time series cross-validation
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Random Forest with hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 5]
        }
        
        rf = RandomForestClassifier(random_state=42)
        
        # Grid search with time series cross-validation
        grid_search = GridSearchCV(
            rf, param_grid, cv=tscv, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        
        logger.info("Starting grid search...")
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        # Feature importance
        feature_importance = self.model.feature_importances_
        logger.info(f"Top 5 most important features:")
        for i in np.argsort(feature_importance)[-5:][::-1]:
            logger.info(f"  Feature {i}: {feature_importance[i]:.3f}")
    
    def save_model(self, model_name: str = None) -> str:
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model_name or f"acs_ml_model_{timestamp}"
        
        model_path = self.models_dir / f"{model_name}.joblib"
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'timestamp': timestamp,
            'config': self.config
        }, model_path)
        
        logger.info(f"Model saved to: {model_path}")
        return str(model_path)
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from indicator signals and price data
        """
        logger.info("Engineering features...")
        
        # Price-based features
        if 'close' in df.columns:
            # Simple technical indicators
            for period in self.config['ml']['lookback_periods']:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                df[f'rsi_{period}'] = self.calculate_rsi(df['close'], period)
                df[f'price_change_{period}'] = df['close'].pct_change(period)
        
        # Currency strength features
        currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']
        for currency in currencies:
            strength_col = f'{currency}_strength'
            if strength_col in df.columns:
                # Strength momentum
                df[f'{currency}_strength_momentum'] = df[strength_col].diff()
                
                # Strength relative to others
                other_strengths = [f'{c}_strength' for c in currencies if c != currency]
                if all(col in df.columns for col in other_strengths):
                    df[f'{currency}_relative_strength'] = df[strength_col] - df[other_strengths].mean(axis=1)
        
        # Market activity features
        if 'market_activity' in df.columns:
            # Market activity as categorical
            df['market_active'] = (df['market_activity'] > 2).astype(int)
        
        # Signal-based features
        signal_cols = ['dual_momentum_signal', 'outer_range_warning', 'hook_alert', 'xstop_signal']
        for col in signal_cols:
            if col in df.columns:
                # Signal persistence
                df[f'{col}_persist'] = df[col].rolling(window=3).sum()
        
        # Drop rows with NaN values from feature engineering
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f"Feature engineering complete. Dropped {initial_rows - len(df)} rows with NaN")
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def create_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target labels for ML training based on future price movements
        """
        logger.info("Creating trading signal labels...")
        
        # Define forward-looking profit/loss periods
        forecast_periods = [5, 10, 20]  # bars ahead
        profit_threshold = 0.001  # 0.1% profit threshold
        
        for period in forecast_periods:
            if 'close' in df.columns:
                future_return = df['close'].shift(-period) / df['close'] - 1
                
                # Create classification labels
                df[f'signal_{period}'] = np.where(
                    future_return > profit_threshold, 1,  # Buy signal
                    np.where(future_return < -profit_threshold, -1, 0)  # Sell signal, Hold
                )
        
        return df
    
    def prepare_features_and_targets(self, df: pd.DataFrame, target_col: str = 'signal_10'):
        """
        Prepare feature matrix and target vector
        """
        # Feature columns (exclude target and metadata columns)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume'] + [col for col in df.columns if col.startswith('signal_')]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove any remaining non-numeric columns
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in feature_cols if col in numeric_cols]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        logger.info(f"Features prepared: {X.shape[1]} features, {len(y)} samples")
        logger.info(f"Target distribution: {np.bincount(y + 1)}")  # Adjust for -1, 0, 1 labels
        
        return X, y, feature_cols
    
    def train_model(self, X, y, model_type: str = "RandomForest"):
        """
        Train the specified model type
        """
        logger.info(f"Training {model_type} model...")
        
        # Use time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        if model_type == "RandomForest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        else:
            logger.error(f"Model type {model_type} not implemented yet")
            return None
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Evaluate with time series cross-validation
        scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            self.model.fit(X_train_fold, y_train_fold)
            y_pred = self.model.predict(X_val_fold)
            scores.append(accuracy_score(y_val_fold, y_pred))
        
        avg_score = np.mean(scores)
        logger.info(f"Cross-validation accuracy: {avg_score:.4f} (+/- {np.std(scores) * 2:.4f})")
        
        return self.model
    
    def save_model(self, model_name: str, feature_cols: list):
        """
        Save trained model and metadata
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.joblib"
        
        # Save model
        model_path = self.models_path / model_filename
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': feature_cols,
            'timestamp': timestamp,
            'config': self.config
        }, model_path)
        
        logger.info(f"Model saved: {model_path}")
        
        # Save latest model reference
        latest_path = self.models_path / f"{model_name}_latest.joblib"
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': feature_cols,
            'timestamp': timestamp,
            'config': self.config
        }, latest_path)
        
        return model_path
    
    def full_training_pipeline(self, data_file: str, model_type: str = "RandomForest"):
        """
        Complete training pipeline
        """
        logger.info("Starting full training pipeline...")
        
        # Load data
        df = self.load_training_data(data_file)
        if df.empty:
            logger.error("No training data loaded")
            return None
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Create trading signals
        df = self.create_trading_signals(df)
        
        # Prepare features and targets
        X, y, feature_cols = self.prepare_features_and_targets(df)
        
        # Filter out samples with no signal (class 0) for initial training
        # Keep only buy (1) and sell (-1) signals
        signal_mask = (y != 0)
        X_filtered = X[signal_mask]
        y_filtered = y[signal_mask]
        
        logger.info(f"Filtered to {len(y_filtered)} samples with clear signals")
        
        if len(y_filtered) < 100:
            logger.warning("Very few samples with clear signals. Consider adjusting profit thresholds.")
        
        # Train model
        model = self.train_model(X_filtered, y_filtered, model_type)
        
        if model is None:
            logger.error("Model training failed")
            return None
        
        # Save model
        model_path = self.save_model(model_type, feature_cols)
        
        logger.info("Training pipeline completed successfully")
        return model_path

def main():
    """
    Main training script
    """
    trainer = ACSMLTrainer()
    
    # Check for training data
    processed_files = list(trainer.processed_data_path.glob("*.csv"))
    
    if not processed_files:
        logger.error("No processed data files found")
        logger.info("Please run data_collector.py first to process indicator signals")
        return
    
    # Use the first available processed file
    data_file = processed_files[0].name
    logger.info(f"Using training data: {data_file}")
    
    # Train model
    model_path = trainer.full_training_pipeline(data_file)
    
    if model_path:
        logger.info(f"Model training completed successfully: {model_path}")
    else:
        logger.error("Model training failed")

if __name__ == "__main__":
    main()