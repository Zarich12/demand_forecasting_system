"""
Forecasting Model Module - Fixed version
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

class DemandForecaster:
    """
    Simplified LSTM forecaster for hospital drug demand.
    """
    
    def __init__(self, lookback_window=30, random_state=42):
        """
        Initialize the forecaster.
        
        Args:
            lookback_window (int): Number of historical days to use for prediction
            random_state (int): Random seed for reproducibility
        """
        self.lookback_window = lookback_window
        self.random_state = random_state
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.is_trained = False
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
    
    def prepare_data(self, data_series):
        """
        Prepare time series data for LSTM training.
        
        Args:
            data_series (pd.Series or np.array): Time series data
            
        Returns:
            tuple: (X, y) arrays for training
        """
        # Convert to numpy array if needed
        if hasattr(data_series, 'values'):
            data_array = data_series.values
        else:
            data_array = data_series
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data_array.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.lookback_window):
            X.append(scaled_data[i:i + self.lookback_window])
            y.append(scaled_data[i + self.lookback_window])
        
        X = np.array(X).reshape(-1, self.lookback_window, 1)
        y = np.array(y)
        
        return X, y
    
    def build_model(self, lstm_units=50, dropout_rate=0.2):
        """
        Build LSTM neural network model.
        
        Args:
            lstm_units (int): Number of LSTM units
            dropout_rate (float): Dropout rate for regularization
            
        Returns:
            tf.keras.Model: Compiled model
        """
        model = Sequential([
            LSTM(lstm_units, activation='relu', 
                 input_shape=(self.lookback_window, 1),
                 return_sequences=True),
            Dropout(dropout_rate),
            LSTM(lstm_units // 2, activation='relu'),
            Dropout(dropout_rate),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the LSTM model.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            validation_split (float): Validation split ratio
            
        Returns:
            dict: Training history
        """
        # Simple training without callbacks for compatibility
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
            shuffle=False
        )
        
        self.is_trained = True
        
        return {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'mae': history.history['mae'],
            'val_mae': history.history['val_mae']
        }
    
    def forecast(self, last_window, horizon):
        """
        Generate multi-step forecast.
        
        Args:
            last_window (np.array): Last available window of data
            horizon (int): Number of steps to forecast
            
        Returns:
            np.array: Forecasted values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before forecasting")
        
        if len(last_window) != self.lookback_window:
            raise ValueError(f"Last window must be of length {self.lookback_window}")
        
        predictions = []
        current_window = last_window.copy()
        
        for _ in range(horizon):
            # Scale current window
            scaled_window = self.scaler.transform(current_window.reshape(-1, 1)).flatten()
            
            # Reshape for model
            model_input = scaled_window.reshape(1, self.lookback_window, 1)
            
            # Predict
            scaled_prediction = self.model.predict(model_input, verbose=0)[0, 0]
            
            # Inverse transform
            prediction = self.scaler.inverse_transform([[scaled_prediction]])[0, 0]
            predictions.append(prediction)
            
            # Update window for next prediction
            current_window = np.roll(current_window, -1)
            current_window[-1] = prediction
        
        return np.array(predictions)
    
    def save(self, filepath):
        """Save model to file."""
        if self.is_trained:
            self.model.save(filepath)
    
    def load(self, filepath):
        """Load model from file."""
        self.model = tf.keras.models.load_model(filepath)
        self.is_trained = True