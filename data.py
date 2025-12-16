"""
Data Generation and Processing Module
Fixed version for compatibility
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_hospital_data(total_days=730):
    """
    Generate synthetic hospital drug demand data with realistic patterns.
    
    Returns:
        pd.DataFrame: Contains 'date' and 'daily_demand' columns
    """
    np.random.seed(42)
    
    # Create date range
    start_date = pd.Timestamp('2023-01-01')
    dates = pd.date_range(start=start_date, periods=total_days, freq='D')
    
    # Base components
    trend = np.linspace(0, 15, total_days)
    annual_seasonality = 20 * np.sin(2 * np.pi * dates.dayofyear / 365)
    weekly_seasonality = 10 * np.sin(2 * np.pi * dates.dayofweek / 7)
    monthly_seasonality = 8 * np.sin(2 * np.pi * dates.day / 30)
    
    # Random components
    random_noise = np.random.normal(0, 8, total_days)
    
    # Special events (outliers)
    special_events = np.zeros(total_days)
    event_indices = np.random.choice(total_days, size=int(total_days * 0.02), replace=False)
    special_events[event_indices] = np.random.uniform(20, 50, len(event_indices))
    
    # Combine all components
    demand = (
        40 +  # Base level
        trend +
        annual_seasonality +
        weekly_seasonality +
        monthly_seasonality +
        random_noise +
        special_events
    )
    
    # Ensure non-negative
    demand = np.maximum(0, demand)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'daily_demand': demand.round().astype(int)
    })
    
    # Add some missing values (5%)
    missing_mask = np.random.random(total_days) < 0.05
    df.loc[missing_mask, 'daily_demand'] = np.nan
    
    # Fill missing values
    df['daily_demand'] = df['daily_demand'].fillna(method='ffill').fillna(method='bfill')
    
    return df

def preprocess_data(dataframe):
    """
    Clean and prepare data for modeling.
    
    Args:
        dataframe (pd.DataFrame): Raw data with 'date' and 'daily_demand'
        
    Returns:
        pd.DataFrame: Cleaned data
    """
    df = dataframe.copy()
    
    # Ensure proper date format
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Remove any remaining NaN values
    df = df.dropna()
    
    # Sort by date
    df = df.sort_values('date')
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df

def get_data_statistics(dataframe):
    """
    Calculate descriptive statistics for the demand data.
    
    Args:
        dataframe (pd.DataFrame): Processed data
        
    Returns:
        dict: Statistics dictionary
    """
    demand_series = dataframe['daily_demand']
    
    stats = {
        'total_records': len(dataframe),
        'date_range': (dataframe['date'].min(), dataframe['date'].max()),
        'total_days': (dataframe['date'].max() - dataframe['date'].min()).days,
        'mean_demand': demand_series.mean(),
        'median_demand': demand_series.median(),
        'std_demand': demand_series.std(),
        'min_demand': demand_series.min(),
        'max_demand': demand_series.max(),
        'percentile_25': demand_series.quantile(0.25),
        'percentile_75': demand_series.quantile(0.75),
        'cv_demand': demand_series.std() / demand_series.mean() if demand_series.mean() > 0 else 0
    }
    
    return stats