"""
Utility Functions for Uncertainty Quantification
Fixed version
"""
import numpy as np
import pandas as pd
from scipy import stats

class UncertaintyAnalyzer:
    """
    Handles uncertainty quantification for forecasts.
    """
    
    def __init__(self, confidence_level=0.95):
        """
        Initialize analyzer.
        
        Args:
            confidence_level (float): Confidence level (0-1)
        """
        self.confidence_level = confidence_level
    
    def calculate_confidence_intervals(self, point_forecasts, error_std=None):
        """
        Calculate confidence intervals for forecasts.
        
        Args:
            point_forecasts (np.array): Point forecasts
            error_std (float, optional): Standard deviation of errors
            
        Returns:
            tuple: (lower_bounds, upper_bounds)
        """
        # If no error_std provided, estimate from point forecasts
        if error_std is None:
            # Simple estimate: 15% of mean as standard deviation
            error_std = np.mean(point_forecasts) * 0.15
        
        # Calculate z-score for given confidence level
        z_score = stats.norm.ppf(0.5 + self.confidence_level / 2)
        
        # Calculate margin of error
        margin_of_error = z_score * error_std
        
        # Calculate bounds
        lower_bounds = np.maximum(0, point_forecasts - margin_of_error)
        upper_bounds = point_forecasts + margin_of_error
        
        return lower_bounds, upper_bounds
    
    def bootstrap_uncertainty(self, historical_data, n_bootstraps=1000):
        """
        Estimate uncertainty using bootstrap method.
        
        Args:
            historical_data (np.array): Historical time series
            n_bootstraps (int): Number of bootstrap samples
            
        Returns:
            dict: Bootstrap statistics
        """
        n_samples = len(historical_data)
        bootstrap_means = []
        
        for _ in range(n_bootstraps):
            # Resample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            sample = historical_data[indices]
            bootstrap_means.append(np.mean(sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Calculate percentiles
        alpha = (1 - self.confidence_level) / 2
        lower_percentile = np.percentile(bootstrap_means, alpha * 100)
        upper_percentile = np.percentile(bootstrap_means, (1 - alpha) * 100)
        
        return {
            'mean': np.mean(bootstrap_means),
            'std': np.std(bootstrap_means),
            'lower_bound': lower_percentile,
            'upper_bound': upper_percentile,
            'confidence_level': self.confidence_level
        }

def calculate_inventory_metrics(current_stock, forecast_demand):
    """
    Calculate inventory-related metrics.
    
    Args:
        current_stock (float): Current inventory level
        forecast_demand (np.array): Forecasted demand
        
    Returns:
        dict: Inventory metrics
    """
    total_forecast = np.sum(forecast_demand)
    avg_daily_demand = np.mean(forecast_demand)
    
    if avg_daily_demand > 0:
        days_of_supply = current_stock / avg_daily_demand
    else:
        days_of_supply = float('inf')
    
    shortage_risk = max(0, total_forecast - current_stock)
    
    return {
        'total_forecast_demand': total_forecast,
        'average_daily_demand': avg_daily_demand,
        'days_of_supply': days_of_supply,
        'shortage_risk': shortage_risk,
        'coverage_ratio': current_stock / total_forecast if total_forecast > 0 else float('inf')
    }

def generate_recommendations(inventory_metrics, confidence_level=0.95):
    """
    Generate inventory management recommendations.
    
    Args:
        inventory_metrics (dict): Inventory metrics
        confidence_level (float): Confidence level for recommendations
        
    Returns:
        dict: Recommendations and severity
    """
    coverage_ratio = inventory_metrics['coverage_ratio']
    days_of_supply = inventory_metrics['days_of_supply']
    
    if coverage_ratio < 0.8 or days_of_supply < 7:
        severity = "CRITICAL"
        color = "red"
        recommendation = """
        **IMMEDIATE ACTION REQUIRED:**
        1. Place emergency order immediately
        2. Contact all available suppliers
        3. Implement usage restrictions if necessary
        4. Escalate to hospital administration
        """
    elif coverage_ratio < 1.0 or days_of_supply < 14:
        severity = "HIGH"
        color = "orange"
        recommendation = """
        **URGENT ACTION NEEDED:**
        1. Expedite next scheduled order
        2. Monitor consumption daily
        3. Prepare contingency plan
        4. Consider alternative treatments
        """
    elif coverage_ratio < 1.2:
        severity = "MODERATE"
        color = "yellow"
        recommendation = """
        **MONITOR CLOSELY:**
        1. Review reorder parameters
        2. Consider slight order increase
        3. Monitor consumption weekly
        4. Update forecast models
        """
    else:
        severity = "LOW"
        color = "green"
        recommendation = """
        **STABLE SITUATION:**
        1. Continue regular monitoring
        2. Maintain current protocols
        3. Review for excess inventory
        4. Document successful strategy
        """
    
    return {
        'severity': severity,
        'color': color,
        'recommendation': recommendation,
        'action_required': severity in ["CRITICAL", "HIGH"]
    }

def prepare_forecast_dataframe(point_forecast, lower_bound, upper_bound, start_date, freq='D'):
    """
    Prepare forecast results as a DataFrame.
    
    Args:
        point_forecast (np.array): Point forecasts
        lower_bound (np.array): Lower bounds
        upper_bound (np.array): Upper bounds
        start_date (pd.Timestamp): Start date for forecast
        freq (str): Frequency string
        
    Returns:
        pd.DataFrame: Forecast DataFrame
    """
    # Create date range
    forecast_dates = pd.date_range(
        start=start_date,
        periods=len(point_forecast),
        freq=freq
    )
    
    # Create DataFrame
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'point_forecast': point_forecast,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'uncertainty_range': upper_bound - lower_bound
    })
    
    # Calculate additional metrics
    forecast_df['cv'] = forecast_df['uncertainty_range'] / forecast_df['point_forecast']
    
    # Categorize risk
    conditions = [
        (forecast_df['cv'] <= 0.1),
        (forecast_df['cv'] <= 0.25),
        (forecast_df['cv'] <= 0.5),
        (forecast_df['cv'] > 0.5)
    ]
    choices = ['Very Low', 'Low', 'Medium', 'High']
    
    forecast_df['risk_level'] = np.select(conditions, choices, default='Unknown')
    
    return forecast_df