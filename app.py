"""
Main Application for Hospital Drug Demand Forecasting
Fixed modular version
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Import local modules
from data import generate_hospital_data, preprocess_data, get_data_statistics
from model import DemandForecaster
from utils import (
    UncertaintyAnalyzer, 
    calculate_inventory_metrics, 
    generate_recommendations,
    prepare_forecast_dataframe
)

def setup_ui():
    """Setup Streamlit page configuration and UI."""
    st.set_page_config(
        page_title="Hospital Drug Demand Forecasting",
        layout="wide",
        page_icon="🏥"
    )
    
    # Display header
    st.markdown("""
    <div style='padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='text-align: center; color: #1e3a8a; margin-bottom: 10px;'>
            🏥 Hospital Drug Demand Forecasting System
        </h1>
        <p style='text-align: center; color: #4b5563; font-size: 16px;'>
            Advanced time-series forecasting with uncertainty quantification for healthcare supply chain optimization
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_controls():
    """Create sidebar controls for user inputs."""
    st.sidebar.header("⚙️ Forecast Configuration")
    
    # Forecast parameters
    forecast_days = st.sidebar.slider(
        "Forecast Horizon (Days)",
        min_value=7,
        max_value=90,
        value=30,
        help="Number of days to forecast into the future"
    )
    
    current_stock = st.sidebar.number_input(
        "Current Inventory Level",
        min_value=0,
        max_value=10000,
        value=1500,
        step=100,
        help="Current stock of the medication"
    )
    
    # Model parameters
    st.sidebar.header("🔧 Model Parameters")
    
    lookback_window = st.sidebar.selectbox(
        "Lookback Window",
        options=[7, 14, 21, 30, 60],
        index=3,
        help="Number of historical days used for prediction"
    )
    
    confidence_level = st.sidebar.slider(
        "Confidence Level",
        min_value=0.80,
        max_value=0.99,
        value=0.95,
        step=0.01,
        help="Statistical confidence level for uncertainty bounds"
    )
    
    return {
        'forecast_days': forecast_days,
        'current_stock': current_stock,
        'lookback_window': lookback_window,
        'confidence_level': confidence_level
    }

def display_historical_data(dataframe):
    """Display historical data with visualizations."""
    st.subheader("📊 Historical Demand Analysis")
    
    # Display statistics
    stats = get_data_statistics(dataframe)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{stats['total_records']:,}")
    with col2:
        st.metric("Average Demand", f"{stats['mean_demand']:.1f}")
    with col3:
        st.metric("Demand Variability", f"{stats['cv_demand']:.2f}")
    with col4:
        st.metric("Date Range", f"{stats['date_range'][0].date()} to {stats['date_range'][1].date()}")
    
    # Plot historical data
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dataframe['date'],
        y=dataframe['daily_demand'],
        mode='lines',
        name='Daily Demand',
        line=dict(color='blue', width=2)
    ))
    
    # Add rolling average
    rolling_avg = dataframe['daily_demand'].rolling(window=30).mean()
    fig.add_trace(go.Scatter(
        x=dataframe['date'],
        y=rolling_avg,
        mode='lines',
        name='30-Day Moving Average',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Historical Daily Demand',
        xaxis_title='Date',
        yaxis_title='Daily Demand',
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data quality information
    with st.expander("📋 Data Quality Information"):
        st.write("""
        **Data Characteristics:**
        - Synthetic hospital drug demand data
        - Includes seasonal patterns and trends
        - Contains realistic noise and missing values
        - Representative of real-world hospital operations
        
        **Data Processing:**
        1. Missing values filled using forward/backward fill
        2. Data sorted chronologically
        3. Outliers handled through statistical methods
        4. Time series continuity ensured
        """)

def run_forecasting_pipeline(dataframe, parameters):
    """
    Run the complete forecasting pipeline.
    
    Args:
        dataframe (pd.DataFrame): Historical data
        parameters (dict): User parameters
        
    Returns:
        dict: Forecast results and metrics
    """
    # Extract demand series
    demand_series = dataframe['daily_demand']
    
    # Initialize forecaster
    forecaster = DemandForecaster(
        lookback_window=parameters['lookback_window'],
        random_state=42
    )
    
    # Prepare data
    X, y = forecaster.prepare_data(demand_series)
    
    # Build and train model
    with st.spinner("🔬 Training forecasting model..."):
        forecaster.build_model(lstm_units=50, dropout_rate=0.2)
        training_history = forecaster.train(
            X, y, 
            epochs=50, 
            batch_size=32,
            validation_split=0.2
        )
    
    # Generate forecast
    with st.spinner("🔮 Generating forecasts..."):
        last_window = demand_series.values[-parameters['lookback_window']:]
        point_forecast = forecaster.forecast(
            last_window=last_window,
            horizon=parameters['forecast_days']
        )
    
    # Calculate uncertainty
    analyzer = UncertaintyAnalyzer(confidence_level=parameters['confidence_level'])
    lower_bound, upper_bound = analyzer.calculate_confidence_intervals(point_forecast)
    
    # Calculate inventory metrics
    inventory_metrics = calculate_inventory_metrics(
        current_stock=parameters['current_stock'],
        forecast_demand=point_forecast
    )
    
    # Generate recommendations
    recommendations = generate_recommendations(inventory_metrics)
    
    # Prepare forecast dates
    last_date = dataframe['date'].iloc[-1]
    start_date = last_date + timedelta(days=1)
    
    # Create forecast dataframe
    forecast_df = prepare_forecast_dataframe(
        point_forecast=point_forecast,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        start_date=start_date
    )
    
    return {
        'point_forecast': point_forecast,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'forecast_df': forecast_df,
        'inventory_metrics': inventory_metrics,
        'recommendations': recommendations,
        'training_history': training_history,
        'last_date': last_date
    }

def display_forecast_results(forecast_results, historical_data, parameters):
    """Display forecast results with visualizations."""
    st.subheader("🔮 Demand Forecast with Uncertainty")
    
    # Create combined plot
    fig = go.Figure()
    
    # Add historical data (last 90 days)
    last_90_days = historical_data.iloc[-90:]
    fig.add_trace(go.Scatter(
        x=last_90_days['date'],
        y=last_90_days['daily_demand'],
        mode='lines',
        name='Historical Demand',
        line=dict(color='blue', width=2)
    ))
    
    # Add forecast
    forecast_df = forecast_results['forecast_df']
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['point_forecast'],
        mode='lines',
        name='Point Forecast',
        line=dict(color='green', width=3)
    ))
    
    # Add uncertainty band
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df['date'], forecast_df['date'][::-1]]),
        y=pd.concat([forecast_df['upper_bound'], forecast_df['lower_bound'][::-1]]),
        fill='toself',
        fillcolor='rgba(0, 100, 80, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name=f'{parameters["confidence_level"]*100:.0f}% Confidence Interval',
        showlegend=True
    ))
    
    fig.update_layout(
        title='Demand Forecast with Uncertainty Bounds',
        xaxis_title='Date',
        yaxis_title='Daily Demand',
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display forecast metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Forecast Demand",
            f"{forecast_results['inventory_metrics']['total_forecast_demand']:.0f}"
        )
    
    with col2:
        st.metric(
            "Average Daily Forecast",
            f"{forecast_results['inventory_metrics']['average_daily_demand']:.1f}"
        )
    
    with col3:
        uncertainty_range = np.mean(forecast_df['upper_bound'] - forecast_df['lower_bound'])
        st.metric(
            "Average Uncertainty Range",
            f"±{uncertainty_range:.1f}"
        )
    
    with col4:
        avg_cv = forecast_df['cv'].mean()
        st.metric(
            "Average CV",
            f"{avg_cv:.3f}"
        )

def display_decision_support(current_stock, forecast_results):
    """Display decision support recommendations."""
    st.subheader("📋 Inventory Decision Support")
    
    metrics = forecast_results['inventory_metrics']
    recommendations = forecast_results['recommendations']
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Stock", f"{current_stock:,}")
    
    with col2:
        st.metric(
            "Days of Supply",
            f"{metrics['days_of_supply']:.1f}"
        )
    
    with col3:
        st.metric(
            "Shortage Risk",
            f"{metrics['shortage_risk']:.0f}"
        )
    
    with col4:
        st.metric(
            "Coverage Ratio",
            f"{metrics['coverage_ratio']:.2f}"
        )
    
    # Display recommendations
    st.subheader("🚨 Recommendations")
    
    if recommendations['severity'] == "CRITICAL":
        st.error(recommendations['recommendation'])
    elif recommendations['severity'] == "HIGH":
        st.warning(recommendations['recommendation'])
    elif recommendations['severity'] == "MODERATE":
        st.info(recommendations['recommendation'])
    else:
        st.success(recommendations['recommendation'])
    
    # Detailed analysis
    with st.expander("📈 Detailed Risk Analysis"):
        st.write("**Risk Factors Considered:**")
        
        risk_factors = pd.DataFrame({
            'Factor': [
                'Current Stock Level',
                'Forecasted Demand',
                'Demand Variability',
                'Lead Time Considerations',
                'Seasonal Patterns',
                'Historical Stockout Frequency'
            ],
            'Impact': ['High', 'High', 'Medium', 'Medium', 'Low', 'Low'],
            'Current Status': [
                f'{current_stock:,} units',
                f'{metrics["total_forecast_demand"]:.0f} units',
                f'{forecast_results["forecast_df"]["cv"].mean():.3f}',
                'Standard (7 days)',
                'Accounted for in model',
                'Simulated data'
            ]
        })
        
        st.dataframe(risk_factors, use_container_width=True)

def display_export_options(forecast_results):
    """Display options for exporting results."""
    st.subheader("💾 Export Options")
    
    if st.button("📥 Download Forecast Data (CSV)"):
        forecast_df = forecast_results['forecast_df']
        csv_data = forecast_df.to_csv(index=False)
        
        st.download_button(
            label="Click to Download CSV",
            data=csv_data,
            file_name=f"hospital_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    if st.button("📊 Download Full Report"):
        # Create comprehensive report
        report_data = {
            'Parameter': [
                'Report Date',
                'Forecast Horizon',
                'Confidence Level',
                'Total Forecast Demand',
                'Average Daily Forecast',
                'Current Stock',
                'Days of Supply',
                'Shortage Risk',
                'Recommendation Severity'
            ],
            'Value': [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                f"{len(forecast_results['forecast_df'])} days",
                f"{forecast_results['recommendations'].get('confidence_level', 0.95)*100:.0f}%",
                f"{forecast_results['inventory_metrics']['total_forecast_demand']:.0f}",
                f"{forecast_results['inventory_metrics']['average_daily_demand']:.1f}",
                f"{forecast_results['inventory_metrics'].get('current_stock', 'N/A'):,}",
                f"{forecast_results['inventory_metrics']['days_of_supply']:.1f}",
                f"{forecast_results['inventory_metrics']['shortage_risk']:.0f}",
                forecast_results['recommendations']['severity']
            ]
        }
        
        report_df = pd.DataFrame(report_data)
        csv_report = report_df.to_csv(index=False)
        
        st.download_button(
            label="Download Summary Report",
            data=csv_report,
            file_name=f"forecast_summary_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

def main():
    """Main application function."""
    try:
        # Setup UI
        setup_ui()
        
        # Get user parameters
        parameters = create_controls()
        
        # Generate and display historical data
        with st.spinner("📊 Loading historical data..."):
            raw_data = generate_hospital_data(total_days=730)
            processed_data = preprocess_data(raw_data)
        
        display_historical_data(processed_data)
        
        # Run forecasting pipeline
        forecast_results = run_forecasting_pipeline(processed_data, parameters)
        
        # Display results
        display_forecast_results(forecast_results, processed_data, parameters)
        
        # Display decision support
        display_decision_support(parameters['current_stock'], forecast_results)
        
        # Display export options
        display_export_options(forecast_results)
        
        # Technical details
        with st.expander("🔬 Technical Implementation Details"):
            st.write("""
            **Model Architecture:**
            - Type: Long Short-Term Memory (LSTM) Neural Network
            - Layers: 2 LSTM layers with dropout regularization
            - Training: 50 epochs with Adam optimizer
            - Validation: 20% split with early stopping
            
            **Uncertainty Quantification:**
            - Method: Parametric confidence intervals
            - Assumption: Normally distributed forecast errors
            - Confidence Level: User-configurable (80-99%)
            
            **Data Processing:**
            - Missing Values: Forward/backward fill
            - Scaling: Min-Max normalization
            - Seasonality: Accounted for in synthetic data generation
            
            **Performance Metrics:**
            - Forecast Horizon: 7-90 days
            - Model Training Time: < 30 seconds
            - Memory Usage: Optimized for healthcare environments
            """)
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: gray; font-size: 12px;'>
                <p>🏥 Hospital Drug Demand Forecasting System v1.0</p>
                <p>For demonstration and educational purposes only | ⚠️ Not for clinical decision-making</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please try adjusting the parameters or refresh the page.")
        
        # Show technical details for debugging
        with st.expander("Technical Error Details"):
            import traceback
            st.code(traceback.format_exc())

# Run the application
if __name__ == "__main__":
    main()