"""
User Interface Module for Hospital Demand Forecasting Dashboard
Handles all Streamlit UI components and layout configuration
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def setup_page_configuration():
    """
    Configures the Streamlit page settings including title, layout, and sidebar state.
    This should be called at the beginning of the main application.
    """
    st.set_page_config(
        page_title="Hospital Drug Demand Forecasting System",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def display_application_header():
    """
    Displays the main application header with title and description.
    Uses HTML/CSS for custom styling while maintaining readability.
    """
    # Main title with custom styling
    st.markdown("""
    <div style='padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='text-align: center; color: #1e3a8a; margin-bottom: 10px;'>
            Hospital Drug Demand Forecasting System
        </h1>
        <p style='text-align: center; color: #4b5563; font-size: 16px;'>
            Time-series forecasting with uncertainty quantification for healthcare supply chain planning
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar_controls():
    """
    Creates and manages all sidebar controls for user input.
    Returns a dictionary of user-selected parameters.
    
    Returns:
        dict: Dictionary containing forecast parameters and current inventory status
    """
    st.sidebar.markdown("## 🎯 Forecast Configuration")
    
    # Forecast horizon control
    forecast_days = st.sidebar.slider(
        label="Forecast Horizon (Days)",
        min_value=7,
        max_value=90,
        value=30,
        help="Number of days to forecast into the future"
    )
    
    # Current inventory control
    current_stock = st.sidebar.number_input(
        label="Current Inventory Level",
        min_value=0,
        max_value=10000,
        value=1200,
        step=100,
        help="Current stock of the drug in inventory"
    )
    
    # Model configuration
    st.sidebar.markdown("## ⚙️ Model Parameters")
    lookback_window = st.sidebar.selectbox(
        label="Lookback Window Size",
        options=[7, 14, 21, 30],
        index=1,
        help="Number of historical days used for making predictions"
    )
    
    confidence_level = st.sidebar.slider(
        label="Confidence Level (%)",
        min_value=80,
        max_value=99,
        value=95,
        help="Statistical confidence level for uncertainty bounds"
    )
    
    return {
        'forecast_days': forecast_days,
        'current_stock': current_stock,
        'lookback_window': lookback_window,
        'confidence_level': confidence_level
    }

def display_historical_data_visualization(dataframe):
    """
    Displays historical demand data with appropriate visualizations.
    
    Args:
        dataframe (pd.DataFrame): Historical data with date and demand columns
    """
    st.markdown("## 📊 Historical Demand Analysis")
    
    # Display data summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Records", value=f"{len(dataframe):,}")
    with col2:
        st.metric(label="Average Daily Demand", 
                 value=f"{dataframe['daily_demand'].mean():.1f}")
    with col3:
        st.metric(label="Peak Demand", 
                 value=f"{dataframe['daily_demand'].max():.0f}")
    with col4:
        st.metric(label="Data Completeness", 
                 value=f"{(len(dataframe) / 730 * 100):.1f}%")
    
    # Plot historical data
    st.line_chart(dataframe.set_index('date')['daily_demand'],
                  use_container_width=True)
    
    # Data quality indicators
    with st.expander("📋 Data Quality Report"):
        st.write("**Missing Data Pattern:** Simulated real-world hospital recording gaps")
        st.write("**Data Characteristics:**")
        st.write("- Seasonal patterns with weekly/quarterly cycles")
        st.write("- Upward trend reflecting population growth")
        st.write("- Measurement noise from manual recording")

def display_forecast_results(forecast_data, historical_data, parameters):
    """
    Displays forecast results with uncertainty bounds and visualizations.
    
    Args:
        forecast_data (dict): Dictionary containing forecast arrays
        historical_data (pd.DataFrame): Historical data for context
        parameters (dict): User-selected parameters
    """
    st.markdown("## 🔮 Demand Forecast with Uncertainty Quantification")
    
    # Create forecast date range
    last_date = historical_data['date'].iloc[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=parameters['forecast_days'],
        freq='D'
    )
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'point_forecast': forecast_data['point_forecast'],
        'lower_bound': forecast_data['lower_bound'],
        'upper_bound': forecast_data['upper_bound']
    })
    
    # Display forecast metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        total_forecast = forecast_data['point_forecast'].sum()
        st.metric(label="Total Forecasted Demand", 
                 value=f"{total_forecast:.0f}")
    with col2:
        avg_daily = forecast_data['point_forecast'].mean()
        st.metric(label="Average Daily Forecast", 
                 value=f"{avg_daily:.1f}")
    with col3:
        uncertainty_range = (forecast_data['upper_bound'] - forecast_data['lower_bound']).mean()
        st.metric(label="Average Uncertainty Range", 
                 value=f"±{uncertainty_range:.1f}")
    
    # Plot forecast with uncertainty bands
    
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_data['date'][-90:],  # Last 90 days
        y=historical_data['daily_demand'][-90:],
        mode='lines',
        name='Historical Demand',
        line=dict(color='blue', width=2)
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_data['point_forecast'],
        mode='lines',
        name='Point Forecast',
        line=dict(color='green', width=3)
    ))
    
    # Add uncertainty band
    fig.add_trace(go.Scatter(
        x=forecast_dates.tolist() + forecast_dates.tolist()[::-1],
        y=forecast_data['upper_bound'].tolist() + forecast_data['lower_bound'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name=f"{parameters['confidence_level']}% Confidence Interval"
    ))
    
    fig.update_layout(
        title='Demand Forecast with Uncertainty Bounds',
        xaxis_title='Date',
        yaxis_title='Daily Demand',
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_decision_support(current_stock, forecast_data, parameters):
    """
    Provides decision support recommendations based on forecast results.
    
    Args:
        current_stock (int): Current inventory level
        forecast_data (dict): Forecast results including total demand
        parameters (dict): User parameters including forecast horizon
    """
    st.markdown("## 📋 Inventory Decision Support")
    
    # Calculate key metrics
    total_forecast_demand = forecast_data['point_forecast'].sum()
    days_of_supply = current_stock / forecast_data['point_forecast'].mean() if forecast_data['point_forecast'].mean() > 0 else float('inf')
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Current Inventory", value=f"{current_stock:,}")
    
    with col2:
        st.metric(label="Forecasted Demand", 
                 value=f"{total_forecast_demand:.0f}")
    
    with col3:
        st.metric(label="Projected Days of Supply", 
                 value=f"{days_of_supply:.1f}")
    
    with col4:
        shortage_risk = max(0, total_forecast_demand - current_stock)
        st.metric(label="Potential Shortage", 
                 value=f"{shortage_risk:.0f}")
    
    # Decision logic
    st.markdown("### 🚨 Recommendation Engine")
    
    if total_forecast_demand > current_stock * 1.2:
        st.error("""
        ⚠️ **HIGH PRIORITY - IMMEDIATE ACTION REQUIRED**
        
        **Risk Assessment:** High probability of stockout within forecast period
        
        **Recommended Actions:**
        1. Place emergency reorder immediately
        2. Contact alternative suppliers
        3. Consider therapeutic alternatives
        4. Review consumption patterns for anomalies
        """)
    elif total_forecast_demand > current_stock:
        st.warning("""
        ⚠️ **MEDIUM PRIORITY - ACTION RECOMMENDED**
        
        **Risk Assessment:** Moderate risk of stockout
        
        **Recommended Actions:**
        1. Schedule regular reorder
        2. Monitor daily consumption closely
        3. Prepare contingency plan
        """)
    else:
        st.success("""
        ✅ **LOW PRIORITY - MONITOR SITUATION**
        
        **Risk Assessment:** Adequate inventory for forecast period
        
        **Recommended Actions:**
        1. Continue regular monitoring
        2. Maintain standard reorder schedule
        3. Review for potential excess inventory
        """)
    
    # Additional insights
    with st.expander("📈 Detailed Risk Analysis"):
        st.write("**Confidence Interval Analysis:**")
        st.write(f"- Upper bound total demand: {forecast_data['upper_bound'].sum():.0f}")
        st.write(f"- Lower bound total demand: {forecast_data['lower_bound'].sum():.0f}")
        st.write(f"- Range: ±{(forecast_data['upper_bound'].sum() - forecast_data['lower_bound'].sum())/2:.0f}")
        
        st.write("\n**Seasonal Considerations:**")
        st.write("- Account for weekly patterns (lower weekend consumption)")
        st.write("- Consider upcoming holidays affecting demand")
        st.write("- Review historical stockout events for similar periods")
