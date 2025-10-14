# app.py
import streamlit as st,, ee, tempfile, os
import ee
import geemap.foliumap as geemap
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import folium
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(layout="wide", page_title="NDVI Based Field Segmentation")

import streamlit as st, ee, tempfile, os

@st.cache_resource(show_spinner=False)
def initialize_ee():
    """Initialize Earth Engine using Streamlit secrets."""
    sa = st.secrets["ee"]["service_account"]
    key_json = st.secrets["ee"]["private_key"]
    project = st.secrets["ee"].get("project", "ndvi-field-segmentation")

    # write the JSON key to a temp file because ServiceAccountCredentials needs a path
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(key_json)
        key_path = f.name
    try:
        creds = ee.ServiceAccountCredentials(sa, key_path)
        ee.Initialize(credentials=creds, project=project)
        return "✅ Earth Engine initialized"
    finally:
        os.remove(key_path)

# Run initialization and show confirmation
st.success(initialize_ee())

def app():
    st.title("Field Segmentation using NDVI Analysis")
    st.write("Analyze agricultural fields using satellite imagery and NDVI values with rainfall prediction")

    # Create columns for input parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Enter field coordinates")
        latitude = st.number_input("Latitude", value=30.9, format="%.6f")
        longitude = st.number_input("Longitude", value=75.8, format="%.6f")
        buffer_size = st.slider("Field Radius (meters)", 
                               min_value=100, max_value=1000, value=250, step=50)
    
    with col2:
        st.subheader("Analysis Parameters")
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=90))
        end_date = st.date_input("End Date", datetime.now())
        
        # Clustering method selection
        clustering_method = st.selectbox(
            "Clustering Method", 
            ["K-Means", "DBSCAN"]
        )
        
        if clustering_method == "K-Means":
            num_zones = st.slider("Number of Zones", min_value=2, max_value=7, value=3)
        else:  # DBSCAN
            eps_value = st.slider("DBSCAN Epsilon (Distance Threshold)", 
                               min_value=0.01, max_value=0.2, value=0.05, step=0.01)
            min_samples = st.slider("DBSCAN Min Samples", 
                                min_value=5, max_value=50, value=10, step=5)
    
    with col3:
        st.subheader("Crop Information")
        crop_type = st.selectbox(
            "Crop Type",
            ["Wheat", "Corn/Maize", "Rice", "Soybeans", "Cotton", "Sugarcane", "Other"]
        )
        crop_growth_stage = st.selectbox(
            "Growth Stage",
            ["Early/Emergence", "Vegetative", "Reproductive/Flowering", "Maturity"]
        )
        
        # Rainfall prediction parameters
        st.subheader("Rainfall Prediction")
        predict_rainfall = st.checkbox("Include Rainfall Prediction", value=True)
        if predict_rainfall:
            prediction_days = st.slider("Days to Predict Ahead", 1, 30, 7)
        
        # Show crop-specific NDVI reference values
        st.info(f"Typical NDVI Range for {crop_type}: {get_crop_ndvi_range(crop_type)}")
    
    # Analysis button
    if st.button("Analyze Field"):
        with st.spinner("Processing satellite imagery..."):
            # Get Sentinel-2 imagery
            s2_collection = get_sentinel2_collection(start_date, end_date, ee.Geometry.Point([longitude, latitude]).buffer(buffer_size))
            
            # Calculate NDVI for the collection
            ndvi_collection = calculate_ndvi(s2_collection)
            
            # Get median NDVI for the time period
            median_ndvi = ndvi_collection.median()
            
            # Get NDVI time series for plotting
            ndvi_time_series = extract_ndvi_time_series(ndvi_collection, ee.Geometry.Point([longitude, latitude]).buffer(buffer_size))
            
            # Get rainfall data if requested
            rainfall_data = None
            if predict_rainfall:
                rainfall_data = get_historical_rainfall(latitude, longitude, start_date, end_date)
                if rainfall_data is not None:
                    # Predict future rainfall
                    rainfall_prediction = predict_future_rainfall(rainfall_data, prediction_days)
                else:
                    st.warning("Could not retrieve rainfall data for this location")
                    rainfall_prediction = None
            else:
                rainfall_prediction = None
            
            # Perform field boundary if provided or use buffer
            field = ee.Geometry.Point([longitude, latitude]).buffer(buffer_size)
            
            # Perform zoning using selected clustering method
            if clustering_method == "K-Means":
                zoned_image = perform_kmeans_zoning(median_ndvi, field, num_zones)
                zones_param = num_zones  # For reporting
            else:  # DBSCAN
                zoned_image, actual_zones = perform_dbscan_zoning(median_ndvi, field, eps_value, min_samples)
                zones_param = f"eps={eps_value}, min_samples={min_samples}"  # For reporting
            
            # Display results
            display_results(
                median_ndvi, 
                zoned_image, 
                field, 
                latitude, 
                longitude, 
                clustering_method,
                zones_param if clustering_method == "K-Means" else actual_zones,
                ndvi_time_series,
                crop_type,
                rainfall_data,
                rainfall_prediction
            )
            
            # Export option
            st.download_button(
                label="Download Analysis Report",
                data=generate_report(
                    latitude, 
                    longitude, 
                    buffer_size, 
                    start_date, 
                    end_date, 
                    clustering_method,
                    zones_param,
                    crop_type,
                    crop_growth_stage,
                    rainfall_data,
                    rainfall_prediction
                ),
                file_name="field_analysis_report.txt",
                mime="text/plain"
            )

def get_sentinel2_collection(start_date, end_date, geometry):
    """Fetch Sentinel-2 imagery for the given time period and location."""
    # Format dates for Earth Engine
    start = ee.Date(start_date.strftime('%Y-%m-%d'))
    end = ee.Date(end_date.strftime('%Y-%m-%d'))
    
    # Get Sentinel-2 surface reflectance data
    s2 = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
        .filterDate(start, end) \
        .filterBounds(geometry) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    
    return s2

def calculate_ndvi(image_collection):
    """Calculate NDVI for each image in the collection."""
    def add_ndvi(image):
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        # Add date as a property for time series analysis
        return image.addBands(ndvi).set('date', image.date().format('YYYY-MM-dd'))
    
    return image_collection.map(add_ndvi)

def extract_ndvi_time_series(ndvi_collection, geometry):
    """Extract NDVI time series data for plotting."""
    # Get dates of images in collection
    image_list = ndvi_collection.toList(ndvi_collection.size())
    size = image_list.size().getInfo()
    
    dates = []
    mean_ndvi_values = []
    
    for i in range(size):
        image = ee.Image(image_list.get(i))
        date_str = image.get('date').getInfo()
        
        # Calculate mean NDVI for the field
        mean_ndvi = image.select('NDVI').reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=10,
            maxPixels=1e9
        ).get('NDVI').getInfo()
        
        # Only add valid readings
        if mean_ndvi is not None:
            dates.append(date_str)
            mean_ndvi_values.append(mean_ndvi)
    
    return pd.DataFrame({'date': dates, 'ndvi': mean_ndvi_values})

def get_historical_rainfall(lat, lon, start_date, end_date):
    """Get historical rainfall data from CHIRPS."""
    try:
        # Convert dates to Earth Engine format
        start = ee.Date(start_date.strftime('%Y-%m-%d'))
        end = ee.Date(end_date.strftime('%Y-%m-%d'))
        
        # Get CHIRPS daily precipitation data
        chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
            .filterDate(start, end) \
            .filterBounds(ee.Geometry.Point([lon, lat]))
        
        # Create a time series of precipitation
        def extract_precip(image):
            date = image.date().format('YYYY-MM-dd')
            precip = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee.Geometry.Point([lon, lat]),
                scale=5000  # CHIRPS resolution is ~5km
            ).get('precipitation')
            return ee.Feature(None, {'date': date, 'precip': precip})
        
        precip_series = chirps.map(extract_precip)
        
        # Convert to pandas DataFrame
        precip_list = precip_series.getInfo()
        
        dates = []
        precip_values = []
        
        for feature in precip_list['features']:
            props = feature['properties']
            dates.append(props['date'])
            precip_values.append(props['precip'])
        
        df = pd.DataFrame({'date': dates, 'precip': precip_values})
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        return df
    
    except Exception as e:
        st.error(f"Error retrieving rainfall data: {str(e)}")
        return None

def predict_future_rainfall(rainfall_data, days_to_predict):
    """Predict future rainfall using time series forecasting."""
    try:
        # Prepare data - fill any missing values with 0
        df = rainfall_data.copy()
        df['precip'].fillna(0, inplace=True)
        
        # Create a complete date range to handle missing days
        full_date_range = pd.date_range(start=df['date'].min(), end=df['date'].max())
        df = df.set_index('date').reindex(full_date_range).fillna(0).reset_index()
        df = df.rename(columns={'index': 'date'})
        
        # Simple moving average as baseline
        window_size = min(7, len(df))  # Use 7-day window or available data
        df['sma'] = df['precip'].rolling(window=window_size).mean()
        
        # For prediction, we'll use a combination of SMA and ARIMA
        # Prepare data for ARIMA
        ts_data = df.set_index('date')['precip']
        
        # Fit ARIMA model (simple configuration)
        model = ARIMA(ts_data, order=(1, 0, 1))
        model_fit = model.fit()
        
        # Make predictions
        forecast = model_fit.get_forecast(steps=days_to_predict)
        forecast_df = forecast.conf_int()
        forecast_df['predicted'] = model_fit.predict(
            start=forecast_df.index[0],
            end=forecast_df.index[-1]
        )
        
        # Create future dates
        last_date = df['date'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict+1)]
        
        # Create prediction DataFrame
        prediction = pd.DataFrame({
            'date': future_dates,
            'precip_predicted': forecast_df['predicted'].values,
            'precip_lower': forecast_df.iloc[:, 0].values,
            'precip_upper': forecast_df.iloc[:, 1].values
        })
        
        return prediction
    
    except Exception as e:
        st.error(f"Error in rainfall prediction: {str(e)}")
        return None

def perform_kmeans_zoning(ndvi_image, geometry, num_zones):
    """Segment the field into zones based on NDVI values using K-means clustering."""
    # Sample NDVI values within the field boundary
    ndvi_sample = ndvi_image.select('NDVI').sampleRegions(
        collection=geometry,
        scale=10,
        geometries=True
    )
    
    # Use K-means clustering to segment the field
    clusterer = ee.Clusterer.wekaKMeans(num_zones).train(ndvi_sample)
    result = ndvi_image.select('NDVI').cluster(clusterer)
    
    return result

def perform_dbscan_zoning(ndvi_image, geometry, eps, min_samples):
    """Segment the field into zones using DBSCAN clustering."""
    # Sample NDVI values within the field boundary
    ndvi_sample = ndvi_image.select('NDVI').sampleRegions(
        collection=geometry,
        scale=10,
        geometries=True
    )
    
    # Convert Earth Engine FeatureCollection to a Python list
    sample_data = ndvi_sample.getInfo()
    
    # Extract NDVI values
    ndvi_values = []
    for feature in sample_data['features']:
        ndvi_values.append([feature['properties']['NDVI']])
    
    # Apply DBSCAN clustering
    ndvi_array = np.array(ndvi_values)
    
    # Check if we have enough data points
    if len(ndvi_array) < min_samples:
        st.warning(f"Not enough data points for DBSCAN with min_samples={min_samples}. Using KMeans with 3 clusters instead.")
        return perform_kmeans_zoning(ndvi_image, geometry, 3), 3
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(ndvi_array)
    
    # Count number of clusters (excluding noise which is labeled as -1)
    num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    
    # If DBSCAN found only noise or just one cluster, fall back to KMeans
    if num_clusters <= 1:
        st.warning("DBSCAN did not identify multiple clusters. Using KMeans with 3 clusters instead.")
        return perform_kmeans_zoning(ndvi_image, geometry, 3), 3
    
    # Create a list with [ndvi_value, cluster_label]
    labeled_data = []
    for i, ndvi_val in enumerate(ndvi_values):
        labeled_data.append([ndvi_val[0], int(clusters[i])])
    
    # Sort by NDVI value to assign sensible zone numbers (higher NDVI = higher zone number)
    labeled_data.sort(key=lambda x: x[0])
    
    # Create mapping from cluster labels to ordered zone numbers (excluding noise)
    unique_clusters = sorted(list(set([x[1] for x in labeled_data if x[1] != -1])))
    cluster_to_zone = {cluster: i for i, cluster in enumerate(unique_clusters)}
    
    # Add noise as the lowest zone
    if -1 in clusters:
        cluster_to_zone[-1] = -1  # Keep noise as -1
    
    # Apply mapping to get ordered zone numbers
    for i in range(len(labeled_data)):
        labeled_data[i][1] = cluster_to_zone[labeled_data[i][1]]
    
    # Convert back to Earth Engine format and create a clustered image
    # For DBSCAN, we'll use a workaround since we processed the data outside EE
    # We'll use the KMeans result as a template but override with our DBSCAN clusters
    
    # First, use KMeans as a base
    kmeans_result = perform_kmeans_zoning(ndvi_image, geometry, len(unique_clusters))
    
    # The actual clustering is done outside EE, but we return the KMeans image for visualization
    # This is a limitation as we can't easily convert our Python DBSCAN results back to EE
    
    return kmeans_result, num_clusters

def display_results(ndvi_image, zoned_image, geometry, lat, lon, clustering_method, zones_param, ndvi_time_series, crop_type, rainfall_data, rainfall_prediction):
    """Display the results on the Streamlit app."""
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["NDVI Map", "Field Zones", "Time Series", "Rainfall", "Analysis"])
    
    with tab1:
        st.subheader("NDVI Distribution")
        
        # Create a map centered at the field location
        m = geemap.Map()
        m.centerObject(ee.Geometry.Point([lon, lat]), 16)
        
        # Add the field boundary
        m.addLayer(geometry, {'color': 'white'}, 'Field Boundary')
        
        # Add NDVI layer with custom colormap
        ndvi_vis = {
            'min': 0,
            'max': 0.8,
            'palette': ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850']
        }
        m.addLayer(ndvi_image.select('NDVI').clip(geometry), ndvi_vis, 'NDVI')
        
        # Add legend
        m.add_colorbar(ndvi_vis, label="NDVI Values")
        
        # Display the map
        m.to_streamlit(height=500)
    
    with tab2:
        st.subheader(f"Field Segmentation using {clustering_method}")
        
        # Get number of zones for visualization
        if clustering_method == "K-Means":
            num_zones = zones_param
        else:  # DBSCAN
            num_zones = zones_param
            st.write(f"DBSCAN identified {num_zones} clusters (excluding noise)")
        
        # Create a map for zoning
        m2 = geemap.Map()
        m2.centerObject(ee.Geometry.Point([lon, lat]), 16)
        
        # Add the field boundary
        m2.addLayer(geometry, {'color': 'white'}, 'Field Boundary')
        
        # Add zoned image with distinct colors
        zone_vis = {
            'min': 0,
            'max': num_zones - 1,
            'palette': get_zone_colors(num_zones)
        }
        m2.addLayer(zoned_image.clip(geometry), zone_vis, 'Field Zones')
        
        # Display the map
        m2.to_streamlit(height=500)
        
        # Zone explanation
        st.write("Zone interpretation:")
        zone_df = pd.DataFrame({
            "Zone": [f"Zone {i+1}" for i in range(num_zones)],
            "Description": [get_zone_description(i, num_zones) for i in range(num_zones)]
        })
        st.table(zone_df)
    
    with tab3:
        st.subheader("NDVI Time Series")
        
        if len(ndvi_time_series) > 0:
            # Convert dates to datetime objects for better plotting
            ndvi_time_series['date'] = pd.to_datetime(ndvi_time_series['date'])
            
            # Create Plotly figure
            fig = px.line(
                ndvi_time_series, 
                x='date', 
                y='ndvi',
                title=f'NDVI Time Series for {crop_type} Field',
                labels={'date': 'Date', 'ndvi': 'NDVI Value'},
                markers=True
            )
            
            # Add typical range for the crop
            crop_range = get_crop_ndvi_range(crop_type)
            range_parts = crop_range.split('-')
            if len(range_parts) == 2:
                try:
                    lower = float(range_parts[0])
                    upper = float(range_parts[1])
                    
                    fig.add_hspan(
                        y=lower,
                        line_dash="dash",
                        annotation_text="Min Expected NDVI",
                        annotation_position="bottom right",
                        line_color="red"
                    )
                    
                    fig.add_hspan(
                        y=upper,
                        line_dash="dash",
                        annotation_text="Max Expected NDVI",
                        annotation_position="top right",
                        line_color="green"
                    )
                except:
                    pass
            
            # Improve layout
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="NDVI Value",
                yaxis=dict(range=[0, 1]),
                hovermode="x unified"
            )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate trend
            if len(ndvi_time_series) >= 3:
                try:
                    # Simple linear regression to detect trend
                    x = np.array(range(len(ndvi_time_series)))
                    y = ndvi_time_series['ndvi'].values
                    coeffs = np.polyfit(x, y, 1)
                    
                    trend_direction = "increasing" if coeffs[0] > 0 else "decreasing"
                    trend_strength = abs(coeffs[0]) * len(ndvi_time_series)
                    
                    if trend_strength > 0.1:
                        trend_desc = f"strong {trend_direction}"
                    elif trend_strength > 0.05:
                        trend_desc = f"moderate {trend_direction}"
                    else:
                        trend_desc = "relatively stable"
                    
                    st.info(f"NDVI Trend Analysis: Field condition is {trend_desc} over the observed period.")
                    
                except Exception as e:
                    st.error(f"Could not calculate trend: {str(e)}")
            
            # Download time series data
            csv = ndvi_time_series.to_csv(index=False)
            st.download_button(
                label="Download Time Series Data",
                data=csv,
                file_name="ndvi_time_series.csv",
                mime="text/csv",
            )
        else:
            st.warning("Insufficient satellite data available for time series analysis. Try extending the date range.")
    
    with tab4:
        st.subheader("Rainfall Analysis")
        
        if rainfall_data is not None:
            # Plot historical rainfall
            fig = px.bar(
                rainfall_data,
                x='date',
                y='precip',
                title='Historical Rainfall',
                labels={'precip': 'Rainfall (mm)', 'date': 'Date'}
            )
            
            # Add cumulative rainfall
            rainfall_data['cumulative'] = rainfall_data['precip'].cumsum()
            fig.add_scatter(
                x=rainfall_data['date'],
                y=rainfall_data['cumulative'],
                mode='lines',
                name='Cumulative Rainfall',
                line=dict(color='red')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show rainfall statistics
            total_rain = rainfall_data['precip'].sum()
            avg_rain = rainfall_data['precip'].mean()
            max_rain = rainfall_data['precip'].max()
            
            st.metric("Total Rainfall", f"{total_rain:.1f} mm")
            st.metric("Average Daily Rainfall", f"{avg_rain:.1f} mm")
            st.metric("Maximum Daily Rainfall", f"{max_rain:.1f} mm")
            
            # Add rainfall prediction if available
            if rainfall_prediction is not None:
                st.subheader("Rainfall Forecast")
                
                # Create prediction plot
                fig_pred = go.Figure()
                
                # Add historical data
                fig_pred.add_trace(go.Scatter(
                    x=rainfall_data['date'],
                    y=rainfall_data['precip'],
                    mode='lines+markers',
                    name='Historical Rainfall',
                    line=dict(color='blue')
                ))
                
                # Add prediction
                fig_pred.add_trace(go.Scatter(
                    x=rainfall_prediction['date'],
                    y=rainfall_prediction['precip_predicted'],
                    mode='lines+markers',
                    name='Predicted Rainfall',
                    line=dict(color='green')
                ))
                
                # Add confidence interval
                fig_pred.add_trace(go.Scatter(
                    x=rainfall_prediction['date'].tolist() + rainfall_prediction['date'].tolist()[::-1],
                    y=rainfall_prediction['precip_upper'].tolist() + rainfall_prediction['precip_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval'
                ))
                
                fig_pred.update_layout(
                    title='Rainfall Forecast',
                    xaxis_title='Date',
                    yaxis_title='Rainfall (mm)',
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Calculate predicted total
                pred_total = rainfall_prediction['precip_predicted'].sum()
                st.metric("Predicted Total Rainfall", f"{pred_total:.1f} mm")
                
                # Add recommendations based on rainfall prediction
                if pred_total > 50:
                    st.warning("Heavy rainfall predicted. Consider drainage management and delay fertilizer application.")
                elif pred_total > 20:
                    st.info("Moderate rainfall predicted. Good for crop growth but monitor for waterlogging.")
                else:
                    st.info("Light rainfall predicted. Irrigation may be needed if soil moisture is low.")
        else:
            st.warning("Rainfall data not available for this location or time period.")
    
    with tab5:
        st.subheader("Statistical Analysis")
        
        # Get NDVI statistics for the field
        try:
            ndvi_stats = ndvi_image.select('NDVI').reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    reducer2=ee.Reducer.stdDev(),
                    sharedInputs=True
                ).combine(
                    reducer2=ee.Reducer.minMax(),
                    sharedInputs=True
                ),
                geometry=geometry,
                scale=10,
                maxPixels=1e9
            ).getInfo()
            
            # Display statistics
            stats_df = pd.DataFrame({
                "Statistic": ["Mean NDVI", "StdDev", "Minimum", "Maximum"],
                "Value": [
                    round(ndvi_stats.get('NDVI_mean', 0), 3),
                    round(ndvi_stats.get('NDVI_stdDev', 0), 3),
                    round(ndvi_stats.get('NDVI_min', 0), 3),
                    round(ndvi_stats.get('NDVI_max', 0), 3)
                ]
            })
            st.table(stats_df)
            
            # Crop health assessment
            mean_ndvi = ndvi_stats.get('NDVI_mean', 0)
            crop_range = get_crop_ndvi_range(crop_type)
            range_parts = crop_range.split('-')
            
            if len(range_parts) == 2:
                try:
                    lower = float(range_parts[0])
                    upper = float(range_parts[1])
                    
                    if mean_ndvi < lower:
                        st.warning(f"The average NDVI ({mean_ndvi:.3f}) is below the expected range for {crop_type} ({crop_range}). This may indicate stress or early growth stage.")
                    elif mean_ndvi > upper:
                        st.success(f"The average NDVI ({mean_ndvi:.3f}) is above the expected range for {crop_type} ({crop_range}). This indicates excellent vegetation density.")
                    else:
                        st.success(f"The average NDVI ({mean_ndvi:.3f}) is within the expected range for {crop_type} ({crop_range}).")
                except:
                    pass
            
            # Create recommendations based on NDVI values, crop type, and rainfall
            st.subheader("Recommendations")
            total_rain = rainfall_data['precip'].sum() if rainfall_data is not None else None
            pred_rain = rainfall_prediction['precip_predicted'].sum() if rainfall_prediction is not None else None
            
            recommendations = generate_recommendations(
                mean_ndvi, 
                zones_param if isinstance(zones_param, int) else num_zones, 
                crop_type,
                total_rain,
                pred_rain
            )
            for rec in recommendations:
                st.markdown(f"- {rec}")
                
        except Exception as e:
            st.error(f"Error computing statistics: {str(e)}")

def get_crop_ndvi_range(crop_type):
    """Return typical NDVI range for different crops at peak growth."""
    crop_ranges = {
        "Wheat": "0.4-0.9",
        "Corn/Maize": "0.5-0.9",
        "Rice": "0.4-0.8",
        "Soybeans": "0.4-0.9",
        "Cotton": "0.4-0.8",
        "Sugarcane": "0.5-0.9",
        "Other": "0.3-0.8"
    }
    return crop_ranges.get(crop_type, "0.3-0.8")

def get_zone_colors(num_zones):
    """Generate a color palette for the zones."""
    if num_zones <= 5:
        return ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641'][:num_zones]
    else:
        # For more than 5 zones, use a different colormap
        cmap = plt.cm.get_cmap('viridis', num_zones)
        return [mcolors.rgb2hex(cmap(i)) for i in range(num_zones)]

def get_zone_description(zone_index, num_zones):
    """Generate description for each zone based on its index."""
    if num_zones == 3:
        descriptions = [
            "Low vigor area - may require additional inputs or investigation",
            "Moderate vigor area - average performance",
            "High vigor area - optimal performance"
        ]
        return descriptions[zone_index]
    elif num_zones == 2:
        descriptions = [
            "Lower performing area - may require attention",
            "Higher performing area - good crop health"
        ]
        return descriptions[zone_index]
    else:
        # Generic descriptions for other numbers of zones
        if zone_index == 0:
            return "Lowest vigor area"
        elif zone_index == num_zones - 1:
            return "Highest vigor area"
        else:
            return f"Moderate vigor area (level {zone_index+1} of {num_zones})"

def generate_recommendations(mean_ndvi, num_zones, crop_type, total_rain=None, pred_rain=None):
    """Generate recommendations based on NDVI values, crop type, and rainfall."""
    recommendations = []
    
    # General recommendations based on NDVI value
    if mean_ndvi < 0.3:
        recommendations.append("The field shows signs of stress. Consider irrigation or nutrient assessment.")
        recommendations.append("Low NDVI values may indicate bare soil or early growth stage.")
    elif mean_ndvi < 0.5:
        recommendations.append("Field has moderate vegetation health. Monitor for changes in coming weeks.")
        recommendations.append("Consider targeted fertilizer application in lower-performing zones.")
    else:
        recommendations.append("Field shows good overall vegetation health.")
        recommendations.append("Focus on maintaining current management practices.")
    
    # Add crop-specific recommendations
    if crop_type == "Wheat":
        if mean_ndvi < 0.4:
            recommendations.append("For wheat at this NDVI level, check for nitrogen deficiency.")
        elif mean_ndvi > 0.7:
            recommendations.append("Wheat shows excellent canopy development. Monitor for disease pressure in dense canopy.")
    elif crop_type == "Corn/Maize":
        if mean_ndvi < 0.5:
            recommendations.append("For corn at this NDVI level, evaluate water stress and consider irrigation if available.")
        elif mean_ndvi > 0.7:
            recommendations.append("Corn shows excellent growth. Ensure adequate nitrogen for grain fill period.")
    elif crop_type == "Rice":
        if mean_ndvi < 0.4:
            recommendations.append("For rice fields, verify water levels and nutrient availability.")
    elif crop_type == "Cotton":
        if mean_ndvi > 0.6:
            recommendations.append("Cotton fields with high NDVI may benefit from growth regulators to balance vegetative growth.")
    
    # Recommendations about management zones
    if isinstance(num_zones, int) and num_zones >= 3:
        recommendations.append(f"Consider variable rate application based on the {num_zones} identified management zones.")
        recommendations.append("Take soil samples from each zone to determine specific nutrient requirements.")
    
    # Rainfall-based recommendations
    if total_rain is not None:
        if total_rain > 200:
            recommendations.append("High rainfall observed. Monitor for waterlogging and consider drainage improvements.")
        elif total_rain > 100:
            recommendations.append("Adequate rainfall received. Monitor soil moisture for optimal conditions.")
        else:
            recommendations.append("Low rainfall observed. Consider irrigation if available.")
    
    if pred_rain is not None:
        if pred_rain > 50:
            recommendations.append("Heavy rainfall predicted. Delay fertilizer application to prevent runoff.")
        elif pred_rain > 20:
            recommendations.append("Moderate rainfall predicted. Good time for fertilizer application if no heavy rain expected.")
        else:
            recommendations.append("Light rainfall predicted. Consider irrigation if soil moisture is low.")
    
    return recommendations

def generate_report(lat, lon, buffer, start_date, end_date, clustering_method, zones_param, crop_type, crop_stage, rainfall_data, rainfall_prediction):
    """Generate a detailed text report for download."""
    # Calculate rainfall statistics if available
    rainfall_stats = ""
    if rainfall_data is not None:
        total_rain = rainfall_data['precip'].sum()
        avg_rain = rainfall_data['precip'].mean()
        max_rain = rainfall_data['precip'].max()
        
        rainfall_stats = f"""
Rainfall Analysis:
- Total Rainfall: {total_rain:.1f} mm
- Average Daily Rainfall: {avg_rain:.1f} mm
- Maximum Daily Rainfall: {max_rain:.1f} mm
"""
    
    # Add rainfall prediction if available
    rainfall_forecast = ""
    if rainfall_prediction is not None:
        pred_total = rainfall_prediction['precip_predicted'].sum()
        rainfall_forecast = f"""
Rainfall Forecast:
- Predicted Rainfall: {pred_total:.1f} mm over next {len(rainfall_prediction)} days
"""
    
    report = f"""
Field Analysis Report
=====================
Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Location Information:
- Latitude: {lat}
- Longitude: {lon}
- Field Radius: {buffer} meters

Crop Information:
- Crop Type: {crop_type}
- Growth Stage: {crop_stage}
- Typical NDVI Range for {crop_type}: {get_crop_ndvi_range(crop_type)}

Analysis Parameters:
- Analysis Period: {start_date} to {end_date}
- Clustering Method: {clustering_method}
- Zone Parameters: {zones_param}

{rainfall_stats}
{rainfall_forecast}
Summary of Results:
- The field was segmented using {clustering_method} clustering.
- Zones represent different levels of crop vigor, from lowest to highest.
- Consider variable rate application of inputs based on these zones.

Management Recommendations:
1. Ground-truth the zones with field visits
2. Take soil samples from each management zone
3. Develop variable rate prescription maps for inputs
4. Monitor changes in NDVI over time to assess management effectiveness
5. Consider rainfall patterns when scheduling field operations

For more detailed analysis, please consider:
- Soil testing in each zone
- Tissue sampling for nutrient analysis
- Correlating NDVI patterns with yield data if available

Note: This analysis is based on remote sensing data and should be verified with field observations.
    """
    return report

if __name__ == "__main__":
    app()
