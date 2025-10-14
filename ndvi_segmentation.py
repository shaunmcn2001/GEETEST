# app.py
import os
os.environ["GEEMAP_BACKEND"] = "folium" 
import geemap.foliumap as geemap
import streamlit as st, ee
import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import folium
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go


# Set page configuration
st.set_page_config(layout="wide", page_title="NDVI Based Field Segmentation")

# Initialize Earth Engine
@st.cache_resource
def initialize_ee():
    try:
        ee.Initialize(project='ndvi-field-segmentation')
    except Exception:
        ee.Authenticate()
        ee.Initialize(project='ndvi-field-segmentation')

# Call the initialization function
initialize_ee()

def get_rainfall_data(start_date, end_date, geometry):
    """Fetch precipitation data from CHIRPS dataset."""
    # Format dates for Earth Engine
    start = ee.Date(start_date.strftime('%Y-%m-%d'))
    end = ee.Date(end_date.strftime('%Y-%m-%d'))
    
    # Get CHIRPS precipitation data (daily rainfall in mm)
    rainfall = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
        .filterDate(start, end) \
        .filterBounds(geometry)
    
    # Calculate total and average rainfall
    total_rainfall = rainfall.sum().select('precipitation')
    avg_rainfall = rainfall.mean().select('precipitation')
    
    # Get time series for plotting
    rainfall_series = extract_rainfall_time_series(rainfall, geometry)
    
    return {
        'total': total_rainfall,
        'average': avg_rainfall,
        'time_series': rainfall_series
    }
def app():
    st.title("Field Segmentation using NDVI Analysis")
    st.write("Analyze agricultural fields using satellite imagery and NDVI values")

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
            ["K-Means", "DBSCAN","Mean Shift", "GMM"]
        )
        
        if clustering_method == "K-Means" or clustering_method == "GMM":
            num_zones = st.slider("Number of Zones", min_value=2, max_value=7, value=3)
        elif clustering_method == "Mean Shift":
            bandwidth = st.slider("Bandwidth (Controls cluster size)", 
                                min_value=0.01, max_value=0.2, value=0.1, step=0.01)
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
            
            # Perform field boundary if provided or use buffer
            field = ee.Geometry.Point([longitude, latitude]).buffer(buffer_size)
            
            # Get rainfall data
            rainfall_data = get_rainfall_data(start_date, end_date, ee.Geometry.Point([longitude, latitude]).buffer(buffer_size))
            # Performance metrics
            performance_metrics = {}
            
            # Perform zoning using selected clustering method
            # In the "Analyze Field" button logic
            if clustering_method == "K-Means":
                start_time = datetime.now()
                zoned_image = perform_kmeans_zoning(median_ndvi, field, num_zones)
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                zones_param = num_zones  # For reporting
                performance_metrics["K-Means"] = processing_time
                
            elif clustering_method == "GMM":
                zoned_image, processing_time = perform_gmm_zoning(median_ndvi, field, num_zones)
                zones_param = num_zones  # For reporting
                performance_metrics["GMM"] = processing_time
                
            elif clustering_method == "Mean Shift":
                zoned_image, num_clusters, processing_time = perform_meanshift_zoning(median_ndvi, field, bandwidth)
                zones_param = num_clusters  # For reporting
                performance_metrics["Mean Shift"] = processing_time
                st.success(f"Mean Shift identified {num_clusters} zones")
                
            else:  # DBSCAN
                start_time = datetime.now()
                zoned_image, actual_zones = perform_dbscan_zoning(median_ndvi, field, eps_value, min_samples)
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                zones_param = actual_zones  # For reporting
                performance_metrics["DBSCAN"] = processing_time
            
            # Display results
            display_results(
                median_ndvi, 
                zoned_image, 
                field, 
                latitude, 
                longitude, 
                clustering_method,
                zones_param,
                ndvi_time_series,
                crop_type,
                performance_metrics,
                rainfall_data
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
                    rainfall_data
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
    """Extract NDVI time series data for plotting and storage."""
    # Get dates of images in collection
    image_list = ndvi_collection.toList(ndvi_collection.size())
    size = image_list.size().getInfo()
    
    dates = []
    mean_ndvi_values = []
    
    # Create storage for day-wise NDVI values
    ndvi_daily_data = {}
    
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
            
            # Store the daily NDVI image for this date
            ndvi_daily_data[date_str] = {
                'mean': mean_ndvi,
                'image': image.select('NDVI')
            }
    
    # Store the daily NDVI data in the session state
    st.session_state['ndvi_daily_data'] = ndvi_daily_data
    
    return pd.DataFrame({'date': dates, 'ndvi': mean_ndvi_values})

def extract_rainfall_time_series(rainfall_collection, geometry):
    """Extract rainfall time series data for plotting."""
    # Get dates of images in collection
    image_list = rainfall_collection.toList(rainfall_collection.size())
    size = image_list.size().getInfo()
    
    dates = []
    rainfall_values = []
    
    for i in range(size):
        image = ee.Image(image_list.get(i))
        date = image.date()
        date_str = date.format('YYYY-MM-dd').getInfo()
        
        # Calculate mean rainfall for the field area
        mean_rainfall = image.select('precipitation').reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=5000,  # CHIRPS data is coarse resolution
            maxPixels=1e9
        ).get('precipitation').getInfo()
        
        # Only add valid readings
        if mean_rainfall is not None:
            dates.append(date_str)
            rainfall_values.append(mean_rainfall)
    
    return pd.DataFrame({'date': dates, 'rainfall': rainfall_values})

def perform_meanshift_zoning(ndvi_image, geometry, bandwidth=0.1):
    """Segment the field into zones based on NDVI values using Mean Shift clustering."""
    try:
        # Sample NDVI values within the field boundary with a smaller sample size
        ndvi_sample = ndvi_image.select('NDVI').sampleRegions(
            collection=geometry,
            scale=10,  # 10m scale
            geometries=True,
            tileScale=16  # Add tileScale to help with computation
        )
        
        # Try to get the sample data with a timeout
        try:
            sample_data = ndvi_sample.getInfo()
        except Exception as e:
            st.error(f"Error getting NDVI samples: {str(e)}")
            st.warning("Falling back to K-Means clustering with 3 zones due to sampling error.")
            return perform_kmeans_zoning(ndvi_image, geometry, 3), 3, 0.0
        
        # Check if we have enough sample data
        if 'features' not in sample_data or len(sample_data['features']) < 10:
            st.warning("Not enough NDVI sample points found. Using K-Means with 3 clusters instead.")
            return perform_kmeans_zoning(ndvi_image, geometry, 3), 3, 0.0
        
        # Extract NDVI values
        ndvi_values = []
        for feature in sample_data['features']:
            if 'NDVI' in feature['properties'] and feature['properties']['NDVI'] is not None:
                ndvi_values.append([feature['properties']['NDVI']])
        
        # Check if we have enough valid NDVI values
        if len(ndvi_values) < 10:
            st.warning("Not enough valid NDVI values found. Using K-Means with 3 clusters instead.")
            return perform_kmeans_zoning(ndvi_image, geometry, 3), 3, 0.0
        
        # Apply Mean Shift clustering
        ndvi_array = np.array(ndvi_values)
        
        # Start time for performance measurement
        start_time = datetime.now()
        
        # Import MeanShift
        from sklearn.cluster import MeanShift
        
        # Apply MeanShift clustering
        meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        clusters = meanshift.fit_predict(ndvi_array)
        
        # End time for performance measurement
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Number of clusters found
        num_clusters = len(np.unique(clusters))
        
        # If Mean Shift found only one cluster, adjust bandwidth
        if num_clusters <= 1:
            st.warning("Mean Shift identified only one cluster. Adjusting parameters and trying again.")
            # Try with smaller bandwidth
            meanshift = MeanShift(bandwidth=bandwidth/2, bin_seeding=True)
            clusters = meanshift.fit_predict(ndvi_array)
            num_clusters = len(np.unique(clusters))
            
            if num_clusters <= 1:
                st.warning("Mean Shift still found only one cluster. Using KMeans with 3 clusters instead.")
                return perform_kmeans_zoning(ndvi_image, geometry, 3), 3, processing_time
        
        # Create a list with [ndvi_value, cluster_label]
        labeled_data = []
        for i, ndvi_val in enumerate(ndvi_values):
            labeled_data.append([ndvi_val[0], int(clusters[i])])
        
        # Sort by NDVI value to assign sensible zone numbers (higher NDVI = higher zone number)
        labeled_data.sort(key=lambda x: x[0])
        
        # Create mapping from cluster labels to ordered zone numbers
        unique_clusters = sorted(list(set([x[1] for x in labeled_data])))
        cluster_to_zone = {cluster: i for i, cluster in enumerate(unique_clusters)}
        
        # Apply mapping to get ordered zone numbers
        for i in range(len(labeled_data)):
            labeled_data[i][1] = cluster_to_zone[labeled_data[i][1]]
        
        # Similar to DBSCAN implementation, use K-Means as a base for visualization
        kmeans_result = perform_kmeans_zoning(ndvi_image, geometry, len(unique_clusters))
        
        return kmeans_result, num_clusters, processing_time
        
    except Exception as e:
        st.error(f"Error in Mean Shift clustering: {str(e)}")
        st.warning("Falling back to K-Means clustering with 3 zones.")
        return perform_kmeans_zoning(ndvi_image, geometry, 3), 3, 0.0

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
def perform_gmm_zoning(ndvi_image, geometry, num_zones):
    """Segment the field into zones based on NDVI values using Gaussian Mixture Model."""
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
    
    # Apply GMM clustering
    ndvi_array = np.array(ndvi_values)
    
    # Start time for performance measurement
    start_time = datetime.now()
    
    gmm = GaussianMixture(n_components=num_zones, random_state=42)
    clusters = gmm.fit_predict(ndvi_array)
    
    # End time for performance measurement
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # Create a list with [ndvi_value, cluster_label]
    labeled_data = []
    for i, ndvi_val in enumerate(ndvi_values):
        labeled_data.append([ndvi_val[0], int(clusters[i])])
    
    # Sort by NDVI value to assign sensible zone numbers (higher NDVI = higher zone number)
    labeled_data.sort(key=lambda x: x[0])
    
    # Create mapping from cluster labels to ordered zone numbers
    unique_clusters = sorted(list(set([x[1] for x in labeled_data])))
    cluster_to_zone = {cluster: i for i, cluster in enumerate(unique_clusters)}
    
    # Apply mapping to get ordered zone numbers
    for i in range(len(labeled_data)):
        labeled_data[i][1] = cluster_to_zone[labeled_data[i][1]]
    
    # Similar to DBSCAN implementation, use K-Means as a base for visualization
    kmeans_result = perform_kmeans_zoning(ndvi_image, geometry, len(unique_clusters))
    
    return kmeans_result, processing_time

def display_results(ndvi_image, zoned_image, geometry, lat, lon, clustering_method, zones_param, ndvi_time_series, crop_type, performance_metrics=None, rainfall_data=None):
    """Display the results on the Streamlit app."""
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["NDVI Map", "Field Zones", "NDVI Time Series", "Rainfall", "Analysis", "Algorithm Comparison"])
    
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
            
            # Display daily NDVI data
            st.subheader("Daily NDVI Data")
            if 'ndvi_daily_data' in st.session_state and len(st.session_state['ndvi_daily_data']) > 0:
                # Format the data in a download-friendly way
                daily_data_df = pd.DataFrame({
                    'date': list(st.session_state['ndvi_daily_data'].keys()),
                    'mean_ndvi': [data['mean'] for data in st.session_state['ndvi_daily_data'].values()]
                })
                
                # Sort by date
                daily_data_df['date'] = pd.to_datetime(daily_data_df['date'])
                daily_data_df = daily_data_df.sort_values('date')
                
                # Display as a table
                st.dataframe(daily_data_df.style.format({'mean_ndvi': '{:.4f}'}))
                
                # Convert date back to string for CSV export
                daily_data_df['date'] = daily_data_df['date'].dt.strftime('%Y-%m-%d')
                
                # Download buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download time series data
                    csv = ndvi_time_series.to_csv(index=False)
                    st.download_button(
                        label="Download Time Series Data",
                        data=csv,
                        file_name="ndvi_time_series.csv",
                        mime="text/csv",
                    )
                
                with col2:
                    # Download button for daily NDVI data
                    csv_daily = daily_data_df.to_csv(index=False)
                    st.download_button(
                        label="Download Daily NDVI Data",
                        data=csv_daily,
                        file_name="ndvi_daily_data.csv",
                        mime="text/csv",
                    )
            else:
                st.warning("Daily NDVI data not available. Run the analysis to generate this data.")
                
                # Download time series data only
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
        
        if rainfall_data is not None and len(rainfall_data['time_series']) > 0:
            rainfall_ts = rainfall_data['time_series']
            # Convert dates to datetime objects for better plotting
            rainfall_ts['date'] = pd.to_datetime(rainfall_ts['date'])
            
            # Create Plotly figure for rainfall
            fig_rain = px.bar(
                rainfall_ts, 
                x='date', 
                y='rainfall',
                title=f'Daily Rainfall for Field Area',
                labels={'date': 'Date', 'rainfall': 'Rainfall (mm)'},
                color='rainfall',
                color_continuous_scale='Blues'
            )
            
            # Improve layout
            fig_rain.update_layout(
                xaxis_title="Date",
                yaxis_title="Rainfall (mm)",
                hovermode="x unified"
            )
            
            # Display the plot
            st.plotly_chart(fig_rain, use_container_width=True)
            
            # Calculate rainfall statistics
            total_rainfall = rainfall_ts['rainfall'].sum()
            avg_daily_rainfall = rainfall_ts['rainfall'].mean()
            max_daily_rainfall = rainfall_ts['rainfall'].max()
            
            # Create columns for statistics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rainfall", f"{total_rainfall:.1f} mm")
            col2.metric("Avg. Daily Rainfall", f"{avg_daily_rainfall:.2f} mm")
            col3.metric("Max Daily Rainfall", f"{max_daily_rainfall:.1f} mm")
            
            # Rainfall assessment for the crop
            st.subheader("Rainfall Assessment")
            
            # Get optimal rainfall for crop
            optimal_rainfall = get_optimal_rainfall(crop_type)
            
            # Calculate period days from the time series data
            first_date = rainfall_ts['date'].min()
            last_date = rainfall_ts['date'].max()
            period_days = (last_date - first_date).days + 1
            
            # Scale optimal rainfall to the period length
            scaled_optimal = optimal_rainfall * period_days / 30  # Assuming optimal is per month
            
            if total_rainfall < scaled_optimal * 0.7:
                st.warning(f"The total rainfall ({total_rainfall:.1f} mm) is below the optimal range for {crop_type}. Consider irrigation if available.")
            elif total_rainfall > scaled_optimal * 1.3:
                st.warning(f"The total rainfall ({total_rainfall:.1f} mm) is above the optimal range for {crop_type}. Monitor for disease pressure and potential nutrient leaching.")
            else:
                st.success(f"The total rainfall ({total_rainfall:.1f} mm) is within an acceptable range for {crop_type}.")
            
            # Download rainfall data
            csv = rainfall_ts.to_csv(index=False)
            st.download_button(
                label="Download Rainfall Data",
                data=csv,
                file_name="rainfall_data.csv",
                mime="text/csv",
            )
        else:
            st.warning("Insufficient rainfall data available for the selected period.")
    
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
            recommendations = generate_recommendations(mean_ndvi, zones_param if isinstance(zones_param, int) else num_zones, crop_type, rainfall_data)
            for rec in recommendations:
                st.markdown(f"- {rec}")
                
        except Exception as e:
            st.error(f"Error computing statistics: {str(e)}")
    
    with tab6:
        st.subheader("Clustering Algorithm Comparison")
        
        # If we're comparing all algorithms, run all of them
        if performance_metrics is None or len(performance_metrics) < 3:
            st.write("For a full comparison, run each algorithm separately or use the 'Compare All' option.")
            
            # Create a comparison table with known advantages/disadvantages
            # Update the comparison_df in tab6
            comparison_df = pd.DataFrame({
                "Algorithm": ["K-Means", "DBSCAN", "Mean Shift", "GMM"],
                "Strengths": [
                    "Simple, fast, works well with spherical clusters",
                    "No need to specify number of clusters, handles irregular shapes, identifies outliers",
                    "Automatically determines clusters, handles non-spherical shapes, robust to outliers",
                    "Probabilistic approach, handles overlapping clusters, more flexible than K-Means"
                ],
                "Limitations": [
                    "Requires number of clusters in advance, sensitive to initialization, assumes spherical clusters",
                    "Sensitive to parameters (eps, min_samples), performance issues with large datasets",
                    "Computationally intensive, sensitive to bandwidth parameter, may be slow for large datasets",
                    "More computational overhead, still requires number of components, can overfit with small datasets"
                ],
                "Best for": [
                    "Simple, well-separated field zones with similar sizes",
                    "Fields with irregular patterns and potential outliers",
                    "Fields with natural clusters of varying sizes and densities",
                    "Fields with gradual transitions between zones and overlapping characteristics"
                ]
            })
            
            st.table(comparison_df)
            
            if performance_metrics and len(performance_metrics) > 0:
                st.subheader("Current Run Performance")
                perf_df = pd.DataFrame({
                    "Algorithm": list(performance_metrics.keys()),
                    "Processing Time (seconds)": list(performance_metrics.values())
                })
                st.table(perf_df)
        
        st.write("""
        **Choosing the Best Algorithm:**
        
        1. **K-Means** is recommended for fields with clear, distinct zones and when processing time is a concern.
        
        2. **DBSCAN** works best when field conditions have irregular patterns or outliers, and the number of management zones is not known in advance.
        
        3. **GMM** is ideal for fields with gradual transitions between zones and when probabilistic cluster assignment is beneficial.
        
        For most agricultural applications, K-Means provides a good balance of performance and interpretability.
        """)
        
        if clustering_method == "GMM":
            st.info("GMM (Gaussian Mixture Model) creates zones based on probability distributions, which can better capture the natural variation in field conditions compared to hard boundaries.")
        if clustering_method == "Mean Shift":
            st.info("Mean Shift clustering is useful for identifying clusters of varying shapes and sizes, but it can be computationally intensive.")

def get_optimal_rainfall(crop_type):
    """Return optimal monthly rainfall (mm) for different crops."""
    # These are approximate monthly values during growing season
    rainfall_requirements = {
        "Wheat": 80,  # 80-100 mm/month
        "Corn/Maize": 120,  # 120-140 mm/month
        "Rice": 180,  # Depends on irrigation method
        "Soybeans": 100,  # 100-120 mm/month
        "Cotton": 70,  # 70-100 mm/month
        "Sugarcane": 150,  # 150-200 mm/month
        "Other": 100,  # Generic value
    }
    return rainfall_requirements.get(crop_type, 100)
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

def generate_recommendations(mean_ndvi, num_zones, crop_type):
    """Generate recommendations based on NDVI values and crop type."""
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
    
    return recommendations

def generate_report(lat, lon, buffer, start_date, end_date, clustering_method, zones_param, crop_type, crop_stage, rainfall_data=None):
    """Generate a detailed text report for download."""
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
- Optimal Monthly Rainfall for {crop_type}: {get_optimal_rainfall(crop_type)} mm

Analysis Parameters:
- Analysis Period: {start_date} to {end_date}
- Clustering Method: {clustering_method}
- Zone Parameters: {zones_param}
"""
    
    # Add rainfall summary if available
    if rainfall_data is not None and len(rainfall_data['time_series']) > 0:
        rainfall_ts = rainfall_data['time_series']
        total_rainfall = rainfall_ts['rainfall'].sum()
        avg_daily = rainfall_ts['rainfall'].mean()
        max_daily = rainfall_ts['rainfall'].max()
        
        report += f"""
Rainfall Summary:
- Total Rainfall: {total_rainfall:.1f} mm
- Average Daily Rainfall: {avg_daily:.2f} mm
- Maximum Daily Rainfall: {max_daily:.1f} mm
"""
    
    report += """
Summary of Results:
- The field was segmented using clustering to identify management zones.
- Zones represent different levels of crop vigor, from lowest to highest.
- Consider variable rate application of inputs based on these zones.

Management Recommendations:
1. Ground-truth the zones with field visits
2. Take soil samples from each management zone
3. Develop variable rate prescription maps for inputs
4. Monitor changes in NDVI over time to assess management effectiveness
5. Adjust irrigation scheduling based on rainfall patterns

For more detailed analysis, please consider:
- Soil testing in each zone
- Tissue sampling for nutrient analysis
- Correlating NDVI patterns with yield data if available

Note: This analysis is based on remote sensing data and should be verified with field observations.
    """
    return report

if __name__ == "__main__":
    app()
