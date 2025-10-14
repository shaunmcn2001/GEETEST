import streamlit as st
import geemap.foliumap as geemap
import pandas as pd
import matplotlib.colors as mcolors
import ee
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def display_results(ndvi_image, geometry, lat, lon, num_zones, clustering_method="kmeans"):
    """Display analysis results in Streamlit tabs."""
    # Run the selected clustering method
    if clustering_method == "kmeans":
        zoned_image = kmeans_clustering(ndvi_image, geometry, num_zones)
        method_name = "K-means"
    elif clustering_method == "gmm":
        zoned_image = gmm_clustering(ndvi_image, geometry, num_zones)
        method_name = "Gaussian Mixture"
    else:
        st.error("Invalid clustering method")
        return
    
    tab1, tab2, tab3 = st.tabs(["NDVI Map", f"Field Zones ({method_name})", "Analysis"])
    
    with tab1:
        _display_ndvi_map(ndvi_image, geometry, lon, lat)
    
    with tab2:
        _display_zones(zoned_image, geometry, lon, lat, num_zones)
    
    with tab3:
        _display_statistics(ndvi_image, zoned_image, geometry, num_zones)

def kmeans_clustering(ndvi_image, geometry, num_zones):
    """Perform K-means clustering on NDVI image."""
    # Sample points from the NDVI image
    samples = ndvi_image.sample(
        region=geometry,
        scale=10,
        numPixels=1000,
        seed=42,
        geometries=True
    )
    
    # Get the image data as a list of dictionaries
    sample_data = samples.getInfo()
    
    # Extract NDVI values
    ndvi_values = [feature['properties']['NDVI'] for feature in sample_data['features']]
    
    # Reshape for K-means
    X = np.array(ndvi_values).reshape(-1, 1)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_zones, random_state=42).fit(X)
    
    # Create cluster centers and sort them to ensure consistent zone numbering
    centers = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(centers)
    zone_map = {old_idx: new_idx + 1 for new_idx, old_idx in enumerate(sorted_indices)}
    
    # Function to map each NDVI value to a zone
    def ndvi_to_zone(ndvi):
        distances = [abs(ndvi - center) for center in centers]
        cluster = np.argmin(distances)
        return zone_map[cluster]
    
    # Create a zone image by mapping NDVI values to zones
    zoned_image = ndvi_image.select('NDVI').clip(geometry).map(
        lambda img: img.expression(
            'ndvi_to_zone(ndvi)', {
                'ndvi': img.select('NDVI'),
                'ndvi_to_zone': ee.Function(ndvi_to_zone)
            }
        ).rename('zone')
    )
    
    return zoned_image

def gmm_clustering(ndvi_image, geometry, num_zones):
    """Perform Gaussian Mixture Model clustering on NDVI image."""
    # Sample points from the NDVI image
    samples = ndvi_image.sample(
        region=geometry,
        scale=10,
        numPixels=1000,
        seed=42,
        geometries=True
    )
    
    # Get the image data as a list of dictionaries
    sample_data = samples.getInfo()
    
    # Extract NDVI values
    ndvi_values = [feature['properties']['NDVI'] for feature in sample_data['features']]
    
    # Reshape for GMM
    X = np.array(ndvi_values).reshape(-1, 1)
    
    # Apply GMM clustering
    gmm = GaussianMixture(n_components=num_zones, random_state=42).fit(X)
    
    # Create cluster centers and sort them to ensure consistent zone numbering
    centers = gmm.means_.flatten()
    sorted_indices = np.argsort(centers)
    zone_map = {old_idx: new_idx + 1 for new_idx, old_idx in enumerate(sorted_indices)}
    
    # Function to map each NDVI value to a zone
    def ndvi_to_zone(ndvi):
        # Calculate the probability of this NDVI value belonging to each cluster
        probs = [
            np.exp(-0.5 * ((ndvi - center) / np.sqrt(gmm.covariances_[i][0][0]))**2) / 
            np.sqrt(2 * np.pi * gmm.covariances_[i][0][0])
            for i, center in enumerate(centers)
        ]
        cluster = np.argmax(probs)
        return zone_map[cluster]
    
    # Create a zone image by mapping NDVI values to zones
    zoned_image = ndvi_image.select('NDVI').clip(geometry).map(
        lambda img: img.expression(
            'ndvi_to_zone(ndvi)', {
                'ndvi': img.select('NDVI'),
                'ndvi_to_zone': ee.Function(ndvi_to_zone)
            }
        ).rename('zone')
    )
    
    return zoned_image

def _display_ndvi_map(ndvi_image, geometry, lon, lat):
    """Helper function for NDVI map display."""
    m = geemap.Map()
    m.centerObject(ee.Geometry.Point([lon, lat]), 16)
    ndvi_vis = {
        'min': 0, 'max': 0.8,
        'palette': ['#d73027', '#f46d43', '#fdae61', '#fee08b', 
                   '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850']
    }
    m.addLayer(ndvi_image.select('NDVI').clip(geometry), ndvi_vis, 'NDVI')
    m.add_colorbar(ndvi_vis, label="NDVI Values")
    m.to_streamlit(height=500)

def _display_zones(zoned_image, geometry, lon, lat, num_zones):
    """Helper function for zone map display."""
    m = geemap.Map()
    m.centerObject(ee.Geometry.Point([lon, lat]), 16)
    
    # Create a color palette for zones
    colors = list(mcolors.TABLEAU_COLORS.values())[:num_zones]
    colors = [mcolors.rgb2hex(mcolors.to_rgb(color)) for color in colors]
    
    zone_vis = {
        'min': 1,
        'max': num_zones,
        'palette': colors
    }
    
    m.addLayer(zoned_image.clip(geometry), zone_vis, 'Management Zones')
    m.add_colorbar(zone_vis, label="Zone Number")
    m.to_streamlit(height=500)

def _display_statistics(ndvi_image, zoned_image, geometry, num_zones):
    """Helper function for displaying statistics."""
    # Calculate zonal statistics
    zonal_stats = ee.Image.cat([
        ndvi_image.select('NDVI'),
        zoned_image.select('zone')
    ]).reduceRegion(
        reducer=ee.Reducer.mean().group(
            groupField=1,
            groupName='zone',
        ),
        geometry=geometry,
        scale=10,
        maxPixels=1e9
    ).get('groups')
    
    # Convert to client-side object
    stats_data = zonal_stats.getInfo()
    
    # Create DataFrame
    data = []
    for zone_data in stats_data:
        zone_num = zone_data['zone']
        ndvi_mean = zone_data['mean']
        data.append({
            'Zone': f"Zone {zone_num}",
            'Mean NDVI': round(ndvi_mean, 3),
            'Recommendation': _get_recommendation(ndvi_mean)
        })
    
    df = pd.DataFrame(data)
    
    # Display statistics
    st.subheader("Zonal Statistics")
    st.dataframe(df)
    
    # Create a bar chart
    st.subheader("Zone NDVI Comparison")
    st.bar_chart(df.set_index('Zone')['Mean NDVI'])

def _get_recommendation(ndvi_value):
    """Generate agricultural recommendations based on NDVI values."""
    if ndvi_value < 0.2:
        return "Very low vegetation - Consider soil test and replanting"
    elif ndvi_value < 0.4:
        return "Low vegetation - Increase irrigation and fertilization"
    elif ndvi_value < 0.6:
        return "Moderate vegetation - Maintain current management"
    else:
        return "Healthy vegetation - Optimal growing conditions"

# Example of how to use the modified function in your Streamlit app
def main():
    st.title("Agricultural Field Management Zones")
    
    # Sample inputs (replace with your actual interface)
    lat = st.number_input("Latitude", value=42.5)
    lon = st.number_input("Longitude", value=-85.0)
    radius = st.number_input("Field radius (meters)", value=100, min_value=10)
    num_zones = st.slider("Number of management zones", min_value=2, max_value=7, value=3)
    clustering_method = st.selectbox(
        "Clustering Method", 
        options=["kmeans", "gmm"], 
        format_func=lambda x: "K-means" if x == "kmeans" else "Gaussian Mixture"
    )
    
    # Create geometry and fetch NDVI data
    geometry = ee.Geometry.Point([lon, lat]).buffer(radius)
    
    # Get most recent Sentinel-2 image
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(geometry) \
        .sort("system:time_start", False) \
        .first()
    
    # Calculate NDVI
    ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndvi_image = ee.Image.cat([s2, ndvi])
    
    # Display results
    if st.button("Generate Management Zones"):
        display_results(ndvi_image, geometry, lat, lon, num_zones, clustering_method)

if __name__ == "__main__":
    # Initialize Earth Engine
    try:
        ee.Initialize()
    except Exception as e:
        st.error(f"Failed to initialize Earth Engine: {e}")
        st.info("Please authenticate GEE before running this app")
    main()