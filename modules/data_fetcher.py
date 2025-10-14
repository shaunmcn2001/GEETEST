import ee
from datetime import datetime

def get_sentinel2_collection(start_date, end_date, geometry):
    """Fetch Sentinel-2 imagery collection."""
    start = ee.Date(start_date.strftime('%Y-%m-%d'))
    end = ee.Date(end_date.strftime('%Y-%m-%d'))
    
    return ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
        .filterDate(start, end) \
        .filterBounds(geometry) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

def calculate_ndvi(image_collection):
    """Calculate NDVI for an image collection."""
    def add_ndvi(image):
        return image.addBands(
            image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        )
    return image_collection.map(add_ndvi)