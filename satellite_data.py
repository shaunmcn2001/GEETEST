import ee

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