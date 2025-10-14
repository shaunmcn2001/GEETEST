from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
import json

app = FastAPI(title="NDVI Field Segmentation API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Earth Engine
try:
    ee.Initialize(project='ndvi-field-segmentation')
except Exception:
    # For production, you'll need to set up service account authentication
    pass

class FieldAnalysisRequest(BaseModel):
    latitude: float
    longitude: float
    buffer_size: int = 250
    start_date: str
    end_date: str
    clustering_method: str = "K-Means"
    num_zones: int = 3
    crop_type: str = "Wheat"
    crop_growth_stage: str = "Vegetative"
    eps_value: float = 0.05
    min_samples: int = 10
    bandwidth: float = 0.1

class NDVITimeSeriesResponse(BaseModel):
    dates: List[str]
    ndvi_values: List[float]
    mean_ndvi: float
    trend: str

class RainfallResponse(BaseModel):
    dates: List[str]
    rainfall_values: List[float]
    total_rainfall: float
    avg_daily_rainfall: float
    max_daily_rainfall: float

class FieldAnalysisResponse(BaseModel):
    ndvi_stats: Dict[str, float]
    zones_identified: int
    processing_time: float
    recommendations: List[str]
    ndvi_time_series: NDVITimeSeriesResponse
    rainfall_data: Optional[RainfallResponse]

def get_crop_ndvi_range(crop_type: str) -> str:
    """Return typical NDVI range for different crops."""
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

def get_optimal_rainfall(crop_type: str) -> int:
    """Return optimal monthly rainfall (mm) for different crops."""
    rainfall_requirements = {
        "Wheat": 80,
        "Corn/Maize": 120,
        "Rice": 180,
        "Soybeans": 100,
        "Cotton": 70,
        "Sugarcane": 150,
        "Other": 100,
    }
    return rainfall_requirements.get(crop_type, 100)

def get_sentinel2_collection(start_date: str, end_date: str, geometry):
    """Fetch Sentinel-2 imagery for the given time period and location."""
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    
    s2 = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
        .filterDate(start, end) \
        .filterBounds(geometry) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    
    return s2

def calculate_ndvi(image_collection):
    """Calculate NDVI for each image in the collection."""
    def add_ndvi(image):
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi).set('date', image.date().format('YYYY-MM-dd'))
    
    return image_collection.map(add_ndvi)

def extract_ndvi_time_series(ndvi_collection, geometry):
    """Extract NDVI time series data."""
    image_list = ndvi_collection.toList(ndvi_collection.size())
    size = image_list.size().getInfo()
    
    dates = []
    mean_ndvi_values = []
    
    for i in range(size):
        image = ee.Image(image_list.get(i))
        date_str = image.get('date').getInfo()
        
        mean_ndvi = image.select('NDVI').reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=10,
            maxPixels=1e9
        ).get('NDVI').getInfo()
        
        if mean_ndvi is not None:
            dates.append(date_str)
            mean_ndvi_values.append(mean_ndvi)
    
    return dates, mean_ndvi_values

def get_rainfall_data(start_date: str, end_date: str, geometry):
    """Fetch precipitation data from CHIRPS dataset."""
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    
    rainfall = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
        .filterDate(start, end) \
        .filterBounds(geometry)
    
    # Extract time series
    image_list = rainfall.toList(rainfall.size())
    size = image_list.size().getInfo()
    
    dates = []
    rainfall_values = []
    
    for i in range(min(size, 100)):  # Limit to avoid timeout
        image = ee.Image(image_list.get(i))
        date = image.date()
        date_str = date.format('YYYY-MM-dd').getInfo()
        
        mean_rainfall = image.select('precipitation').reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=5000,
            maxPixels=1e9
        ).get('precipitation').getInfo()
        
        if mean_rainfall is not None:
            dates.append(date_str)
            rainfall_values.append(mean_rainfall)
    
    return dates, rainfall_values

def perform_clustering(ndvi_image, geometry, method: str, **params):
    """Perform clustering based on the selected method."""
    start_time = datetime.now()
    
    if method == "K-Means":
        num_zones = params.get('num_zones', 3)
        ndvi_sample = ndvi_image.select('NDVI').sampleRegions(
            collection=geometry,
            scale=10,
            geometries=True
        )
        clusterer = ee.Clusterer.wekaKMeans(num_zones).train(ndvi_sample)
        result = ndvi_image.select('NDVI').cluster(clusterer)
        zones_identified = num_zones
    else:
        # For other methods, we'll use K-Means as fallback for now
        # In a full implementation, you'd implement the other clustering methods
        zones_identified = params.get('num_zones', 3)
        ndvi_sample = ndvi_image.select('NDVI').sampleRegions(
            collection=geometry,
            scale=10,
            geometries=True
        )
        clusterer = ee.Clusterer.wekaKMeans(zones_identified).train(ndvi_sample)
        result = ndvi_image.select('NDVI').cluster(clusterer)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    return result, zones_identified, processing_time

def generate_recommendations(mean_ndvi: float, num_zones: int, crop_type: str, rainfall_data: Optional[RainfallResponse]) -> List[str]:
    """Generate recommendations based on analysis results."""
    recommendations = []
    
    if mean_ndvi < 0.3:
        recommendations.append("The field shows signs of stress. Consider irrigation or nutrient assessment.")
        recommendations.append("Low NDVI values may indicate bare soil or early growth stage.")
    elif mean_ndvi < 0.5:
        recommendations.append("Field has moderate vegetation health. Monitor for changes in coming weeks.")
        recommendations.append("Consider targeted fertilizer application in lower-performing zones.")
    else:
        recommendations.append("Field shows good overall vegetation health.")
        recommendations.append("Focus on maintaining current management practices.")
    
    # Crop-specific recommendations
    if crop_type == "Wheat" and mean_ndvi < 0.4:
        recommendations.append("For wheat at this NDVI level, check for nitrogen deficiency.")
    elif crop_type == "Corn/Maize" and mean_ndvi < 0.5:
        recommendations.append("For corn at this NDVI level, evaluate water stress and consider irrigation if available.")
    
    if num_zones >= 3:
        recommendations.append(f"Consider variable rate application based on the {num_zones} identified management zones.")
        recommendations.append("Take soil samples from each zone to determine specific nutrient requirements.")
    
    return recommendations

@app.get("/")
async def root():
    return {"message": "NDVI Field Segmentation API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/analyze-field", response_model=FieldAnalysisResponse)
async def analyze_field(request: FieldAnalysisRequest):
    try:
        # Create geometry
        geometry = ee.Geometry.Point([request.longitude, request.latitude]).buffer(request.buffer_size)
        
        # Get Sentinel-2 imagery
        s2_collection = get_sentinel2_collection(request.start_date, request.end_date, geometry)
        
        # Calculate NDVI
        ndvi_collection = calculate_ndvi(s2_collection)
        median_ndvi = ndvi_collection.median()
        
        # Get NDVI time series
        dates, ndvi_values = extract_ndvi_time_series(ndvi_collection, geometry)
        
        if not ndvi_values:
            raise HTTPException(status_code=400, detail="No NDVI data available for the specified period")
        
        mean_ndvi = np.mean(ndvi_values)
        
        # Calculate trend
        if len(ndvi_values) >= 3:
            x = np.array(range(len(ndvi_values)))
            y = np.array(ndvi_values)
            coeffs = np.polyfit(x, y, 1)
            trend = "increasing" if coeffs[0] > 0 else "decreasing"
        else:
            trend = "insufficient data"
        
        # Perform clustering
        zoned_image, zones_identified, processing_time = perform_clustering(
            median_ndvi, 
            geometry, 
            request.clustering_method,
            num_zones=request.num_zones,
            eps_value=request.eps_value,
            min_samples=request.min_samples,
            bandwidth=request.bandwidth
        )
        
        # Get NDVI statistics
        ndvi_stats = median_ndvi.select('NDVI').reduceRegion(
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
        
        # Get rainfall data
        try:
            rainfall_dates, rainfall_values = get_rainfall_data(request.start_date, request.end_date, geometry)
            rainfall_data = RainfallResponse(
                dates=rainfall_dates,
                rainfall_values=rainfall_values,
                total_rainfall=sum(rainfall_values) if rainfall_values else 0,
                avg_daily_rainfall=np.mean(rainfall_values) if rainfall_values else 0,
                max_daily_rainfall=max(rainfall_values) if rainfall_values else 0
            )
        except:
            rainfall_data = None
        
        # Generate recommendations
        recommendations = generate_recommendations(mean_ndvi, zones_identified, request.crop_type, rainfall_data)
        
        # Prepare response
        ndvi_time_series = NDVITimeSeriesResponse(
            dates=dates,
            ndvi_values=ndvi_values,
            mean_ndvi=mean_ndvi,
            trend=trend
        )
        
        response = FieldAnalysisResponse(
            ndvi_stats={
                "mean": ndvi_stats.get('NDVI_mean', 0),
                "stddev": ndvi_stats.get('NDVI_stdDev', 0),
                "min": ndvi_stats.get('NDVI_min', 0),
                "max": ndvi_stats.get('NDVI_max', 0)
            },
            zones_identified=zones_identified,
            processing_time=processing_time,
            recommendations=recommendations,
            ndvi_time_series=ndvi_time_series,
            rainfall_data=rainfall_data
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/crop-info/{crop_type}")
async def get_crop_info(crop_type: str):
    return {
        "ndvi_range": get_crop_ndvi_range(crop_type),
        "optimal_rainfall": get_optimal_rainfall(crop_type)
    }
