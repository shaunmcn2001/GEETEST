import ee

def perform_zoning(ndvi_image, geometry, num_zones):
    """Perform K-means clustering for field zoning."""
    ndvi_sample = ndvi_image.select('NDVI').sampleRegions(
        collection=geometry,
        scale=10,
        geometries=True
    )
    clusterer = ee.Clusterer.wekaKMeans(num_zones).train(ndvi_sample)
    return ndvi_image.select('NDVI').cluster(clusterer)