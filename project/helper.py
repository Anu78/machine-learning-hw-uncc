import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import random 

class Shapefile:
    def __init__(self, path):
        """
        path -> path to a .shx shapefile
        """
        gpd.options.io_engine = "pyogrio" # use the faster engine for geopandas
        self.gdf = gpd.read_file(path)
        print(self.gdf.head())

    def plot(self, size = 5):
        """
        Plots the polygons in a shapefile
        """
        _, ax = plt.subplots(figsize=(size, size))
        self.gdf.plot(ax=ax, color='orange', edgecolor='black', linewidth=0.5)
        ax.set_title('US States Map')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()
    
    def generateCoordinates(self):
        for index, row in self.gdf.iterrows():
            polygon = row['geometry']
            print(f"processing {row['State_Code']} | area is {polygon.area}")
            


def coordinate_to_miles(first, second) -> float:
    """
    Returns the distance between two lat/long pairs (in miles)
    """
    pass

def savePairs(coordinates) -> None:
    """
    Saves a coordinates array as an npz file for future use.
    """
    pass