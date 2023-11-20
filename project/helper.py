import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import random 

class Shapefile:
    def __init__(self, path):
        """
        path -> path to a .shx shapefile
        """
        self.gdf = gpd.read_file(path)
        print(self.gdf.head())

    def plot(self, size = 5):
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
    Finds the distance between two long/lat pairs in miles
    first -> first long/lat pair
    second -> second long/lat pair
    """
    pass

def savePairs(coordinates) -> None:
    """
    Save the long/lat pairs to an .npz file for later use by the api
    coordinates -> 2d list of long/lat pairs
    """
    pass


if __name__ == "__main__":
    gpd.options.io_engine = "pyogrio" # use the faster engine for geopandas
    
    path = "data/shapefiles/States_shapefile.shx"
    s = Shapefile(path)
    # s.plot()
    s.generateCoordinates()