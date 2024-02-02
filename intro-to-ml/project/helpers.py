from __future__ import division
from shapely.geometry import LineString
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely import get_x, get_y

class Shapefile:
    def __init__(
        self,
        state,
        locations,
        path=None,
    ):
        gpd.options.io_engine = "pyogrio"  # use the faster engine for geopandas
        self.gdf = None
        
        if path is not None:
            self.gdf = gpd.read_file(path)
            print(f"Initialized {len(self.gdf)} roads in {state}.")

        # need to generate more locations to compensate for invalids later on
        self.locations = locations
        self.state = state
    
    def plotState(self):
        res = self.generateCoordinates()

        x = [i[1] for i in res]
        y = [i[0] for i in res]

        plt.scatter(x, y)
        plt.show()

    def plot(self, size=5):
        if self.gdf is None:
            raise ValueError("You have not defined a gdf path.")
        _, ax = plt.subplots(figsize=(size, size))
        self.gdf.plot(ax=ax, color="orange", edgecolor="black", linewidth=0.5)
        ax.set_title(f"Map of roads in {self.state}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()

    def generateCoordinates(self):
        import random
        if self.gdf is None:
            raise ValueError("You have not defined a gdf path.")

        l = len(self.gdf)
        coordinates = []
        i = 0
        self.increment = l // self.locations
        # recalc_interval = l // 20  # Interval at which to recalculate increment
        # last_recalc_index = 0  # Tracks the last index where increment was recalculated
        # inc = self.increment

        while i < l:
            linestring = LineString(self.gdf.iloc[i]["geometry"])
            
            r = random.uniform(0, linestring.length)
            point = linestring.interpolate(r)

            coordinates.append((get_y(point), get_x(point)))

            # # Check if it's time to recalculate increment for the next set of rows
            # if i - last_recalc_index >= recalc_interval or i == 0:
            #     next_segment_end = min(i + recalc_interval, l)
            #     inc = self.__calculateDensityIncrement(gdf=self.gdf.iloc[i:next_segment_end]["geometry"])
            #     last_recalc_index = i  # Update last recalculation index

            i += self.increment

        return coordinates

    def combine(self, path):
        try:
            if self.gdf is None:
                self.gdf = gpd.read_file(path)
                return
            gdf2 = gpd.read_file(path)

            # Perform the combination here inside the try block, where gdf2 is known to be defined
            combined_shape = gpd.GeoDataFrame(
                pd.concat([self.gdf, gdf2], ignore_index=True)
            )
            combined_shape = combined_shape.set_geometry(combined_shape.geometry)

            self.gdf = combined_shape

        except Exception as e:
            print(f"An error occurred while reading the shapefile.\n {e}")
            return

    def write(self, path):
        import os
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)

        self.gdf.to_file(path)