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
        ax.set_title("Map of roads in state")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()

    def __calculateDensityIncrement(self, gdf):
        linestrings = [LineString(row) for row in gdf]
        total = sum(line.length for line in linestrings)

        box = gpd.GeoSeries(linestrings).unary_union.envelope 
        area = box.area
        density = total / area
        print(density)
        newInc = self.increment
        if density >= 35 and density < 50:
            newInc = self.increment * 5
        elif density >= 50:
            newInc = self.increment * 10
        
        return int(newInc)

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


def distBetweenCoordinates(first, second, iterations=100) -> float:
    from math import atan
    from math import atan2
    from math import cos
    from math import radians
    from math import sin
    from math import sqrt
    from math import tan

    a = 6378137.0  # radius at equator in meters (WGS-84)
    f = 1 / 298.257223563  # flattening of the ellipsoid (WGS-84)
    b = (1 - f) * a
    miles_conversion = 0.000621371

    (
        phi_1,
        L_1,
    ) = first  # (lat=L_?,lon=phi_?)
    (
        phi_2,
        L_2,
    ) = second

    u_1 = atan((1 - f) * tan(radians(phi_1)))
    u_2 = atan((1 - f) * tan(radians(phi_2)))

    L = radians(L_2 - L_1)

    Lambda = L  # set initial value of lambda to L

    sin_u1 = sin(u_1)
    cos_u1 = cos(u_1)
    sin_u2 = sin(u_2)
    cos_u2 = cos(u_2)

    tolerance = 10**-12

    # begin iterations
    for _ in range(iterations):
        cos_lambda = cos(Lambda)
        sin_lambda = sin(Lambda)
        sin_sigma = sqrt(
            (cos_u2 * sin(Lambda)) ** 2
            + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda) ** 2
        )
        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda
        sigma = atan2(sin_sigma, cos_sigma)
        sin_alpha = (cos_u1 * cos_u2 * sin_lambda) / sin_sigma
        cos_sq_alpha = 1 - sin_alpha**2
        cos2_sigma_m = cos_sigma - ((2 * sin_u1 * sin_u2) / cos_sq_alpha)
        C = (f / 16) * cos_sq_alpha * (4 + f * (4 - 3 * cos_sq_alpha))
        Lambda_prev = Lambda
        Lambda = L + (1 - C) * f * sin_alpha * (
            sigma
            + C
            * sin_sigma
            * (cos2_sigma_m + C * cos_sigma * (-1 + 2 * cos2_sigma_m**2))
        )

        # successful convergence
        diff = abs(Lambda_prev - Lambda)
        if diff <= tolerance:
            break

    u_sq = cos_sq_alpha * ((a**2 - b**2) / b**2)
    A = 1 + (u_sq / 16384) * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
    B = (u_sq / 1024) * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))
    delta_sig = (
        B
        * sin_sigma
        * (
            cos2_sigma_m
            + 0.25
            * B
            * (
                cos_sigma * (-1 + 2 * cos2_sigma_m**2)
                - (1 / 6)
                * B
                * cos2_sigma_m
                * (-3 + 4 * sin_sigma**2)
                * (-3 + 4 * cos2_sigma_m**2)
            )
        )
    )

    m = b * A * (sigma - delta_sig)

    return m * miles_conversion