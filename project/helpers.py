from __future__ import division
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely import Point

state_diversity = {
    'TX': 1, 'MT': 2, 'CA': 3, 'WA': 4, 'DE': 5, 'CO': 6, 'LA': 7, 'MD': 8, 'FL': 9, 'NJ': 10, 
    'ID': 11, 'OR': 12, 'NC': 13, 'UT': 14, 'RI': 15, 'OK': 16, 'MI': 17, 'HI': 18, 'NM': 18, 'WI': 19, 
    'OH': 20, 'MA': 21, 'MN': 22, 'SC': 23, 'NY': 24, 'TN': 25, 'VA': 26, 'WY': 27, 'SD': 28, 
    'CT': 29, 'KY': 30, 'GA': 31, 'MS': 32, 'AR': 33, 'NE': 34, 'MO': 35, 'PA': 36, 'AL': 37, 
    'IN': 38, 'KS': 39, 'AZ': 40, 'ND': 41, 'IL': 42, 'NV': 43, 'VT': 44, 'ME': 45, 'NH': 46, 
    'WV': 47, 'IA': 48
}

class Shapefile:
    def __init__(self, path, ):
        """
        path -> path to a .shx shapefile
        """
        gpd.options.io_engine = "pyogrio"  # use the faster engine for geopandas
        self.gdf = gpd.read_file(path)
        self.LOC_EQUATION = lambda area: -0.0415168*area*area + 8.4791*area + 23.43
        self.DIVER_EQUATION = lambda rank: 1.3 - ((rank-1)*(0.6))/(47)

    def plot(self, size=5):
        """
        Plots the polygons in a shapefile
        """
        _, ax = plt.subplots(figsize=(size, size))
        self.gdf.plot(ax=ax, color="orange", edgecolor="black", linewidth=0.5)
        ax.set_title("US States Map")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
    
    def gridSpacingForState(self, bounds, area, stateCode):
        """
        Returns the grid spacing for a state considering some parameters.
        """
        minx, miny, maxx, maxy = bounds
        
        locationsPerState = int(self.LOC_EQUATION(area) * self.DIVER_EQUATION(state_diversity[stateCode])) 
        return ((maxx - minx) * (maxy - miny)) / locationsPerState, locationsPerState

    def generateCoordinates(self):
        """
        Returns a list of valid coordinates for each polygon in a shapefile, seperated by self.gridSpace
        """
        final = np.array([])
        total = 0
        for _, row in self.gdf.iterrows():
            polygon = row["geometry"]
            area = polygon.area
            stateCode = row["State_Code"]
            
            if stateCode == "AK" or stateCode == "DC":
                continue # not enough google street view data in alaska; why is DC on the state list?  
             
            # calculate gridspace so that there are 400 locations per state. 
            gridSpace, locs = self.gridSpacingForState(polygon.bounds, area, stateCode)
            total += locs
            minx, miny, maxx, maxy = polygon.bounds

            print(f"processing {stateCode} | area: {area:.2f} | {locs} locations")

            gridPoints = np.array([])
            for x in np.arange(minx, maxx, gridSpace):
                for y in np.arange(miny, maxy, gridSpace):
                    point = Point(x, y)
                    if polygon.contains(point):
                        np.append(gridPoints, point)
            np.append(final, gridPoints)
        
        print(f"total of {total} locations. ")
        return final

def distBetweenCoordinates(first, second, iterations=100) -> float:
    """
    Returns the distance between two lat/long pairs (in miles)
    """
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


def savePairs(coordinates, labels, label_name) -> None:
    base_path = "./data/compressed/"
    """
    Saves images and labels as an npz file for training.
    """
    np.savez_compressed(file=base_path+label_name)
