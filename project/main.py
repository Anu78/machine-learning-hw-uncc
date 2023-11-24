from api import StreetViewAPI
from dotenv import load_dotenv
from helpers import Shapefile
from helpers import distBetweenCoordinates
import matplotlib.pyplot as plt

def main():
    path = "data/shapefiles/States_shapefile.shx"
    s = Shapefile(path) 
    res = s.generateCoordinates()

    # plots all coverage points over US map - will move to helper function later
    # s.plot()
    # for j in range(len(res)):
    #     plt.scatter([shapely.get_x(i) for i in res[j]], [shapely.get_y(i) for i in res[j]])
    # plt.show()


if __name__ == "__main__":
    main()
