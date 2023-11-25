from api import StreetViewAPI
from helpers import Shapefile


def main():
    # path = "data/shapefiles/States_shapefile.shx"
    # s = Shapefile(path)
    # res = s.generateCoordinates()
    # plots all coverage points over US map - will move to helper function later
    # s.plot()
    # for j in range(len(res)):
    #     plt.scatter([shapely.get_x(i) for i in res[j]], [shapely.get_y(i) for i in res[j]])
    # plt.show()

    s = StreetViewAPI(imageSize=(640, 480), fov=80)

    s.testAPI(0, 0, (34.492661, -85.844476))


if __name__ == "__main__":
    main()
