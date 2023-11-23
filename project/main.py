import torch
from helper import Shapefile

def main():
    path = "data/shapefiles/States_shapefile.shx"
    s = Shapefile(path)
    s.plot()
    s.generateCoordinates()


if __name__ == "__main__":
    main()