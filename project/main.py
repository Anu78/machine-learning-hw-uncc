from api import StreetViewAPI
from helpers import Shapefile, unpackHDF
import asyncio
import matplotlib.pyplot as plt
from model import train, validate
import os

async def createDataset(states, batchSize, shapefilePath, h5Path):
    label = 1
    api = StreetViewAPI((640, 480), 80)
    shapefiles = sorted(os.listdir("./data/shapefiles"))
    for state in shapefiles:
        s = Shapefile(os.path.join("./data/shapefiles", state, "Shapefile.shp"), state)
        coordinates = s.generateCoordinates()
        api.saveImages()
        label += 1

async def main():
    # create dataset, save to ./data/compressed
    await createDataset(states = ["NC"], batchSize=100, shapefilePath="./data/shapefiles", h5Path="./data/compressed/")
    

if __name__ == "__main__":
    asyncio.run(main())