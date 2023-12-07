from api import StreetViewAPI
from helpers import Shapefile
import asyncio
from model import train, validate
import os
import time

async def main():
    # parameters
    locationsPerState = 15

    # create dataset
    api = StreetViewAPI(imageSize=(640, 480), fov=90, batchSize=250, imagesPerState=locationsPerState, csvPath="./data/valid.csv")
    shapefiles = sorted(os.listdir("./data/shapefiles"))
    for state in shapefiles:
        if state == "DC":
            continue
        s = Shapefile(state=state, path=f"./data/shapefiles/{state}/Shapefile.shp", locations=locationsPerState)
        coordinates = s.generateCoordinates()
        await api.saveImages(coordinates=coordinates)

        print(f"Finished {state}.")

        time.sleep(4)
 
    # # train model 
    # train(epochs=20, batchSize=32, shuffle=True, lr=0.01)

    # # validate model
    # validate()


if __name__ == "__main__":
    asyncio.run(main())