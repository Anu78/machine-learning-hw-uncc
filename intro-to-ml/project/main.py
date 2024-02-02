from api import StreetViewAPI
from helpers import Shapefile
import asyncio
import os
import time

async def main():
    # # parameters
    # locationsPerState = 20

    # # create dataset
    # api = StreetViewAPI(imageSize=(640, 480), fov=90, batchSize=250, imagesPerState=locationsPerState, csvPath="./data/valid.csv")
    # shapefiles = sorted(os.listdir("./data/shapefiles"))
    # for state in shapefiles:
    #     if state in ["DC", "AK", "HI"]:
    #         continue
    #     s = Shapefile(state=state, path=f"./data/shapefiles/{state}/Shapefile.shp", locations=locationsPerState)
    #     coordinates = s.generateCoordinates()
    #     await api.saveImages(coordinates=coordinates)

    #     print(f"Finished {state}.")

    #     time.sleep(7)
 
    # train model 
    # train(epochs=20, batchSize=32, momentum=0.9, lr=0.001)

    # # validate model
    # validate()

    s = Shapefile(state="NC", path="./data/shapefiles/NC/Shapefile.shp", locations=200)

    s.plot()

if __name__ == "__main__":
    asyncio.run(main())