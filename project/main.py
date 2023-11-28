from api import StreetViewAPI
from helpers import Shapefile, unpackHDF
import asyncio
import matplotlib.pyplot as plt


async def main(locations, batchSize=100):
    trainImages, validImages, trainCoords, validCoords = unpackHDF(
        "./data/compressed/NC.h5"
    )


if __name__ == "__main__":
    asyncio.run(main(2300))
