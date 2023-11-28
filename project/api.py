import asyncio
import h5py
import random
import aiohttp
from PIL import Image
import io
import numpy as np
import requests
from dotenv import dotenv_values

config = dotenv_values(".env")
MAPS_KEY = config["MAPS_KEY"]


class StreetViewAPI:
    def __init__(self, imageSize, fov):
        self.imageSize = imageSize
        self.fov = fov
        self.staticBaseURL = "https://maps.googleapis.com/maps/api/streetview"
        self.metadataBaseURL = (
            "https://maps.googleapis.com/maps/api/streetview/metadata"
        )
        self.images = np.array([])
        self.coordinates = np.array([]) 

    async def generateURLs(self, coordinates, pitch=0):
        urls = []
        for lat, lng in coordinates:
            heading = random.randint(0, 360)
            url = f"{self.staticBaseURL}?size={self.imageSize[0]}x{self.imageSize[1]}&fov={self.fov}&location={lat},{lng}&heading={heading}&pitch={pitch}&key={MAPS_KEY}"
            urls.append(url)
        return urls

    async def validateCoordinates(self, coordinates):
        urls = [
            f"{self.metadataBaseURL}?location={lat},{lng}&key={MAPS_KEY}"
            for lat, lng in coordinates
        ]
        responses = await self.fetchMultiple(urls)
        validCoords = [
            (res["location"]["lat"], res["location"]["lng"])
            for res in responses
            if res["status"] == "OK"
        ]
        print(f"Validated {len(validCoords)} out of {len(coordinates)} coordinates.")
        return validCoords

    async def fetchMultiple(self, urls, batchSize=50):
        responses = []
        n = len(urls) // batchSize
        for i in range(0, len(urls), batchSize):
            print(f"on batch {i//batchSize} out of {n}")
            batch = urls[i : i + batchSize]
            async with aiohttp.ClientSession() as session:
                tasks = [self.fetchURL(url, session) for url in batch]
                responses.extend(await asyncio.gather(*tasks))
        return responses

    async def fetchURL(self, url, session):
        async with session.get(url) as response:
            content_type = response.headers["Content-Type"]
            if "application/json" in content_type:
                return await response.json()
            return await response.read()

    async def saveImages(self, coordinates, batchSize=50):
        self.coordinates = np.array(await self.validateCoordinates(coordinates))
        imageUrls = await self.generateURLs(self.coordinates)
        byteDataList = await self.fetchMultiple(imageUrls, batchSize=batchSize)

        for byteData in byteDataList:
            image = Image.open(io.BytesIO(byteData))
            npImg = np.array(image).astype(np.float32) / 255.0
            np.append(self.images, npImg)

        self.writeDataToFile("./data/compressed/NC.h5")

    def writeDataToFile(self, path):
        # split dataset into training and validation
        lenI = int(len(self.images) * 0.9)
        lenC = int(len(self.coordinates) * 0.9)

        with h5py.File(path, 'w') as hdf:
            hdf.create_dataset('trainImages', data=self.images[:lenI], compression="gzip", compression_opts=9)
            hdf.create_dataset('validImages', data=self.images[lenI:], compression="gzip", compression_opts=9)

            hdf.create_dataset('trainCoords', data=self.coordinates[lenC:], compression="gzip", compression_opts=9)
            hdf.create_dataset('validCoords', data=self.coordinates[:lenC], compression="gzip", compression_opts=9)
