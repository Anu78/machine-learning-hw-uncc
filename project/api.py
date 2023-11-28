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
        self.batchSize = 50
        self.compressionStrength = 6

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
        return validCoords

    async def fetchMultiple(self, urls):
        responses = []
        n = len(urls) // self.batchSize
        for i in range(0, len(urls), self.batchSize):
            batch = urls[i : i + self.batchSize]
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

    async def saveImages(self, coordinates, path, batchSize=50):
        self.batchSize = batchSize
        print(
            f"Starting the process to save images for {len(coordinates)} coordinates."
        )

        validatedCoordinates = await self.validateCoordinates(coordinates)
        self.coordinates = np.array(validatedCoordinates)
        print(f"Number of coordinates after validation: {len(self.coordinates)}")

        imageUrls = await self.generateURLs(self.coordinates)
        byteDataList = await self.fetchMultiple(imageUrls)

        if len(byteDataList) != len(self.coordinates):
            print(
                f"Warning: Number of fetched images ({len(byteDataList)}) does not match number of coordinates ({len(self.coordinates)})"
            )

        imageList = []
        for byteData in byteDataList:
            try:
                image = Image.open(io.BytesIO(byteData))
                npImg = np.array(image, dtype=np.uint8)
                imageList.append(npImg)
            except Exception as e:
                print(f"Error processing image: {e}")

        self.images = np.array(imageList)
        print(f"Number of images after fetching and processing: {len(self.images)}")

        self.writeDataToFile(path)

    def writeDataToFile(self, path):
        # split dataset into training and validation
        lenI = int(len(self.images) * 0.9)
        lenC = int(len(self.coordinates) * 0.9)
        print(f"Writing {lenI} images to {path}!")
        with h5py.File(path, "w") as hdf:
            hdf.create_dataset(
                "trainImages",
                data=self.images[:lenI],
                compression="gzip",
                compression_opts=self.compressionStrength,
            )
            hdf.create_dataset(
                "validImages",
                data=self.images[lenI:],
                compression="gzip",
                compression_opts=self.compressionStrength,
            )

            hdf.create_dataset(
                "trainCoords",
                data=self.coordinates[:lenC],
                compression="gzip",
                compression_opts=self.compressionStrength,
            )
            hdf.create_dataset(
                "validCoords",
                data=self.coordinates[lenC:],
                compression="gzip",
                compression_opts=self.compressionStrength,
            )
