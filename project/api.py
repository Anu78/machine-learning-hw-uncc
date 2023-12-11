import asyncio
import random
import aiohttp
import numpy as np
from dotenv import dotenv_values

config = dotenv_values(".env")
MAPS_KEY = config["MAPS_KEY"]

class StreetViewAPI:
    def __init__(self, imageSize, fov, batchSize, imagesPerState, csvPath):
        self.imageSize = imageSize
        self.fov = fov
        self.staticBaseURL = "https://maps.googleapis.com/maps/api/streetview"
        self.metadataBaseURL = (
            "https://maps.googleapis.com/maps/api/streetview/metadata"
        )
        self.images = np.array([])
        self.coordinates = np.array([])
        self.batchSize = batchSize
        self.imagesPerState = imagesPerState
        self.cur = 0
        self.csvPath = csvPath

    async def generateURLs(self, coordinates, pitch=0):
        urls = []
        for lat, lng in coordinates:
            heading = random.randint(0, 360 - 90)
            url = f"{self.staticBaseURL}?size={self.imageSize[0]}x{self.imageSize[1]}&fov={self.fov}&location={lat},{lng}&heading={heading}&pitch={pitch}&key={MAPS_KEY}"
            urls.append(url)
            
            heading = max(heading+90, 360)
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

    async def saveImages(self, coordinates):
        validatedCoordinates = await self.validateCoordinates(coordinates)
        self.coordinates = np.array(validatedCoordinates)

        imageUrls = await self.generateURLs(self.coordinates)
        byteDataList = await self.fetchMultiple(imageUrls)

        if len(byteDataList) != len(self.coordinates):
            print(
                f"Warning: Number of fetched images ({len(byteDataList)}) does not match number of coordinates ({len(self.coordinates)})"
            )

        self.images = byteDataList

        self.writeImages()

    def writeImages(self):
        import os
        path = "./data/valid_images/"
        csvFile = open(self.csvPath, 'a')
        os.makedirs(path, exist_ok=True)

        new_coordinates = []
        for x, y in self.coordinates:
            new_coordinates.append((x, y))
            new_coordinates.append((x, y))
        
        coord_index = 0

        print("Writing images to file.")
        for imageBytes in self.images:
            with open(os.path.join(path, str(self.cur)+".png", ), 'wb') as image:
                image.write(imageBytes)

            csvFile.write(f"{self.cur}, {new_coordinates[coord_index][0]}, {new_coordinates[coord_index][1]}\n")
            
            self.cur += 1
            coord_index += 1
