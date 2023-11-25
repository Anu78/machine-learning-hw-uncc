import asyncio
import requests
import aiohttp
from dotenv import dotenv_values

config = dotenv_values(".env")
MAPS_KEY = config["MAPS_KEY"]


class StreetViewAPI:
    def __init__(self, imageSize, fov):
        self.url = "https://maps.googleapis.com/maps/api/streetview?"
        self.url += f"size={imageSize[0]}x{imageSize[1]}&"
        self.url += f"fov={fov}&"
        self.url += f"key={MAPS_KEY}"

    async def fetchURL(self, url, session):
        """
        Gathers a list of asyncio tasks for response(). not intended for use by user.
        """
        async with session.get(url) as response:
            return await response.text

    def generateURL(self, coordinate, heading, pitch):
        """
        Returns a complete Maps API url for the given parameters.
        """
        # add coordinate, heading, pitch to url
        customURL = self.url + f"location:{coordinate[0]},{coordinate[1]}&"
        customURL += f"heading={heading}&"
        customURL += f"pitch={pitch}"
        return customURL

    async def response(self, urls):
        """
        Performs multiple requests asynchronously and returns all responses.
        """

        async with aiohttp.ClientSession() as session:
            tasks = [self.fetchURL(url, session) for url in urls]

            responses = await asyncio.gather(*tasks)

            for i, response in enumerate(responses):
                print(f"Response from URL {i + 1}: {response[:50]}...")

    def testAPI(self, heading, pitch, coordinates):
        response = requests.get(
            self.url,
            params={
                "heading": heading,
                "pitch": pitch,
                "location": f"{coordinates[0]},{coordinates[1]}",
            },
        )
        print(response.headers)

        image_bytes = response.content

        with open("./test.jpg", "wb") as file:
            file.write(image_bytes)
