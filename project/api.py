import asyncio
import aiohttp


class StreetViewAPI:
    def __init__(self, imageSize, fov, apiKey, signature):
        self.url = "https://maps.googleapis.com/maps/api/streetview?"
        
        self.url += f"size={imageSize[0]}x{imageSize[1]}&"
        self.url += f"fov={fov}&"
        self.url += f"key={apiKey}&"
        self.url += f"signature={signature}&"


    async def fetchURL(self, url, session):
        """
        Gathers a list of asyncio tasks for response(). not intended for use by user.
        """
        async with session.get(url) as response:
            return await response.text()

    def generateURLs(self, coordinate, heading, pitch):
        """
        Returns a complete Maps API url for the given parameters.
        """
        # add coordinate, heading, pitch to url
        customURL = self.url + f"location:{coordinate[0]},{coordinate[1]}&"
        customURL += f"heading={heading}&"
        customURL += f"pitch={pitch}"
        return customURL

    async def response(self, batch_size=50):
        """
        Performs multiple requests asynchronously and returns all responses. 
        """
        urls = self.generateURLs()

        async with aiohttp.ClientSession() as session:
            tasks = [self.fetchURL(url, session) for url in urls]

            responses = await asyncio.gather(*tasks)

            for i, response in enumerate(responses):
                print(f"Response from URL {i + 1}: {response[:50]}...")
