from bs4 import BeautifulSoup
import re
import requests
import os
import zipfile
import tempfile
import pandas as pd
from helpers import Shapefile

url = "https://www2.census.gov/geo/tiger/TIGER_RD18/LAYER/ROADS/"
query_params = {"downloadformat": "zip"}
digit_pattern = r"\d+"
fips_csv = "./data/fips.csv"

df = pd.read_csv(fips_csv)


def combine(stateCode, tempPath):
    stateShapefile = Shapefile()

    for dir in os.listdir(tempPath):
        for file in os.listdir(os.path.join(tempPath, dir)):
            if file.endswith(".shp"):
                stateShapefile.combine(os.path.join(tempPath, dir, file))

    print(f"writing {stateCode} to files.")
    stateShapefile.write(os.path.join("./data/shapefiles", stateCode, "Shapefile.shp"))


def main(fips_start):
    tempdir = tempfile.TemporaryDirectory()
    tempPath = tempdir.name

    response = requests.get(url).text

    soup = BeautifulSoup(response, "html.parser")
    currentState = None
    threshold = False
    for link in soup.find_all("a", attrs={"href": re.compile("^tl_rd22")}):
        fips = int(re.findall(digit_pattern, link.get("href"))[1])
        if not threshold:
            if fips >= fips_start:
                threshold = True
            else:
                continue
        if fips > 56045:
            break
        location = df[df["fips"] == fips]

        if location.empty:
            print(f"no matching county found for fips code {fips}")
            continue

        state = location.iloc[0]["state"]

        print(f"on {fips}", end="\r")
        if currentState is not None and state != currentState:
            print(f"now processing {state}")
            combine(currentState, tempPath)

            tempdir.cleanup()  # reset for next batch of files

            tempdir = tempfile.TemporaryDirectory()
            tempPath = tempdir.name

        currentState = state

        download = url + link.get("href")
        file = requests.get(download, params=query_params)

        if file.status_code == 200:
            zipPath = os.path.join(tempPath, str(fips))
            with open(zipPath, "wb") as zip_file:
                zip_file.write(file.content)

            extractPath = os.path.join(tempPath, "extracted_" + str(fips))
            os.makedirs(extractPath, exist_ok=True)

            with zipfile.ZipFile(zipPath, "r") as zipref:
                zipref.extractall(extractPath)

            os.remove(zipPath)
        else:
            print(f"Download of {download} failed.")

    if currentState is not None:
        combine(currentState, tempPath)
        tempdir.cleanup()


if __name__ == "__main__":
    main(56001)
