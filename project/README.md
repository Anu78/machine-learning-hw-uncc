## GeoGuessr AI 
A machine learning model to predict where a picture was taken in the US within 50 miles, consistently. 

### Scope
To begin, this model will learn to predict locations in the US. If that is a success, the model will be expanded to other countries.

### Process
1. gather training data from the google street view api 
2. 

### Roadmap 
- [ ] Set up google cloud project and get api key
- [ ] Write api implementation (fetch multiple images at a time)
- [ ] Figure out how to get a grid of lat/long pairs from shapefile for NC
- [ ] Test model architecture on NC

### Questions I don't know the answer to 
1. How many input pictures should the model receive per prediction? Just one, or three? 
2. Should we pick locations randomly within each state or follow a grid system? 
3. Should the amount of locations per state be the same regardless of the area of the state? Will having a larger percentage (5%) of the training data in one state reduce model performance? 
4. Should the model return a probability array of 50 states (classification) or return a lat/long pair (regression)? 
5. Can the loss function just be the difference in lat/long (MSE loss) or do I need to have a more accurate formula? 

### The answers 
1. One picture
2. Grid system for now. I don't see an issue with this at the moment. This way, we are guaranteed to cover the entire state.
3. Locations per state is currently based on the state's area and then multiplied by a factor between 1.3 and 0.7 depending on how diverse the landscape of the state is compared to all other states. 
4. This will be a regression model. Hawaii being so far away from the other states might become a problem for our loss function later. 
5. (predicted lat - real lat) ** 2 + (predicted long - real long) ** 2. Simple MSE. Might have to adjust this in case of Hawaii.

