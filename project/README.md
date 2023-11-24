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

Number of Input Pictures per Prediction:

    Pros of Using One Picture: Simpler model, less computational load, quicker predictions.
    Pros of Using Multiple Pictures: More context, potentially more accurate predictions, especially in areas where single images might be ambiguous.
    Recommendation: Start with one picture to establish a baseline. If accuracy is not satisfactory, consider experimenting with multiple images.
    Location Selection: Random vs. Grid System:

Random Selection: Ensures a diverse set of images, reducing the chance of bias towards certain types of locations.
Grid System: Provides a structured and even coverage, ensuring all regions are represented.
Recommendation: Use a hybrid approach. Start with a grid system for even coverage, and then introduce randomness within each grid cell to capture diversity.
Locations per State and Training Data Distribution:

Equal Distribution by State: Might not represent the actual diversity and size differences between states.
Proportional Distribution: Reflects the geographical and cultural diversity more accurately.
Recommendation: Consider a proportional approach based on factors like state area, population density, and diversity of landscapes. This helps in learning features relevant to larger or more diverse states more effectively.
Model Output: Classification vs. Regression:

Classification (State Probability Array): Easier to implement, but less precise.
Regression (Lat/Long Pair): Offers more precise location prediction, but is more complex.
Recommendation: Given your goal of predicting locations within 50 miles, a regression approach seems more appropriate. However, start with a classification model to establish a baseline and then transition to regression for finer precision.
Loss Function:

MSE Loss for Lat/Long: Simple and directly aligns with your objective.
Geospatially-Aware Loss Function: Accounts for the actual distance between predicted and true locations, considering the Earth's curvature.
Recommendation: Begin with a simple MSE loss for ease of implementation. If the results are not satisfactory, consider a more complex, geospatially-aware loss function like the Haversine formula or a similar approach.
