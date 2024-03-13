import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import vader

app = FastAPI()
app.state.model = vader

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
@app.get("/predict")
def predict(
        sentence : str
    ):
    """
    Make a single sentence prediction.
    """

    y_pred = pd.DataFrame(
        {'sentence' : sentence},
        index=[0]
    )

    result = app.state.model(y_pred)
    return result

@app.get("/")
def root():
    return {'greeting' : 'Hello'}
