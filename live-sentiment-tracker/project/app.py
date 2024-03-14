from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import vader, vader_scores
from preprocess import SlangTranslation

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?sentence=i%20was%20sad%20but%20now%20im%20happy
@app.get("/predict")
def predict(sentence : str):

    """
    Make a single sentence prediction.
    """
    sentence = SlangTranslation(sentence).apply_translator(sentence)
    result = vader(sentence)
    return result

# http://127.0.0.1:8000/scores?sentence=i%20was%20sad%20but%20now%20im%20happy
@app.get("/scores")
def scores(sentence: str):

    sentence = SlangTranslation(sentence).apply_translator(sentence)
    result = vader_scores(sentence)
    return result

@app.get("/")
def root():
    return {'its' : 'working'}
