from fastapi import FastAPI
import uvicorn
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image

app = FastAPI()

model = load_model('model.h5')

@app.get('/')
def main():
    return 'Main'