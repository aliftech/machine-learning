import tensorflow as tf
from fastapi import FastAPI, UploadFile
import uvicorn


MODEL = tf.keras.models.load_model('trained_model/')

app = FastAPI()

@app.get('/')
async def index():
    return {"Message": "This is Index"}

@app.post('/predict')
async def predic(file: UploadFile):
    image = file.filename
    classes = MODEL.predict(image)
    if classes==0:
        return "Kamar rapi"
    else:
        return "Kamar berantakan!"