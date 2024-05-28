from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.responses import JSONResponse
from io import BytesIO
from PIL import Image
import os
import base64

from model import load_model
from utils import ImagePayload, preprocess_image

app = FastAPI()

model = None

prefix = '/opt/ml/model'

model_path = os.path.join(prefix, 'model.pth')

@app.on_event("startup")
async def startup_event():
    global model
    model = load_model(model_path)

@app.get("/ping")
async def ping():
    health = model is not None
    status = 200 if health else 404
    return JSONResponse(content={'status': 'ok' if health else 'error'}, status_code=status)

@app.post("/invocations")
async def invocations(payload: ImagePayload):
    try:
        image_data = base64.b64decode(payload.image)
        image = Image.open(BytesIO(image_data))
        image_tensor = preprocess_image(image)
        prediction = model(image_tensor).argmax(dim=1).item()
        return JSONResponse(content={'prediction': prediction})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
