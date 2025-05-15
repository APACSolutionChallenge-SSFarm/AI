from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from predict_models import predict_image

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        item, grade = predict_image(contents)
        return JSONResponse(content={
            "item": item,
            "quality": grade
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
