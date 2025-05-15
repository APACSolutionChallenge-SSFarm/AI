from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from predict_models import predict_image_with_price
import os
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# FastAPI 앱 생성
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 파일 읽기
        contents = await file.read()

        # 예측 + Gemini API 질의
        item, grade, price_info = predict_image_with_price(contents)

        return JSONResponse(content={
            "item": item,
            "quality": grade,
            "gemini_price_estimate": price_info
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
