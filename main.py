from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from predict_models import predict_image_with_price
import requests
import os
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# 외부 서버 주소
PREDICT_RESULT_URL = "https://29e0-180-71-14-251.ngrok-free.app/api/ai/predict-result"

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 파일 내용 읽기
        contents = await file.read()

        # 예측 수행
        item, grade, price_info = predict_image_with_price(contents)

        result = {
            "name": item,
            "qualityGrade": grade,
            "recommendedPrice": price_info
        }

        # 외부 서버에 결과 POST
        try:
            response = requests.post(PREDICT_RESULT_URL, json=result, timeout=5)
            status_code = response.status_code
            response_text = response.text
        except Exception as e:
            status_code = 500
            response_text = f"외부 서버 전송 실패: {str(e)}"

        return JSONResponse(content={
            "localResult": result,
            "externalResponseStatus": status_code,
            "externalResponse": response_text
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
