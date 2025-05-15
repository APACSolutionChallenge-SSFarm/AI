import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
import io
import requests
import os
from datetime import datetime
from dotenv import load_dotenv

# .env 파일에서 API 키 불러오기
load_dotenv()

# 클래스 이름
item_classes = ['carrot', 'potato']
grade_classes = ['HIGH', 'MEDIUM', 'LOW', 'UGLY']

# 전처리 함수
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 모델 불러오기 함수
def load_model(path, num_classes):
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

# 모델 로드
model_item = load_model("best_detection_model.pth", num_classes=len(item_classes))
model_grade = load_model("best_model.pth", num_classes=len(grade_classes))

# 이미지에서 품목과 등급 예측
def predict_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        # 품목 예측
        item_output = model_item(tensor)
        item_idx = torch.argmax(item_output, dim=1).item()
        item_name = item_classes[item_idx]

        # 등급 예측
        grade_output = model_grade(tensor)
        grade_idx = torch.argmax(grade_output, dim=1).item()
        grade_name = grade_classes[grade_idx]

    return item_name, grade_name

# Gemini API를 통해 시세 질의
def ask_price_gemini(item, quality):
    today = datetime.today().strftime('%Y년 %m월 %d일')
    prompt = f"Give me only the price in the form of '0000원' for {quality} grade {item} in Korea on {today}. If you don't know the real price, estimate it based on past or common sense. Do not explain anything. Just return a number like '1234원'. Never say 'unknown'."

    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    api_key = os.getenv("GEMINI_API_KEY")

    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    response = requests.post(
        f"{endpoint}?key={api_key}",
        headers=headers,
        json=data
    )

    if response.status_code == 200:
        content = response.json()
        return content["candidates"][0]["content"]["parts"][0]["text"]
    else:
        raise Exception(f"GEMINI API error: {response.status_code}, {response.text}")

# 예측 + Gemini 가격 질의 통합
def predict_image_with_price(file_bytes):
    item, grade = predict_image(file_bytes)
    price_info = ask_price_gemini(item, grade)
    return item, grade, price_info
