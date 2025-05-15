import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
import io

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

# 추론 함수
def predict_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    # 품목 예측
    with torch.no_grad():
        item_output = model_item(tensor)
        item_idx = torch.argmax(item_output, dim=1).item()
        item_name = item_classes[item_idx]

        # 등급 예측
        grade_output = model_grade(tensor)
        grade_idx = torch.argmax(grade_output, dim=1).item()
        grade_name = grade_classes[grade_idx]

    return item_name, grade_name
