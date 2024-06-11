from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from catboost import CatBoostClassifier

# FastAPI의 BaseModel을 상속받아 생성
class UserInput(BaseModel):
    age: float
    hypertension: int
    heart_disease: int
    avg_glucose_level: float
    bmi: float
    smoking_status: int

# FastAPI 앱 생성 - 서버의 인스턴스
app = FastAPI()

# 저장된 모델 로드
model = CatBoostClassifier()
model.load_model("catboost_model.cbm")

# 예측 결과는 JSON 형식으로 반환
@app.post("/predict")
def predict_stroke(input_data: UserInput):
    data = np.array([[
        input_data.age,
        input_data.hypertension,
        input_data.heart_disease,
        input_data.avg_glucose_level,
        input_data.bmi,
        input_data.smoking_status
    ]])
    prediction = model.predict(data)
    prediction_probability = model.predict_proba(data)
    return {"prediction": int(prediction[0]), "probability" : int(prediction_probability[0])}

