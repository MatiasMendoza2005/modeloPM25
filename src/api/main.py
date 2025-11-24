from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# imports relativos correctos
from .validation import PM25Request
from .inference import predictor

app = FastAPI(title="PM2.5 Prediction API – TCN v2")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "model": "TCN-v2 ready"}

@app.post("/predict/pm25")
def predict_pm25(request: PM25Request):
    try:
        pred_value = predictor.predict(request.dict())
        return {"pm25_prediction": pred_value}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
