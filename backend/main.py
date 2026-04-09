from fastapi import FastAPI
from optimizer import run_optimization
from utils import analyze_system
import torch
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all for now
    allow_credentials=False,  # 🔥 IMPORTANT CHANGE
    allow_methods=["*"],
    allow_headers=["*"],
)




# ✅ Request Body Model
class RequestData(BaseModel):
    data_path: str
    ram: int
    cpu: int
    gpu: bool


@app.get("/")
def home():
    return {"message": "AutoML CV Backend Running"}


# ✅ FIXED ENDPOINT
@app.post("/optimize")
def optimize(data: RequestData):
    device = "cuda" if data.gpu and torch.cuda.is_available() else "cpu"

    system_info = analyze_system(data.ram, data.cpu, data.gpu)
    best = run_optimization(data.data_path, device)

    return {
        "system_recommendation": system_info,
        "best_hyperparameters": best["params"],
        "accuracy": best["accuracy"],
        "trials": best["trials"]   # 🔥 new
}
