import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routers import predict

app = FastAPI(title="SentinelAI")

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(predict.router)