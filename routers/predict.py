from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from services.inference import predict_text

router = APIRouter()

templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": None
        }
    )


@router.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):

    result = predict_text(text)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "safe_probability": result["safe_probability"],
            "toxic_probability": result["toxic_probability"],
            "text": text
        }
    )