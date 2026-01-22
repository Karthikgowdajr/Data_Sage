from fastapi import FastAPI, UploadFile, File, Form
import pandas as pd
from app.ai_engine import analyze

app = FastAPI(title="Data Sage AI")

@app.post("/analyze")
async def analyze_data(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    filename = file.filename.lower()

    # -------- FILE HANDLING --------
    if filename.endswith(".csv"):
        df = pd.read_csv(file.file)

    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(file.file)

    elif filename.endswith(".txt"):
        df = pd.read_csv(file.file, delimiter="\t")

    else:
        return {
            "error": "Unsupported file type. Please upload CSV, Excel, or TXT."
        }

    # -------- AI ANALYSIS --------
    result = analyze(df, question)

    # -------- STANDARD RESPONSE --------
    return {
        "answer": result
    }

