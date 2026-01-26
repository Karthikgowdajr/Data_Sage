from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import pandas as pd
from app.ai_engine import analyze

app = FastAPI(title="Data Sage AI")

@app.post("/analyze")
async def analyze_data(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    try:
        filename = file.filename.lower()

        # -------- FILE HANDLING --------
        if filename.endswith(".csv"):
            df = pd.read_csv(file.file)

        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            file.file.seek(0)
            df = pd.read_excel(file.file)

        elif filename.endswith(".txt"):
            file.file.seek(0)
            df = pd.read_csv(file.file, delimiter="\t")

        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Upload CSV, Excel, or TXT."
            )

        # Safety check
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        # -------- AI ANALYSIS --------
        result = analyze(df, question)

        return {
            "question": question,
            "answer": result
        }

    except Exception as e:
        # 🔥 THIS IS THE MOST IMPORTANT PART
        print("❌ ANALYZE ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get("/")
def root():
    return {
        "status": "Data Sage backend is live 🚀"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }
