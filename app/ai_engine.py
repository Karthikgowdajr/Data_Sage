import os
import pandas as pd
from pandasai import SmartDataframe
from openai import OpenAI

def analyze(df, query):
    llm = OpenAI(api_token=os.getenv("OPENAI_API_KEY"))
    sdf = SmartDataframe(df, config={"llm": llm})

    raw_result = sdf.chat(query)

    outputs = []

    # ---------- HANDLE CHART ----------
    if isinstance(raw_result, dict) and raw_result.get("type") == "chart":
        outputs.append({
            "type": "chart",
            "value": raw_result.get("value")
        })

    # ---------- HANDLE TEXT ----------
    elif isinstance(raw_result, str):
        outputs.append({
            "type": "text",
            "value": raw_result
        })

    # ---------- FALLBACK ----------
    else:
        outputs.append({
            "type": "text",
            "value": str(raw_result)
        })

    return outputs
