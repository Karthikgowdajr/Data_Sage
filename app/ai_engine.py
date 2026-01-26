import os
import base64
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI as PandasAIOpenAI

def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyze(df, query):
    llm = PandasAIOpenAI(
        api_token=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )

    sdf = SmartDataframe(
        df,
        config={
            "llm": llm,
            "save_charts": True,
            "verbose": False
        }
    )

    result = sdf.chat(query)

    outputs = []

    # ---------- CHART ----------
    if isinstance(result, dict) and result.get("type") == "chart":
        chart_path = result.get("value")

        outputs.append({
            "type": "chart",
            "value": image_to_base64(chart_path)
        })

    # ---------- TABLE ----------
    elif hasattr(result, "to_dict"):
        outputs.append({
            "type": "table",
            "value": result.to_dict()
        })

    # ---------- TEXT ----------
    else:
        outputs.append({
            "type": "text",
            "value": str(result)
        })

    return outputs