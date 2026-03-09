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

    try:
        result = sdf.chat(query)

    except Exception as e:
        error_message = str(e).lower()

        # ---------- QUOTA ERROR ----------
        if "quota" in error_message or "insufficient_quota" in error_message:
            return [{
                "type": "text",
                "value": "⚠️ Data Sage AI usage limit reached. Please try again later."
            }]

        # ---------- RATE LIMIT ----------
        if "rate limit" in error_message or "429" in error_message:
            return [{
                "type": "text",
                "value": "⚠️ Too many requests right now. Please try again in a few minutes."
            }]

        # ---------- GENERAL ERROR ----------
        return [{
            "type": "text",
            "value": "⚠️ Something went wrong while analyzing the data."
        }]

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