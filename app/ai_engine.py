import os
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI as PandasAIOpenAI

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
        outputs.append({
            "type": "chart",
            "value": result.get("value")  # path to image
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
