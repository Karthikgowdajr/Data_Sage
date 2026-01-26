import os
import uuid
import matplotlib.pyplot as plt
from pandasai import SmartDataframe
from openai import OpenAI

def detect_chart_type(question: str):
    q = question.lower()

    if "pie" in q:
        return "pie"
    if "bar" in q:
        return "bar"
    if "line" in q or "trend" in q:
        return "line"
    if "scatter" in q:
        return "scatter"
    if "hist" in q or "distribution" in q:
        return "hist"
    return None


def analyze(df, question):
    llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    chart_type = detect_chart_type(question)

    sdf = SmartDataframe(
        df,
        config={
            "llm": llm,
            "verbose": False,
            "save_charts": True,
            "save_charts_path": "exports/charts",
        },
    )

    outputs = []

    # -------- FORCE CHART GENERATION --------
    if chart_type:
        forced_prompt = f"""
        Create a {chart_type} chart to answer this question:
        "{question}"

        IMPORTANT:
        - Always generate a new chart
        - Do NOT reuse previous charts
        - Save the chart as an image
        """

        result = sdf.chat(forced_prompt)

        # Find latest chart
        chart_files = sorted(
            os.listdir("exports/charts"),
            key=lambda x: os.path.getmtime(os.path.join("exports/charts", x)),
        )

        if chart_files:
            outputs.append({
                "type": "chart",
                "value": os.path.join("exports/charts", chart_files[-1])
            })
        else:
            outputs.append({
                "type": "text",
                "value": str(result)
            })

        return outputs

    # -------- TEXT / TABLE FALLBACK --------
    result = sdf.chat(question)

    if isinstance(result, str):
        outputs.append({"type": "text", "value": result})
    else:
        outputs.append({"type": "text", "value": str(result)})

    return outputs
