import os
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze(df: pd.DataFrame, query: str):
    query_lower = query.lower()

    # --------- CHART INTENT DETECTION ---------
    if "pie chart" in query_lower:
        return generate_pie_chart(df)

    # --------- TEXT FALLBACK ---------
    preview = df.head(20).to_string()

    prompt = f"""
Dataset preview:
{preview}

User question:
{query}

Answer clearly.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return [{
        "type": "text",
        "value": response.choices[0].message.content
    }]


def generate_pie_chart(df: pd.DataFrame):
    if "2023 Squad" not in df.columns:
        return [{
            "type": "text",
            "value": "Column '2023 Squad' not found in dataset."
        }]

    team_counts = df["2023 Squad"].value_counts()

    plt.figure(figsize=(6, 6))
    team_counts.plot.pie(autopct="%1.1f%%", startangle=90)
    plt.title("Players Bought by Team (2023)")
    plt.ylabel("")

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)

    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return [{
        "type": "chart",
        "value": image_base64
    }]

