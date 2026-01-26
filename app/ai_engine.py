import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------- MAIN ENTRY ----------------
def analyze(df: pd.DataFrame, query: str):
    intent = detect_chart_intent(query)

    if intent == "pie":
        return pie_chart(df)

    if intent == "bar":
        return bar_chart(df)

    if intent == "line":
        return line_chart(df)

    if intent == "hist":
        return histogram(df)

    if intent == "scatter":
        return scatter_plot(df)

    if intent == "box":
        return box_plot(df)

    if intent == "heatmap":
        return heatmap(df)

    # ---------- TEXT FALLBACK ----------
    return text_answer(df, query)


# ---------------- INTENT DETECTOR ----------------
def detect_chart_intent(query: str):
    q = query.lower()

    if "pie" in q or "percentage" in q or "share" in q:
        return "pie"

    if "compare" in q or "bar" in q or "count" in q:
        return "bar"

    if "trend" in q or "over time" in q or "year" in q:
        return "line"

    if "distribution" in q or "hist" in q:
        return "hist"

    if "relationship" in q or "vs" in q:
        return "scatter"

    if "outlier" in q or "box" in q:
        return "box"

    if "correlation" in q or "heatmap" in q:
        return "heatmap"

    return "text"


# ---------------- CHART HELPERS ----------------
def save_plot():
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def pie_chart(df):
    col = df.select_dtypes(include="object").columns[0]
    data = df[col].value_counts()

    data.plot.pie(autopct="%1.1f%%", figsize=(6, 6))
    plt.title(f"Distribution of {col}")

    return [{"type": "chart", "value": save_plot()}]


def bar_chart(df):
    col = df.select_dtypes(include="object").columns[0]
    data = df[col].value_counts().head(10)

    sns.barplot(x=data.values, y=data.index)
    plt.title(f"Top {len(data)} categories in {col}")

    return [{"type": "chart", "value": save_plot()}]


def line_chart(df):
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) < 1:
        return text_error("No numeric column for line chart")

    df[num_cols[0]].plot()
    plt.title(f"Trend of {num_cols[0]}")

    return [{"type": "chart", "value": save_plot()}]


def histogram(df):
    num_col = df.select_dtypes(include="number").columns[0]
    df[num_col].plot.hist(bins=20)
    plt.title(f"Distribution of {num_col}")

    return [{"type": "chart", "value": save_plot()}]


def scatter_plot(df):
    nums = df.select_dtypes(include="number").columns
    if len(nums) < 2:
        return text_error("Need two numeric columns")

    plt.scatter(df[nums[0]], df[nums[1]])
    plt.xlabel(nums[0])
    plt.ylabel(nums[1])
    plt.title("Relationship Plot")

    return [{"type": "chart", "value": save_plot()}]


def box_plot(df):
    num_col = df.select_dtypes(include="number").columns[0]
    sns.boxplot(y=df[num_col])
    plt.title(f"Outliers in {num_col}")

    return [{"type": "chart", "value": save_plot()}]


def heatmap(df):
    num_df = df.select_dtypes(include="number")
    corr = num_df.corr()

    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")

    return [{"type": "chart", "value": save_plot()}]


# ---------------- TEXT FALLBACK ----------------
def text_answer(df, query):
    preview = df.head(15).to_string()

    prompt = f"""
Dataset preview:
{preview}

User question:
{query}

Answer clearly and briefly.
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return [{"type": "text", "value": res.choices[0].message.content}]


def text_error(msg):
    return [{"type": "text", "value": msg}]


