import pandas as pd

def load_file(file_path: str):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)

    elif file_path.endswith(".xlsx"):
        return pd.read_excel(file_path)

    elif file_path.endswith(".txt"):
        return pd.read_csv(file_path, delimiter="\t")

    else:
        raise ValueError("Unsupported file format")
