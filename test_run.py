from app.file_loader import load_file
from app.ai_engine import ask_data

df = load_file("ipl_2023_dataset.csv")

answer = ask_data(df, "give me a table of total spending by each team")

print(answer)
