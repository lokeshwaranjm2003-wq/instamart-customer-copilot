
from pathlib import Path
import sqlite3
import pandas as pd

db_path = Path("data/instamart.db")
sql_path = Path("sql/queries.sql")

sql_text = sql_path.read_text(encoding="utf-8")

# Split the SQL file into individual queries
queries = [
    query.strip()
    for query in sql_text.split(";")
    if query.strip()
]

with sqlite3.connect(db_path) as conn:
    for i, query in enumerate(queries, start=1):
        result = pd.read_sql_query(query, conn)
        print(f"\nQ{i}: {len(result)} rows")
        print(result.head(5).to_string(index=False))

print("\nSQL query execution completed.")
