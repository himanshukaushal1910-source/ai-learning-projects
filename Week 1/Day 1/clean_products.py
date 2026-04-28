import pandas as pd

# Step 1 — Load the CSV file into a pandas DataFrame
df = pd.read_csv("products.csv")

# Step 2 — Print the shape (rows, columns) and first 5 rows to understand the data
print("=== RAW DATA ===")
print(f"Shape: {df.shape}")
print(df.head())
print()

# Step 3 — Remove rows where description is empty, null, or only whitespace
rows_before = len(df)
df = df[df["description"].notna()]           # removes NaN / null values
df = df[df["description"].str.strip() != ""] # removes rows that are only spaces

# Step 4 — Clean the description column: lowercase and strip leading/trailing whitespace
df["description"] = df["description"].str.lower().str.strip()

# Step 5 — Print how many rows were removed during cleaning
rows_after = len(df)
rows_removed = rows_before - rows_after
print("=== CLEANING REPORT ===")
print(f"Rows before cleaning: {rows_before}")
print(f"Rows after cleaning:  {rows_after}")
print(f"Rows removed:         {rows_removed}")
print()

# Step 6 — Pull the clean descriptions out as a plain Python list
descriptions = df["description"].tolist()

# Step 7 — Print the first 3 items from the list
print("=== FIRST 3 CLEANED DESCRIPTIONS ===")
for i, text in enumerate(descriptions[:3], start=1):
    print(f"{i}. {text}")
print()

# Step 8 — Save the cleaned DataFrame to a new CSV file
df.to_csv("cleaned_products.csv", index=False)
print("Saved cleaned data to cleaned_products.csv")
