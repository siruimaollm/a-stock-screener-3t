from src.data_fetcher import load_stock_info_sqlite
meta = load_stock_info_sqlite("data/stock_data.db")
# Write to file to check encoding
with open("output/meta_sample.txt", "w", encoding="utf-8") as f:
    for code, info in list(meta.items())[:10]:
        f.write(f"{code}: {info['name']} | {info['industry']}\n")
print("Written to output/meta_sample.txt")
