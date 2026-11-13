import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def retreive_parse_data_from_excel(file_path: str) -> list[str]:
    df = pd.read_excel(file_path,  parse_dates=["Product Expiry Date"])

    # Convert dataframe to list of dictionaries
    data = df.to_dict(orient="records")

    parsed_data = [
        (
            f"Product Name: {item['Product Name']}\n"
            f"Product Description: {item['Product Description']}\n"
            f"Product Quantity: {item['Product Quantity']}\n"
            f"Product Expiry Date: {item['Product Expiry Date'].strftime('%Y-%m-%d') if pd.notnull(item['Product Expiry Date']) else 'N/A'}\n\n"
        )
        for item in data
    ]
    return parsed_data

def build_vector_db_from_texts(texts: list[str], model) -> None:
    try:
        vectordb = Chroma.from_texts(
            texts=texts,
            embedding=model,
            collection_name="Inventory",
            persist_directory="./data/vector_db/chroma_data"
        )
        print("Vector DB built and persisted successfully.")
    except Exception as e:
        print(f"Error building vector DB: {e}")

if __name__ == "__main__":
    excel_file_path = "./data/inventory_data.xlsx"
    parsed_texts = retreive_parse_data_from_excel(excel_file_path)

    embeddings = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 16, "normalize_embeddings": True},
        query_encode_kwargs={
            "prompt": "Instruct: Given an inventory management query, retrieve the relevant product records, descriptions, quantities, pricing and expiry dates details that match the request\nQuery: "
        },
        show_progress=True
    )
    #build_vector_db_from_texts(parsed_texts, embeddings)
    query = "Dairy products"
    db3 = Chroma(collection_name='Inventory', persist_directory="./data/vector_db/chroma_data", embedding_function=embeddings)
    docs = db3.similarity_search(query)
    print(docs[0].page_content)