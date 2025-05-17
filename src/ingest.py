import os
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm

from utils import load_documents, split_documents

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_DIR = Path(__file__).resolve().parent.parent / "db"

os.environ["HF_HOME"] = "/tmp/huggingface"

def main() -> None:
    print(f" Loading documents from {DATA_DIR} ...")
    raw_docs = load_documents(DATA_DIR)
    print(f"   ➜ Loaded {len(raw_docs)} documents")

    print(" Splitting into chunks ...")
    docs = split_documents(raw_docs)
    print(f"   ➜ {len(docs)} chunks ready for embedding")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(" Embedding & persisting to Chroma ...")
    _ = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=str(DB_DIR),
        collection_name="personal-docs",
    )
    print(f"Done! Vector store saved to {DB_DIR}")

if __name__ == "__main__":
    main()