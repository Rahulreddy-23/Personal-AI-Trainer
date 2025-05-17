import os
import sys
from pathlib import Path

os.environ["HF_HOME"] = "/tmp/huggingface"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

DB_DIR = Path(__file__).resolve().parent.parent / "db"

def build_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
        collection_name="personal-docs",
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = OllamaLLM(model="llama3")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    return qa_chain

def chat_loop(chain):
    print("\n Ask anything about your documents. Type 'exit' to quit.\n")
    while True:
        try:
            query = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        result = chain.invoke({"query": query})
        print("\nAnswer:\n")
        print(result["result"])
        print("-" * 40)

def main() -> None:
    if not DB_DIR.exists():
        print(" Vector store not found. Run `python src/ingest.py` first.", file=sys.stderr)
        sys.exit(1)

    chain = build_chain()
    chat_loop(chain)

if __name__ == "__main__":
    main()