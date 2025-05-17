from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


# --------- Config --------- #
CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 200
ENCODING = "utf-8"  # encoding for text files
# -------------------------- #


def _get_loader(path: Path):
    """Return an appropriate LangChain loader for *path*."""
    if path.suffix.lower() == ".pdf":
        return PyPDFLoader(str(path))
    if path.suffix.lower() == ".txt":
        return TextLoader(str(path), encoding=ENCODING)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def load_documents(data_dir: Path) -> List[Document]:
    """
    Load all supported documents from *data_dir* (non-recursively).

    Returns a flat list of LangChain `Document`s.
    """
    files = [p for p in data_dir.iterdir() if p.suffix.lower() in {".pdf", ".txt"}]
    docs: List[Document] = []

    for path in files:
        loader = _get_loader(path)
        docs.extend(loader.load())

    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    """Chunk documents for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )
    return splitter.split_documents(docs)