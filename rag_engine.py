"""PDF indexing and grounded context retrieval with BioBERT and ChromaDB."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_GUIDELINES_DIR = Path("data/guidelines")
DEFAULT_CHROMA_DIR = Path("artifacts/chroma_db")
DEFAULT_COLLECTION_NAME = "clinical_guidelines"
DEFAULT_EMBEDDING_MODEL = "dmis-lab/biobert-base-cased-v1.1"


def create_embeddings(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    *,
    model_kwargs: dict | None = None,
    encode_kwargs: dict | None = None,
) -> HuggingFaceEmbeddings:
    """Create the requested Hugging Face embedding model.

    BioBERT is used as requested. It is a biomedical language model rather
    than a sentence-embedding-tuned model, so retrieval quality should be
    evaluated on the project's own clinical queries and documents.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs or {"device": "cpu"},
        encode_kwargs=encode_kwargs or {"normalize_embeddings": True},
    )


def load_guideline_pdfs(guidelines_dir: str | Path = DEFAULT_GUIDELINES_DIR) -> list[Document]:
    """Load PDF pages recursively from the provided guidelines directory."""
    directory = Path(guidelines_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Guidelines directory does not exist: {directory}")
    pdf_paths = list(directory.rglob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF guideline files found in: {directory}")
    return PyPDFDirectoryLoader(str(directory), recursive=True).load()


def split_guideline_documents(
    documents: Sequence[Document],
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """Split loaded page documents into overlapping retrieval chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(list(documents))


class MedicalRAG:
    """Persistent Chroma index for locally supplied medical guideline PDFs."""

    def __init__(
        self,
        *,
        persist_directory: str | Path = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        device: str = "cpu",
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embeddings = create_embeddings(
            embedding_model,
            model_kwargs={"device": device},
        )
        self.vectorstore: Chroma | None = None

    def index_pdfs(
        self,
        guidelines_dir: str | Path = DEFAULT_GUIDELINES_DIR,
        *,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        reset: bool = False,
    ) -> int:
        """Index guideline PDF chunks into the persistent Chroma collection.

        Returns the number of chunks indexed. Set ``reset=True`` to replace
        the existing collection contents with a fresh index.
        """
        documents = load_guideline_pdfs(guidelines_dir)
        chunks = split_guideline_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not chunks:
            raise ValueError("The guideline PDFs contained no indexable text")

        self.persist_directory.mkdir(parents=True, exist_ok=True)
        if reset:
            # Chroma delete_collection clears this named collection only.
            Chroma(
                collection_name=self.collection_name,
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings,
            ).delete_collection()

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=str(self.persist_directory),
        )
        return len(chunks)

    def open_index(self) -> Chroma:
        """Open an existing persistent collection for retrieval."""
        if not self.persist_directory.is_dir():
            raise FileNotFoundError(f"Chroma index directory does not exist: {self.persist_directory}")
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=str(self.persist_directory),
            embedding_function=self.embeddings,
        )
        return self.vectorstore

    def search(
        self,
        disease_label: str,
        *,
        top_k: int = 4,
    ) -> list[Document]:
        """Return the top-K clinical context chunks for a disease label."""
        query = disease_label.strip()
        if not query:
            raise ValueError("disease_label must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        vectorstore = self.vectorstore or self.open_index()
        return vectorstore.similarity_search(query, k=top_k)

    def search_with_scores(
        self,
        disease_label: str,
        *,
        top_k: int = 4,
    ) -> list[tuple[Document, float]]:
        """Return matching documents and Chroma's distance scores."""
        query = disease_label.strip()
        if not query:
            raise ValueError("disease_label must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        vectorstore = self.vectorstore or self.open_index()
        return vectorstore.similarity_search_with_score(query, k=top_k)
