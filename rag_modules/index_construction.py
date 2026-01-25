from pathlib import Path
from typing import List

# Langchain imports for Index Construction
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class IndexConstructionModule:

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5",
                 index_save_path: str = "./vector_index"):
        # Note: Default model is English optimized (bge-small-en-v1.5) for Wikivoyage data.
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.embeddings = None
        self.vectorstore = None
        self.setup_embeddings()

    def setup_embeddings(self):
        """Initialize the embedding model"""
        print(f"Loading embedding model: {self.model_name}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("Embedding model loaded successfully.")

    def build_vector_index(self, chunks: List[Document]) -> FAISS:
        """Build the vector index"""
        if not chunks:
            raise ValueError("The list of document chunks cannot be empty.")

        print(f"Building FAISS index with {len(chunks)} chunks...")
        # Extract text content and metadata
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Build FAISS vector index
        self.vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )

        print("FAISS index built successfully.")
        return self.vectorstore

    def save_index(self):
        """Save the vector index to the configured path"""
        if not self.vectorstore:
            raise ValueError("Please build the vector index first.")

        # Ensure the save directory exists
        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)

        self.vectorstore.save_local(self.index_save_path)
        print(f"Vector index saved to: {self.index_save_path}")

    def load_index(self):
        """Load the vector index from the configured path"""
        if not self.embeddings:
            self.setup_embeddings()

        if not Path(self.index_save_path).exists():
            return None

        self.vectorstore = FAISS.load_local(
            self.index_save_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        return self.vectorstore