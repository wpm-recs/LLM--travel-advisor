import uuid
import json
from pathlib import Path
from typing import List, Dict
# Requires langchain and langchain-text-splitters
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter


class DataPreparationModule:

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.parent_child_map: Dict[str, str] = {}

    def load_documents(self) -> List[Document]:
        """Load local Markdown documents."""
        documents = []
        data_path_obj = Path(self.data_path)

        if not data_path_obj.exists():
            raise ValueError(f"Path does not exist: {self.data_path}")

        print(f"Loading documents from: {data_path_obj.resolve()} ...")
        # Recursively find all .md files
        for md_file in data_path_obj.rglob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                parent_id = str(uuid.uuid4())

                # Calculate relative path for hierarchy analysis
                try:
                    relative_path = md_file.relative_to(data_path_obj)
                except ValueError:
                    relative_path = md_file.name

                doc = Document(
                    page_content=content,
                    metadata={
                        # Temporarily keep source and filename for _enhance_metadata
                        # They will be removed at the end of this function
                        "source": str(md_file),
                        "filename": md_file.name,
                        "parent_id": parent_id,
                        "doc_type": "parent",
                        "relative_path": str(relative_path)
                    }
                )
                documents.append(doc)
            except Exception as e:
                print(f"Error loading {md_file}: {e}")

        # 1. Enhance metadata (requires source/filename)
        for doc in documents:
            self._enhance_metadata(doc)

        # 2. Metadata Cleanup: Remove unnecessary fields to save token space
        keys_to_remove = ['filename', 'source', 'hierarchy_depth', 'hierarchy_string']
        for doc in documents:
            for key in keys_to_remove:
                doc.metadata.pop(key, None)

        self.documents = documents
        print(f"Successfully loaded {len(documents)} parent documents .")
        return documents

    def _enhance_metadata(self, doc: Document):
        """Enhance metadata: Extract place name and hierarchy tags."""
        # Note: We still access 'source' here before it is deleted
        file_path = Path(doc.metadata.get('source', ''))

        # Extract place name (filename without extension)
        doc.metadata['place_name'] = file_path.stem

        # Simple keyword tagging
        content_lower = doc.page_content.lower()
        tags = []
        if "visa" in content_lower: tags.append("policy")
        if "currency" in content_lower: tags.append("finance")
        doc.metadata['tags'] = tags

    def chunk_documents(self) -> List[Document]:
        """Structured chunking based on Markdown headers."""
        if not self.documents:
            raise ValueError("Please load documents first.")

        print("Chunking documents...")

        # Define Wikivoyage standard header structure
        headers_to_split_on = [
            ("#", "Title"),
            ("##", "Section"),  # e.g., Eat, Sleep, See
            ("###", "Sub_Section"),  # e.g., Budget, or specific area
            ("####", "Item_Name")  # e.g., Specific Restaurant/Hotel Name
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )

        # Intent Mapping: Map Wikivoyage headers to retrieval intents
        section_intent_map = {
            "Understand": "History/Background",
            "Get in": "Transport/Arrival",
            "Get around": "Transport/Internal",
            "See": "Attractions/Sightseeing",
            "Do": "Activities/Experience",
            "Buy": "Shopping",
            "Eat": "Food/Dining",
            "Drink": "Nightlife/Bars",
            "Sleep": "Accommodation/Hotels",
            "Connect": "Communication/Internet",
            "Stay safe": "Safety/Security",
            "Go next": "Nearby/Next Destination",
            "Cope": "Services/Practicalities",
            "Respect": "Etiquette/Culture",
            "Stay healthy": "Healthcare/Medical"
        }

        all_chunks = []

        for doc in self.documents:
            md_chunks = markdown_splitter.split_text(doc.page_content)
            parent_id = doc.metadata["parent_id"]

            for i, chunk in enumerate(md_chunks):
                child_id = str(uuid.uuid4())

                # Inherit parent metadata
                chunk.metadata.update(doc.metadata)

                # Add child specific metadata
                chunk.metadata.update({
                    "chunk_id": child_id,
                    "parent_id": parent_id,
                    "doc_type": "child",
                    "chunk_index": i
                })

                # Process Section Mapping
                current_section = chunk.metadata.get("Section", "")
                chunk.metadata['category'] = section_intent_map.get(current_section, "Others")


                self.parent_child_map[child_id] = parent_id

            all_chunks.extend(md_chunks)

        self.chunks = all_chunks
        print(f"Successfully split into {len(all_chunks)} chunks.")
        return all_chunks

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """
        Retrieve corresponding parent documents based on child chunks (Smart Deduplication).
        Logic:
        1. Count occurrences of each parent_id in child_chunks (Relevance Score).
        2. Retrieve full parent document objects from memory using parent_id.
        3. Return parent documents sorted by relevance.
        """
        if not self.documents:
            print("Warning: Parent document list is empty, cannot perform retrieval.")
            return []

        # 1. Collect relevance scores
        parent_relevance = {}
        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1

        # 2. Find parent document objects
        needed_ids = set(parent_relevance.keys())
        found_docs_map = {}

        for doc in self.documents:
            pid = doc.metadata.get("parent_id")
            if pid in needed_ids:
                found_docs_map[pid] = doc
                if len(found_docs_map) == len(needed_ids):
                    break

        # 3. Sort by relevance and return unique list
        sorted_parent_ids = sorted(
            parent_relevance.keys(),
            key=lambda x: parent_relevance[x],
            reverse=True
        )

        final_parent_docs = []
        for parent_id in sorted_parent_ids:
            if parent_id in found_docs_map:
                final_parent_docs.append(found_docs_map[parent_id])

        return final_parent_docs


# --- Execution Example & Export ---
# if __name__ == "__main__":
#     # Configuration
#     INPUT_PATH = "../wikivoyage_sg"
#     OUTPUT_FILE = "all_chunks_output.txt"
#
#     processor = DataPreparationModule(INPUT_PATH)
#
#     try:
#         # 1. Load
#         processor.load_documents()
#
#         # 2. Chunk
#         chunks = processor.chunk_documents()
#
#         # 3. Demo: Smart Parent Retrieval
#         if len(chunks) > 5:
#             print("\n--- Testing Smart Deduplication (get_parent_documents) ---")
#             simulated_retrieved_chunks = [chunks[0], chunks[300], chunks[1]]
#             parent_docs = processor.get_parent_documents(simulated_retrieved_chunks)
#             print(f"Input Chunk Count: {len(simulated_retrieved_chunks)}")
#             print(f"Retrieved Parent Docs: {len(parent_docs)}")
#             if parent_docs:
#                 print(f"Most Relevant Parent: {parent_docs[0].metadata.get('place_name')}")
#
#         # 4. Export to TXT for verification
#         if chunks:
#             print(f"\nWriting content to {OUTPUT_FILE} ...")
#             with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
#                 for i, chunk in enumerate(chunks):
#                     f.write("=" * 80 + "\n")
#                     f.write(f"CHUNK NO: {i} | ID: {chunk.metadata.get('chunk_id')}\n")
#
#                     f.write("-" * 30 + " METADATA " + "-" * 30 + "\n")
#                     f.write(json.dumps(chunk.metadata, ensure_ascii=False, indent=2))
#                     f.write("\n")
#
#                     f.write("-" * 30 + " CONTENT " + "-" * 31 + "\n")
#                     f.write(chunk.page_content)
#                     f.write("\n\n")
#
#             print(
#                 f"✅ Export complete! Please check {OUTPUT_FILE} to confirm 'sub_category', 'poi_name', and 'filename' are removed.")
#         else:
#             print("⚠️ No chunks generated, nothing to export.")
#
#     except ValueError as e:
#         print(f"❌ Error: {e}")