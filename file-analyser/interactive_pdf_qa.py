"""
Enhanced PDF QA System with Cosine Similarity and Relaxed Thresholds
"""

import os
import hashlib
from typing import List, Dict, Tuple
from datetime import datetime
import pickle
import re
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# PDF extraction libraries
import fitz  # PyMuPDF
import pdfplumber
import PyPDF2

# Optional OCR
try:
    import pytesseract
    from pdf2image import convert_from_path

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

from sentence_transformers import SentenceTransformer, util
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import shutil


class EnhancedLocalPDFQASystem:
    """
    Enhanced Local PDF QA System with cosine similarity
    """

    EMBEDDING_DIM = 768  # For all-mpnet-base-v2

    def __init__(self, storage_dir="pdf_storage", model_name="google/flan-t5-base"):
        print("Initializing Enhanced Local PDF QA System...")

        self.storage_dir = storage_dir
        self.blob_dir = os.path.join(storage_dir, "blobs")
        self.index_dir = os.path.join(storage_dir, "indexes")
        self.meta_dir = os.path.join(storage_dir, "metadata")

        for dir_path in [self.blob_dir, self.index_dir, self.meta_dir]:
            os.makedirs(dir_path, exist_ok=True)

        print("Loading embedding model...")
        self.embedder = SentenceTransformer('all-mpnet-base-v2')

        print(f"Loading LLM model ({model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.qa_pipeline = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=512,
            do_sample=True,
            temperature=0.3,
            device=-1
        )

        self.documents = []
        self.document_chunks = []
        self.chunk_metadata = []
        self.index = None

        self._load_existing_data()

        print("System ready!\n")

    def _get_file_hash(self, file_path: str) -> str:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    def _load_existing_data(self):
        index_file = os.path.join(self.index_dir, "faiss.index")
        meta_file = os.path.join(self.meta_dir, "metadata.pkl")

        try:
            if os.path.exists(index_file) and os.path.exists(meta_file):
                print("Loading existing index...")
                self.index = faiss.read_index(index_file)

                # Check dimension (no type check for IP, but ensure dim matches)
                if self.index.d != self.EMBEDDING_DIM:
                    print(
                        f"Dimension mismatch! Expected {self.EMBEDDING_DIM}, but index has {self.index.d}. Recreating index.")
                    raise ValueError("Dimension mismatch")

                with open(meta_file, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.document_chunks = data['chunks']
                    self.chunk_metadata = data['metadata']

                print(f"Loaded {len(self.documents)} documents with {len(self.document_chunks)} chunks")
            else:
                raise FileNotFoundError("No existing data")
        except Exception as e:
            print(f"Failed to load existing data ({e}). Creating new index.")
            self.index = faiss.IndexFlatIP(self.EMBEDDING_DIM)  # Inner product for cosine
            self.documents = []
            self.document_chunks = []
            self.chunk_metadata = []

    def _save_data(self):
        print("Saving index and metadata...")
        faiss.write_index(self.index, os.path.join(self.index_dir, "faiss.index"))
        with open(os.path.join(self.meta_dir, "metadata.pkl"), 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'chunks': self.document_chunks,
                'metadata': self.chunk_metadata
            }, f)

    def reset_storage(self):
        """Clear all stored data"""
        for dir_path in [self.blob_dir, self.index_dir, self.meta_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                os.makedirs(dir_path)
        self.documents = []
        self.document_chunks = []
        self.chunk_metadata = []
        self.index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
        print("Storage reset successfully!")

    def _extract_text_multiple_methods(self, pdf_path: str) -> str:
        text = ""

        # Method 1: pdfplumber
        try:
            print("Trying pdfplumber extraction...")
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                    if page_text:
                        text += page_text + "\n\n"
            if len(text.strip()) > 100:
                print(f"✓ pdfplumber extracted {len(text)} characters")
                return text
        except Exception as e:
            print(f"pdfplumber failed: {e}")

        # Method 2: PyMuPDF
        try:
            print("Trying PyMuPDF extraction...")
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text("text") + "\n\n"
            doc.close()
            if len(text.strip()) > 100:
                print(f"✓ PyMuPDF extracted {len(text)} characters")
                return text
        except Exception as e:
            print(f"PyMuPDF failed: {e}")

        # Method 3: PyPDF2
        try:
            print("Trying PyPDF2 extraction...")
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n\n"
            if len(text.strip()) > 50:
                print(f"✓ PyPDF2 extracted {len(text)} characters")
                return text
        except Exception as e:
            print(f"PyPDF2 failed: {e}")

        # OCR fallback
        if OCR_AVAILABLE and len(text.strip()) < 100:
            try:
                print("Trying OCR extraction...")
                images = convert_from_path(pdf_path)
                for image in images:
                    text += pytesseract.image_to_string(image) + "\n\n"
                if len(text.strip()) > 50:
                    print(f"✓ OCR extracted {len(text)} characters")
                    return text
            except Exception as e:
                print(f"OCR failed: {e}")

        if not text.strip():
            print("Warning: No usable text extracted.")
        return text

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(\n\s*){2,}', '\n\n', text)
        lines = text.split('\n')
        cleaned_lines = []
        seen_lines = set()
        for line in lines:
            line = line.strip()
            if len(line) < 10 or line.isdigit() or line in seen_lines:
                continue
            seen_lines.add(line)
            cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)

    def _create_improved_chunks(self, text: str, chunk_size: int = 1200, overlap: int = 300,
                                min_chunk_size: int = 300) -> List[str]:
        text = self._clean_text(text)
        if len(text) < min_chunk_size:
            return [text] if text else []

        paragraphs = re.split(r'\n\n+', text)
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph or len(paragraph) < 20:
                continue
            if len(current_chunk) + len(paragraph) + 1 > chunk_size:
                if len(current_chunk) >= min_chunk_size:
                    chunks.append(current_chunk.strip())
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + paragraph
            else:
                current_chunk += "\n" + paragraph if current_chunk else paragraph

        if len(current_chunk) >= min_chunk_size:
            chunks.append(current_chunk.strip())

        # Aggressive merge for small chunks
        merged_chunks = []
        current_merge = ""
        for chunk in chunks:
            if len(current_merge) + len(chunk) <= chunk_size:
                current_merge += "\n" + chunk if current_merge else chunk
            else:
                if len(current_merge) >= min_chunk_size:
                    merged_chunks.append(current_merge)
                current_merge = chunk
        if len(current_merge) >= min_chunk_size:
            merged_chunks.append(current_merge)

        return merged_chunks if merged_chunks else [text[:chunk_size]]

    def upload_pdf(self, pdf_path: str) -> Dict:
        if not os.path.exists(pdf_path):
            return {"success": False, "error": "File not found"}

        file_hash = self._get_file_hash(pdf_path)
        filename = os.path.basename(pdf_path)

        for doc in self.documents:
            if doc['hash'] == file_hash:
                return {"success": True, "message": "File already processed", "doc_id": doc['id']}

        print(f"\nUploading: {filename}")
        blob_path = os.path.join(self.blob_dir, f"{file_hash}.pdf")
        shutil.copy2(pdf_path, blob_path)
        print("✓ File stored in blob storage")

        text = self._extract_text_multiple_methods(pdf_path)
        if not text or len(text.strip()) < 50:
            return {"success": False, "error": "Could not extract sufficient text from PDF"}

        print(f"✓ Extracted {len(text)} characters")

        chunks = self._create_improved_chunks(text)
        print(f"✓ Created {len(chunks)} chunks")

        print("\nFirst few chunks (for debugging):")
        for i, chunk in enumerate(chunks[:5]):
            print(f"Chunk {i + 1} ({len(chunk)} chars): {chunk[:200]}...")

        print("Generating embeddings...")
        embeddings = self.embedder.encode(chunks, show_progress_bar=True,
                                          normalize_embeddings=True)  # Normalize for cosine

        start_idx = len(self.document_chunks)
        self.index.add(np.array(embeddings).astype('float32'))

        doc_id = len(self.documents)
        doc_info = {
            'id': doc_id,
            'filename': filename,
            'hash': file_hash,
            'blob_path': blob_path,
            'upload_time': datetime.now().isoformat(),
            'num_chunks': len(chunks),
            'char_count': len(text)
        }
        self.documents.append(doc_info)

        for i, chunk in enumerate(chunks):
            self.document_chunks.append(chunk)
            self.chunk_metadata.append({
                'doc_id': doc_id,
                'chunk_id': i,
                'filename': filename,
                'chunk_start': start_idx + i
            })

        self._save_data()
        print(f"✓ Document processed successfully!")
        return {"success": True, "doc_id": doc_id, "info": doc_info}

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, Dict, float]]:
        if not self.document_chunks:
            return []

        print(f"Searching for: '{query}'")
        query_embedding = self.embedder.encode([query], normalize_embeddings=True).astype('float32')
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.document_chunks)))

        results = []
        print("\nTop search results:")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.document_chunks):
                continue
            chunk = self.document_chunks[idx]
            metadata = self.chunk_metadata[idx]
            print(f"  {i + 1}. Score: {dist:.3f} | {chunk[:150]}...")
            results.append((chunk, metadata, float(dist)))

        return results

    def ask_question(self, question: str) -> str:
        if not self.document_chunks:
            return "No documents loaded. Please upload a PDF first."

        print(f"\nQuestion: {question}")
        search_results = self.search(question, top_k=10)

        if not search_results:
            return "No relevant information found in the search."

        # Sort by score (higher better for cosine/IP)
        sorted_results = sorted(search_results, key=lambda x: x[2], reverse=True)

        context_chunks = []
        sources = set()
        total_context_length = 0
        max_context_length = 3000
        min_similarity = 0.1  # Relaxed threshold (0-1 scale)

        for chunk, metadata, score in sorted_results:
            if score > min_similarity and total_context_length + len(chunk) < max_context_length:
                context_chunks.append(chunk)
                sources.add(metadata['filename'])
                total_context_length += len(chunk)

        # If too few, add top 3 regardless
        if len(context_chunks) < 3 and sorted_results:
            for i in range(min(3, len(sorted_results))):
                chunk, metadata, _ = sorted_results[i]
                if chunk not in context_chunks:
                    context_chunks.append(chunk)
                    sources.add(metadata['filename'])
                    total_context_length += len(chunk)

        if not context_chunks:
            return "No sufficiently relevant information found."

        context = "\n\n".join(context_chunks)
        print(f"Using {len(context_chunks)} chunks for context ({len(context)} characters)")

        prompt = f"""Using the following document excerpts, provide the best possible answer to the question. If the excerpts don't contain the information, say "I can't find this in the provided excerpts" and explain why.

Document excerpts:
{context}

Question: {question}

Answer:"""

        print("Generating answer...")
        response = self.qa_pipeline(prompt, max_length=300, do_sample=True, temperature=0.3)
        answer = response[0]['generated_text'].strip()

        formatted_answer = f"{answer}\n\nSources: {', '.join(sources)}"
        return formatted_answer

    def debug_document(self, doc_id: int) -> str:
        if doc_id >= len(self.documents):
            return "Document not found"

        doc = self.documents[doc_id]
        doc_chunks = [(i, self.document_chunks[i]) for i, m in enumerate(self.chunk_metadata) if m['doc_id'] == doc_id]

        # Sample raw text (first 500 chars)
        raw_text_sample = " ".join([c for _, c in doc_chunks])[:500] + "..." if doc_chunks else "No text"

        debug_info = f"""
Document Debug Info:
- Filename: {doc['filename']}
- Total chunks: {len(doc_chunks)}
- Character count: {doc['char_count']}
- Raw text sample (first 500 chars): {raw_text_sample}

First 5 chunks:
"""
        for i, (chunk_idx, chunk) in enumerate(doc_chunks[:5]):
            debug_info += f"\nChunk {i + 1} (global index {chunk_idx}, {len(chunk)} chars):\n{chunk}\n{'-' * 50}"
        return debug_info

    def list_documents(self) -> List[Dict]:
        return self.documents


class EnhancedPDFChat:
    """Enhanced interactive interface with reset command"""

    def __init__(self):
        self.qa_system = EnhancedLocalPDFQASystem()

    def run(self):
        print("\n" + "=" * 60)
        print("Enhanced PDF Question-Answering System (Local)")
        print("=" * 60)
        print("\nCommands:")
        print("  upload <path>    - Upload a PDF file")
        print("  list            - List all documents")
        print("  ask <question>  - Ask a question")
        print("  debug <doc_id>  - Debug document chunks")
        print("  search <query>  - Test search functionality")
        print("  reset           - Clear all stored data")
        print("  quit            - Exit")
        print("\n")

        while True:
            try:
                command = input(">> ").strip()

                if command.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                elif command.startswith('upload '):
                    path = command[7:].strip().strip('"\'')
                    result = self.qa_system.upload_pdf(path)
                    if result['success']:
                        print(f"✓ Uploaded successfully! Document ID: {result.get('doc_id', 'N/A')}")
                    else:
                        print(f"✗ Error: {result.get('error', 'Unknown error')}")

                elif command == 'list':
                    docs = self.qa_system.list_documents()
                    if not docs:
                        print("No documents uploaded yet.")
                    else:
                        print("\nDocuments:")
                        for doc in docs:
                            print(
                                f"  [{doc['id']}] {doc['filename']} - {doc['num_chunks']} chunks ({doc['char_count']} chars)")

                elif command.startswith('ask '):
                    question = command[4:].strip()
                    if question:
                        answer = self.qa_system.ask_question(question)
                        print("\n" + "-" * 50)
                        print(answer)
                        print("-" * 50 + "\n")

                elif command.startswith('debug '):
                    try:
                        doc_id = int(command[6:].strip())
                        debug_info = self.qa_system.debug_document(doc_id)
                        print(debug_info)
                    except ValueError:
                        print("Please provide a valid document ID")

                elif command.startswith('search '):
                    query = command[7:].strip()
                    if query:
                        results = self.qa_system.search(query, top_k=5)
                        print(f"\nSearch results for: '{query}'")
                        for i, (chunk, metadata, score) in enumerate(results):
                            print(f"\n{i + 1}. Score: {score:.3f} | File: {metadata['filename']}")
                            print(f"   {chunk[:300]}...")

                elif command == 'reset':
                    self.qa_system.reset_storage()

                else:
                    print("Unknown command. Available commands: upload, list, ask, debug, search, reset, quit")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    # Check dependencies
    try:
        import fitz
        import pdfplumber
        from sentence_transformers import SentenceTransformer

        SentenceTransformer('all-mpnet-base-v2')
    except ImportError:
        print("Missing core dependencies. Run: pip install PyMuPDF pdfplumber sentence-transformers")
        exit(1)

    if not OCR_AVAILABLE:
        print("OCR not available. For scanned PDFs, run: pip install pytesseract pdf2image")
        print("And install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")

    # Run
    chat = EnhancedPDFChat()
    chat.run()

