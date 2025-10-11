import chromadb
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from PIL import Image
from pathlib import Path

logger = logging.getLogger(__name__)

class TextRAGSystem:
    """A RAG system specialized for text documents."""
    def __init__(self, db_path: str = "data/chroma_db_text"):
        # This model is optimized for understanding text sentences.
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device="cpu")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="text_documents")
        logger.info(f"Text RAG System initialized. Documents: {self.collection.count()}")

    def add_documents(self, docs_to_add: List[Dict[str, Any]]):
        if not docs_to_add: return
        texts = [doc['text'] for doc in docs_to_add]
        metadatas = [doc['metadata'] for doc in docs_to_add]
        ids = [meta.get('source', str(hash(text))) for meta, text in zip(metadatas, texts)]
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=False).tolist()
        self.collection.add(embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids)
        logger.info(f"Added {len(docs_to_add)} documents to Text RAG. Total: {self.collection.count()}")

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        if self.collection.count() == 0: return []
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=k)
        
        formatted_results = []
        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i]
            score = 1 - distance
            if score > 0.6:
                formatted_results.append({
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "score": score
                })
        return sorted(formatted_results, key=lambda x: x['score'], reverse=True)

class ImageRAGSystem:
    """A RAG system specialized for searching images with text."""
    def __init__(self, db_path: str = "data/chroma_db_image"):
        # This CLIP model understands both images and text.
        self.embedding_model = SentenceTransformer('clip-ViT-B-32', device="cpu")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="image_documents")
        logger.info(f"Image RAG System initialized. Images: {self.collection.count()}")

    def add_images_from_folder(self, folder_path: str):
        """Finds images in a folder, creates embeddings, and adds them to the database."""
        image_paths = list(Path(folder_path).glob('*.[jp][pn]g')) # Find .jpg, .jpeg, .png
        if not image_paths:
            logger.warning(f"No images found in folder: {folder_path}")
            return
            
        images = [Image.open(filepath) for filepath in image_paths]
        filepath_strs = [str(filepath) for filepath in image_paths]
        
        image_embeddings = self.embedding_model.encode(images, convert_to_tensor=False).tolist()

        self.collection.add(
            embeddings=image_embeddings,
            metadatas=[{"filepath": fp} for fp in filepath_strs],
            ids=filepath_strs
        )
        logger.info(f"Added {len(image_paths)} images to Image RAG. Total: {self.collection.count()}")

    def search_images_by_text(self, text_query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Searches for images using a text query."""
        if self.collection.count() == 0: return []
        
        # The same model encodes the text query
        query_embedding = self.embedding_model.encode([text_query]).tolist()
        
        results = self.collection.query(query_embeddings=query_embedding, n_results=k)
        
        # Returns the metadata of the most similar images
        return results.get('metadatas', [[]])[0]