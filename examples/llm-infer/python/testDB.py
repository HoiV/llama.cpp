import sys
import requests
import json
import os
import numpy as np
from typing import List, Dict
import hnswlib
from sentence_transformers import SentenceTransformer
import logging


# Database paths - can be changed at runtime
DB_CONFIG = {
    "index_path": "./rag_index.bin",
    "chunks_path": "./chunks.json",
    "metadata_path": "./metadata.json"
}

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print(f"Loaded embedding model: all-MiniLM-L6-v2, dimension={embedder.get_sentence_embedding_dimension()}")

def retrieve_chunks(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    """Retrieve relevant chunks using HNSWLIB"""
    print(f"Retrieving chunks for query: {query}")
    try:
        # Get dimension from metadata or model
        dimension = embedder.get_sentence_embedding_dimension()
        if os.path.exists(DB_CONFIG["metadata_path"]):
            try:
                with open(DB_CONFIG["metadata_path"], 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    if "dimension" in metadata:
                        dimension = int(metadata["dimension"])
                        print(f"Using dimension from metadata: {dimension}")
            except Exception as e:
                print(f"Error reading metadata: {e}")
        
        # Load chunks first to have them available regardless of index success
        try:
            with open(DB_CONFIG["chunks_path"], 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            print(f"Loaded {len(chunk_data)} chunks from {DB_CONFIG['chunks_path']}")
        except Exception as e:
            print(f"Failed to load chunks: {e}")
            return []
        
        # Create and load index
        try:
            print(f"Loading index with dimension {dimension}")
            index = hnswlib.Index(space='l2', dim=dimension)
            index.load_index(DB_CONFIG["index_path"])
            index.set_ef(50)  # Search parameter
            print(f"RAG database loaded from: {DB_CONFIG["index_path"]}")
        except Exception as e:
            print(f"Failed to load index: {e}")
            return []
        
        # Generate query embedding
        query_embedding = embedder.encode(query, convert_to_numpy=True)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Search for nearest chunks
        try:
            k = min(top_k, len(chunk_data))
            labels, distances = index.knn_query(query_embedding, k=k)
        except Exception as e:
            print(f"Search failed: {e}")
            return []
        print("labels:")
        print(labels)
        print("^^^^^^^^")
        print("distances:")
        print(distances)
        print("^^^^^^^^")
        # Format results
        retrieved = []
        for i, idx in enumerate(labels[0]):
            if 0 <= idx < len(chunk_data):
                chunk = chunk_data[idx]
                
                # Handle different possible chunk formats
                if isinstance(chunk, dict):
                    text = chunk.get("text", "")
                    source = chunk.get("source", f"Document {idx}")
                else:
                    text = str(chunk)
                    source = f"Chunk {idx}"
                    
                retrieved.append({
                    "text": text,
                    "source": source,
                    "distance": float(distances[0][i])
                })
            
        print(f"Retrieved {len(retrieved)} chunks")
        context_text = "\n\n".join(chunk["text"] for chunk in retrieved)
        augmented_message = f"Use the following information to answer the question:\n\n{context_text}\n\nQuestion: {query}"
        print(f"--> [{augmented_message}]")

        chunk_index = 0
        for chunk in retrieved:
            print(f"==> Chunk {chunk_index}: (distance {chunk["distance"]}) ")
            print(f"   {chunk["text"]}")
            print(f"===")
            chunk_index += 1

        return retrieved
        
    except Exception as e:
        print(f"Error in retrieve_chunks: {str(e)}", exc_info=True)
        return []

def check_database():
    # Check for required files
    index_path = DB_CONFIG["index_path"]
    chunks_path = DB_CONFIG["chunks_path"]
    metadata_path = DB_CONFIG["metadata_path"]
    if not os.path.exists(index_path):
        print(f"Index file missing: {index_path}")
        return
        
    if not os.path.exists(chunks_path):
        print(f"Chunks file missing: {chunks_path}")
        return
    
    if os.path.exists(metadata_path):
        print(f"metadata file missing: {metadata_path}")
        return
    
    try:
        with open(DB_CONFIG["chunks_path"], 'r', encoding='utf-8') as f:
            chunks = json.load(f)
            chunk_count = len(chunks)
    
        # Try to read metadata if available
        with open(DB_CONFIG["metadata_path"], 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            doc_count = metadata.get("document_count", "?")
            print(f"{chunk_count} chunks from {doc_count} documents")
    except Exception as e:
        print(f"load database failed: {e}")
        return    

if __name__ == "__main__":
    check_database()
    retrieve_chunks("Who is Rupo Zhang")
    retrieve_chunks("Who is Santa Claus")
    exit()