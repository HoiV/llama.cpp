#
# rag_service.py RAG database builder
# it can process .docx .PDF and text based files
# rupo zhang (rizhang) 2/28/2025
# 

import os
import time
import sys
import json
import logging
import numpy as np
import torch
import hnswlib
import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
import uvicorn
import threading
from docx import Document
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, BackgroundTasks, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Setup logging
logging.basicConfig(
    filename='rag_service.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('rag_service')

# FastAPI app
app = FastAPI(title="RAG Database Builder Service")

# Models for API
class RagBuildRequest(BaseModel):
    folder_path: str
    output_dir: str
    max_chunk_size: int = 1000
    similarity_threshold: float = 0.7
    model_name: str = 'all-MiniLM-L6-v2'
    use_gpu: bool = True
    index_params: dict = {
        "ef_construction": 200,
        "M": 16, 
        "ef_search": 100
    }

class RagBuildResponse(BaseModel):
    job_id: str
    status: str
    message: str
    using_gpu: bool

class JobStatus(BaseModel):
    status: str
    progress: float
    message: str
    completed: bool
    using_gpu: bool = False
    document_count: int = 0
    document_types: dict = {}
    chunk_count: int = 0
    total_tokens: int = 0

# Global job tracking
jobs = {}

# Document extraction function with support for multiple file types
def extract_text_from_documents(folder_path):
    texts = []
    filenames = []
    metadata = []
    
    logger.info(f"Extracting text from documents in {folder_path}")
    
    # Try to import PDF extraction libraries
    pdf_support = False
    try:
        from pdfminer.high_level import extract_text as pdf_extract_text
        pdf_support = True
        logger.info("PDF support is available")
    except ImportError:
        logger.warning("PDF support not available. Install with: pip install pdfminer.six")
    
    # Track document types for reporting
    doc_types = {"docx": 0, "txt": 0, "pdf": 0, "unsupported": 0, "failed": 0}
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue
            
        # Prepare file metadata
        file_info = {
            "filename": filename,
            "path": file_path,
            "size_bytes": os.path.getsize(file_path),
            "extraction_time": 0,
            "char_count": 0
        }
        
        start_time = time.time()
        
        # Process DOCX files
        if filename.lower().endswith(".docx"):
            try:
                doc = Document(file_path)
                text = " ".join([para.text for para in doc.paragraphs if para.text.strip()])
                if text:
                    texts.append(text)
                    filenames.append(filename)
                    file_info["char_count"] = len(text)
                    file_info["type"] = "docx"
                    file_info["extraction_time"] = time.time() - start_time
                    metadata.append(file_info)
                    doc_types["docx"] += 1
                    logger.info(f"Extracted DOCX: {filename} ({len(text)} chars)")
                else:
                    logger.warning(f"DOCX file {filename} contained no text")
                    doc_types["failed"] += 1
            except Exception as e:
                logger.error(f"Error extracting text from DOCX {filename}: {str(e)}")
                doc_types["failed"] += 1
        
        # Process TXT files
        elif filename.lower().endswith(".txt"):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                    text = file.read()
                if text.strip():
                    texts.append(text)
                    filenames.append(filename)
                    file_info["char_count"] = len(text)
                    file_info["type"] = "txt"
                    file_info["extraction_time"] = time.time() - start_time
                    metadata.append(file_info)
                    doc_types["txt"] += 1
                    logger.info(f"Extracted TXT: {filename} ({len(text)} chars)")
                else:
                    logger.warning(f"TXT file {filename} was empty")
                    doc_types["failed"] += 1
            except Exception as e:
                logger.error(f"Error extracting text from TXT {filename}: {str(e)}")
                doc_types["failed"] += 1
        
        # Process PDF files
        elif filename.lower().endswith(".pdf") and pdf_support:
            try:
                text = pdf_extract_text(file_path)
                
                # Clean up PDF text - remove excessive whitespace
                text = ' '.join(text.split())
                
                if text.strip():
                    texts.append(text)
                    filenames.append(filename)
                    file_info["char_count"] = len(text)
                    file_info["type"] = "pdf"
                    file_info["extraction_time"] = time.time() - start_time
                    metadata.append(file_info)
                    doc_types["pdf"] += 1
                    logger.info(f"Extracted PDF: {filename} ({len(text)} chars)")
                else:
                    logger.warning(f"PDF {filename} contained no extractable text - possible scanned document")
                    doc_types["failed"] += 1
            except Exception as e:
                logger.error(f"Error extracting text from PDF {filename}: {str(e)}")
                doc_types["failed"] += 1
        else:
            # Unsupported file type
            doc_types["unsupported"] += 1
    
    logger.info(f"Document extraction summary: {doc_types}")
    return texts, filenames, metadata, doc_types

# Estimate token count (useful for context tracking)
def estimate_token_count(text):
    """Get a rough estimate of token count for transformer models"""
    return len(text.split()) * 1.3  # Rough approximation

# Semantic chunking with embedding-based boundaries
def semantic_chunking(text, embedder, max_chunk_size=1000, similarity_threshold=0.7):
    # Split into sentences (simple approach)
    sentences = []
    current = ""
    for char in text:
        current += char
        if char in ['.', '!', '?'] and len(current.strip()) > 0:
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())
    
    if not sentences:
        return []
        
    chunks = []
    current_chunk = []
    current_length = 0

    # Check if we can use GPU for embeddings
    device = next(embedder.parameters()).device

    try:
        # Process in batches to avoid OOM
        batch_size = min(32, len(sentences))
        sentence_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            with torch.no_grad():
                batch_embeddings = embedder.encode(batch, convert_to_tensor=True)
                sentence_embeddings.append(batch_embeddings)
                
        sentence_embeddings = torch.cat(sentence_embeddings, dim=0).cpu().numpy()
    except Exception as e:
        logger.error(f"Error computing sentence embeddings: {str(e)}")
        # Fallback to simple length-based chunking
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    # Create chunks based on semantic similarity
    for i, sentence in enumerate(sentences):
        sentence_len = len(sentence)
        
        if current_length + sentence_len > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_len
        else:
            if i > 0 and current_chunk:
                # Calculate cosine similarity
                sim = np.dot(sentence_embeddings[i], sentence_embeddings[i-1]) / (
                    np.linalg.norm(sentence_embeddings[i]) * np.linalg.norm(sentence_embeddings[i-1]) + 1e-8
                )
                if sim < similarity_threshold and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_len
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_len
            else:
                current_chunk.append(sentence)
                current_length += sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# Build RAG database
async def build_rag_database(job_id, folder_path, output_dir, max_chunk_size, 
                          similarity_threshold, model_name, use_gpu, index_params):
    using_gpu = False
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        jobs[job_id] = {
            "status": "extracting", 
            "progress": 0.0, 
            "message": "Extracting text from documents", 
            "completed": False,
            "using_gpu": False,
            "document_count": 0,
            "document_types": {},
            "chunk_count": 0,
            "total_tokens": 0
        }
        
        # Extract documents
        documents, filenames, metadata, doc_types = extract_text_from_documents(folder_path)
        
        if not documents:
            jobs[job_id] = {
                "status": "failed", 
                "progress": 0.0, 
                "message": "No documents found or text could be extracted", 
                "completed": True,
                "using_gpu": False,
                "document_count": 0,
                "document_types": doc_types,
                "chunk_count": 0,
                "total_tokens": 0
            }
            return
        
        jobs[job_id]["document_count"] = len(documents)
        jobs[job_id]["document_types"] = doc_types
        jobs[job_id]["status"] = "loading_model"
        jobs[job_id]["progress"] = 0.1
        jobs[job_id]["message"] = f"Loading embedding model: {model_name}"
        
        # Load the sentence transformer model
        try:
            embedder = SentenceTransformer(model_name)
            
            # Use GPU if available and requested
            if use_gpu and torch.cuda.is_available():
                embedder = embedder.to(torch.device("cuda:0"))
                using_gpu = True
                logger.info(f"Using GPU for embeddings with {model_name}")
            else:
                logger.info(f"Using CPU for embeddings with {model_name}")
                
            jobs[job_id]["using_gpu"] = using_gpu
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            jobs[job_id] = {
                "status": "failed", 
                "progress": 0.0, 
                "message": f"Failed to load embedding model: {str(e)}", 
                "completed": True,
                "using_gpu": False,
                "document_count": len(documents),
                "document_types": doc_types,
                "chunk_count": 0,
                "total_tokens": 0
            }
            return
            
        # Process documents into chunks
        jobs[job_id]["status"] = "chunking"
        jobs[job_id]["progress"] = 0.2
        jobs[job_id]["message"] = "Chunking documents"
        
        all_chunks = []
        chunk_sources = []
        total_tokens = 0
        
        for i, (doc, filename) in enumerate(zip(documents, filenames)):
            try:
                chunks = semantic_chunking(doc, embedder, max_chunk_size, similarity_threshold)
                all_chunks.extend(chunks)
                chunk_sources.extend([filename] * len(chunks))
                
                # Estimate token count
                doc_tokens = sum(estimate_token_count(chunk) for chunk in chunks)
                total_tokens += doc_tokens
                
                jobs[job_id] = {
                    "status": "chunking", 
                    "progress": 0.2 + 0.3 * ((i + 1) / len(documents)),
                    "message": f"Chunking documents ({i+1}/{len(documents)})", 
                    "completed": False,
                    "using_gpu": using_gpu,
                    "document_count": len(documents),
                    "document_types": doc_types,
                    "chunk_count": len(all_chunks),
                    "total_tokens": int(total_tokens)
                }
                logger.info(f"Document {i+1}/{len(documents)}: {filename} - {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error processing document {filename}: {str(e)}")
                # Continue with other documents
        
        if not all_chunks:
            jobs[job_id] = {
                "status": "failed", 
                "progress": 0.0, 
                "message": "No chunks could be created", 
                "completed": True,
                "using_gpu": using_gpu,
                "document_count": len(documents),
                "document_types": doc_types,
                "chunk_count": 0,
                "total_tokens": 0
            }
            return
            
        # Generate embeddings
        jobs[job_id]["status"] = "embedding"
        jobs[job_id]["progress"] = 0.5
        jobs[job_id]["message"] = f"Generating embeddings for {len(all_chunks)} chunks"
        
        # Process in batches to avoid memory issues
        batch_size = 32 if using_gpu else 64
        embeddings_list = []
        
        for i in range(0, len(all_chunks), batch_size):
            batch_end = min(i + batch_size, len(all_chunks))
            batch = all_chunks[i:batch_end]
            
            # Generate batch embeddings
            batch_embeddings = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embeddings_list.append(batch_embeddings)
            
            # Update progress
            progress = 0.5 + 0.3 * (batch_end / len(all_chunks))
            jobs[job_id]["progress"] = progress
            jobs[job_id]["message"] = f"Embedding chunks ({batch_end}/{len(all_chunks)})"
            
            # Log progress every 5 batches
            if (i // batch_size) % 5 == 0 or batch_end == len(all_chunks):
                logger.info(f"Embedding progress: {batch_end}/{len(all_chunks)} chunks")
        
        # Combine all batch embeddings
        embeddings = np.vstack(embeddings_list)
        
        # Build HNSWLIB index
        jobs[job_id]["status"] = "indexing"
        jobs[job_id]["progress"] = 0.8
        jobs[job_id]["message"] = "Building HNSWLIB index"
        
        dimension = embeddings.shape[1]
        num_elements = embeddings.shape[0]
        
        # Create the HNSWLIB index with the specified parameters
        ef_construction = index_params.get("ef_construction", 200)
        M = index_params.get("M", 16)
        ef_search = index_params.get("ef_search", 100)
        
        # Use L2 space for compatibility with C++ code
        index = hnswlib.Index(space='l2', dim=dimension)
        index.init_index(max_elements=max(num_elements, 1000), ef_construction=ef_construction, M=M)
        index.set_ef(ef_search)
        
        # Add all vectors
        index.add_items(embeddings, np.arange(num_elements))
        
        logger.info(f"Built HNSWLIB index with {num_elements} vectors, dim={dimension}")
        
        # Save results
        jobs[job_id]["status"] = "saving"
        jobs[job_id]["progress"] = 0.9
        jobs[job_id]["message"] = "Saving index and chunks"
        
        index_path = os.path.join(output_dir, "rag_index.bin")
        chunks_path = os.path.join(output_dir, "chunks.json")
        
        # Save the HNSWLIB index
        index.save_index(index_path)
        
        # Save the chunks with source information
        chunk_data = [
            {
                "id": i, 
                "text": chunk, 
                "source": source,
                "tokens": int(estimate_token_count(chunk))
            }
            for i, (chunk, source) in enumerate(zip(all_chunks, chunk_sources))
        ]
        
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)
            
        # Save metadata
        with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump({
                "created_at": time.time(),
                "model": model_name, 
                "dimension": dimension,
                "document_count": len(documents),
                "document_types": doc_types,
                "chunk_count": len(all_chunks),
                "total_tokens": int(total_tokens),
                "index_params": index_params,
                "embedding_time": time.time(),
                "documents": metadata
            }, f, ensure_ascii=False, indent=2)
            
        # Success!
        jobs[job_id] = {
            "status": "completed", 
            "progress": 1.0, 
            "message": f"Database built successfully with {len(all_chunks)} chunks from {len(documents)} documents", 
            "completed": True,
            "using_gpu": using_gpu,
            "document_count": len(documents),
            "document_types": doc_types,
            "chunk_count": len(all_chunks),
            "total_tokens": int(total_tokens)
        }
        
        logger.info(f"Job {job_id} completed. Built RAG database with {len(all_chunks)} chunks. GPU used: {using_gpu}")
        
    except Exception as e:
        logger.error(f"Error building RAG database: {str(e)}")
        jobs[job_id] = {
            "status": "failed", 
            "progress": 0.0, 
            "message": f"Error: {str(e)}", 
            "completed": True,
            "using_gpu": using_gpu,
            "document_count": jobs[job_id].get("document_count", 0),
            "document_types": jobs[job_id].get("document_types", {}),
            "chunk_count": jobs[job_id].get("chunk_count", 0),
            "total_tokens": jobs[job_id].get("total_tokens", 0)
        }

# API endpoints
@app.post("/build", response_model=RagBuildResponse)
async def build_database(request: RagBuildRequest, background_tasks: BackgroundTasks):
    job_id = f"job_{int(time.time())}"
    
    # Validate folder path
    if not os.path.exists(request.folder_path):
        raise HTTPException(status_code=400, detail=f"Folder path does not exist: {request.folder_path}")
    
    # Check if GPU is available
    gpu_available = torch.cuda.is_available()
    will_use_gpu = request.use_gpu and gpu_available
    
    # Start the build process in the background
    background_tasks.add_task(
        build_rag_database, 
        job_id, 
        request.folder_path, 
        request.output_dir, 
        request.max_chunk_size, 
        request.similarity_threshold,
        request.model_name,
        request.use_gpu,
        request.index_params
    )
    
    logger.info(f"Job {job_id} started. Building RAG database from {request.folder_path}. GPU requested: {request.use_gpu}, GPU available: {gpu_available}")
    return {
        "job_id": job_id, 
        "status": "started", 
        "message": f"RAG database build started from {request.folder_path}",
        "using_gpu": will_use_gpu
    }

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.get("/jobs", response_model=Dict[str, JobStatus])
async def list_jobs():
    """List all jobs and their status"""
    return jobs

@app.get("/health")
async def health_check():
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"
    
    return {
        "status": "ok", 
        "service": "RAG Database Builder", 
        "version": "1.0.0",
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "active_jobs": len([j for j in jobs.values() if not j.get("completed", True)]),
        "total_jobs": len(jobs)
    }

@app.post("/test")
async def test_index(index_path: str, query: str, k: int = 5):
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail=f"Index file not found: {index_path}")
    
    chunks_path = os.path.join(os.path.dirname(index_path), "chunks.json")
    if not os.path.exists(chunks_path):
        raise HTTPException(status_code=404, detail=f"Chunks file not found: {chunks_path}")
    
    try:
        # Load the index
        index = hnswlib.Index(space='l2', dim=384)  # Assuming dimension, will resize automatically
        index.load_index(index_path)
        
        # Load the chunks
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunk_data = json.load(f)
            
        # Load model for encoding the query
        use_gpu = torch.cuda.is_available()
        model = SentenceTransformer('all-MiniLM-L6-v2')
        if use_gpu:
            model = model.to(torch.device("cuda:0"))
        
        # Encode the query
        query_embedding = model.encode(query, convert_to_numpy=True)
        
        # Search
        ids, distances = index.knn_query(query_embedding, k=k)
        
        # Format results
        results = []
        for i, (idx, dist) in enumerate(zip(ids[0], distances[0])):
            chunk = chunk_data[int(idx)]
            results.append({
                "rank": i + 1,
                "distance": float(dist),
                "text": chunk["text"],
                "source": chunk.get("source", "unknown"),
                "tokens": chunk.get("tokens", int(estimate_token_count(chunk["text"])))
            })
            
        return {
            "query": query,
            "using_gpu": use_gpu,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error testing index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error testing index: {str(e)}")

# Windows service class
class RagService(win32serviceutil.ServiceFramework):
    _svc_name_ = "RagDatabaseBuilder"
    _svc_display_name_ = "RAG Database Builder Service"
    _svc_description_ = "Service for building RAG databases from document folders using HNSWLIB"

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        socket.setdefaulttimeout(60)
        self.is_running = False
        self.server = None
        self.server_thread = None

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        self.is_running = False
        if self.server:
            self.server.should_exit = True
        logger.info("RAG Database Builder service is stopping...")

    def SvcDoRun(self):
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, '')
        )
        self.is_running = True
        logger.info("RAG Database Builder service is starting...")
        self.main()

    def run_server(self):
        config = uvicorn.Config(app=app, host="127.0.0.1", port=8000, log_level="info")
        self.server = uvicorn.Server(config)
        self.server.run()

    def main(self):
        # Start FastAPI server in a separate thread
        self.server_thread = threading.Thread(target=self.run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        logger.info("FastAPI server started on http://127.0.0.1:8000")
        
        # Main service loop
        while self.is_running:
            # Wait for service stop signal
            rc = win32event.WaitForSingleObject(self.hWaitStop, 5000)
            if rc == win32event.WAIT_OBJECT_0:
                # Service stop signal received
                break
        
        logger.info("RAG Database Builder service stopped")

# Service main
if __name__ == '__main__':
        print("\n*** RAG Database Builder Service - ***")
        print("API will be available at http://127.0.0.1:8000")
        
        # Print GPU information
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"\nGPU Support: Available ({device_count} device(s))")
            for i in range(device_count):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("\nGPU Support: Not available (CPU only)")
              
        # Run the FastAPI app directly with uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8000)
