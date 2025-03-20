import sys
import requests
import json
import os
import numpy as np
from typing import List, Dict
import hnswlib
from sentence_transformers import SentenceTransformer
import logging
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTextEdit, QPushButton, QLineEdit, QFileDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

# Enable high-DPI scaling
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

# Base directory for bundled files
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration
LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"
MODEL_NAME = "phi3"
MAX_TOKENS = 2000
TEMPERATURE = 0.7
LOG_FILE = os.path.join(BASE_DIR, "chatbot_debug.log")

# Database paths - can be changed at runtime
DB_CONFIG = {
    "index_path": os.path.join(BASE_DIR, "rag_index.bin"),
    "chunks_path": os.path.join(BASE_DIR, "chunks.json"),
    "metadata_path": os.path.join(BASE_DIR, "metadata.json")
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE)]
)
logger = logging.getLogger(__name__)

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')
logger.info(f"Loaded embedding model: all-MiniLM-L6-v2, dimension={embedder.get_sentence_embedding_dimension()}")

def retrieve_chunks(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    """Retrieve relevant chunks using HNSWLIB"""
    logger.debug(f"Retrieving chunks for query: {query}")
    try:
        # Verify files exist
        if not os.path.exists(DB_CONFIG["index_path"]):
            logger.error(f"Index file missing: {DB_CONFIG['index_path']}")
            return []
            
        if not os.path.exists(DB_CONFIG["chunks_path"]):
            logger.error(f"Chunks file missing: {DB_CONFIG['chunks_path']}")
            return []
        
        # Get dimension from metadata or model
        dimension = embedder.get_sentence_embedding_dimension()
        if os.path.exists(DB_CONFIG["metadata_path"]):
            try:
                with open(DB_CONFIG["metadata_path"], 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    if "dimension" in metadata:
                        dimension = int(metadata["dimension"])
                        logger.info(f"Using dimension from metadata: {dimension}")
            except Exception as e:
                logger.warning(f"Error reading metadata: {e}")
        
        # Load chunks first to have them available regardless of index success
        try:
            with open(DB_CONFIG["chunks_path"], 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            logger.info(f"Loaded {len(chunk_data)} chunks from {DB_CONFIG['chunks_path']}")
        except Exception as e:
            logger.error(f"Failed to load chunks: {e}")
            return []
        
        # Create and load index
        try:
            logger.info(f"Loading index with dimension {dimension}")
            index = hnswlib.Index(space='l2', dim=dimension)
            index.load_index(DB_CONFIG["index_path"])
            index.set_ef(50)  # Search parameter
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
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
            logger.error(f"Search failed: {e}")
            return []
        
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
            
        logger.info(f"Retrieved {len(retrieved)} chunks")
        return retrieved
        
    except Exception as e:
        logger.error(f"Error in retrieve_chunks: {str(e)}", exc_info=True)
        return []

class LlamaChatbot:
    def __init__(self):
        self.chat_history: List[Dict[str, str]] = []
        logger.info("Chatbot initialized")

    def clear_history(self):
        self.chat_history = []
        logger.info("Chat history cleared")

class StreamWorker(QThread):
    chunk_received = pyqtSignal(str)
    finished = pyqtSignal(tuple)

    def __init__(self, chatbot, message, use_rag=True):
        super().__init__()
        self.chatbot = chatbot
        self.message = message
        self.use_rag = use_rag
        self._is_running = True
        self.citations = []

    def stop(self):
        self._is_running = False
        logger.info("Generation stopped by user")

    def run(self):
        if self.use_rag:
            try:
                # Retrieve relevant chunks
                retrieved_chunks = retrieve_chunks(self.message)
                
                if retrieved_chunks:
                    # Build augmented prompt with context
                    context_text = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)
                    augmented_message = f"Use the following information to answer the question:\n\n{context_text}\n\nQuestion: {self.message}"
                    self.citations = [chunk["source"] for chunk in retrieved_chunks]
                    logger.info(f"Using RAG with {len(retrieved_chunks)} chunks")
                else:
                    augmented_message = self.message
                    self.citations = []
                    logger.warning("No chunks retrieved, using direct question")
            except Exception as e:
                logger.error(f"Error in RAG: {e}")
                augmented_message = self.message
                self.citations = []
        else:
            augmented_message = self.message
            self.citations = []

        payload = {
            "model": MODEL_NAME,
            "messages": self.chatbot.chat_history + [{"role": "user", "content": augmented_message}],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "stream": True
        }

        assistant_message = ""
        try:
            response = requests.post(LLAMA_SERVER_URL, json=payload, stream=True, timeout=60)
            
            if response.status_code != 200:
                error_msg = f"Server returned error {response.status_code}: {response.text}"
                logger.error(error_msg)
                self.finished.emit((error_msg, []))
                return
                
            for chunk in response.iter_lines():
                if not self._is_running:
                    self.finished.emit(("Generation stopped by user.", []))
                    return
                    
                if chunk:
                    raw_chunk = chunk.decode("utf-8").strip()
                    if raw_chunk.startswith("data: ") and raw_chunk != "data: [DONE]":
                        try:
                            data = json.loads(raw_chunk.replace("data: ", ""))
                            if "choices" in data and data["choices"]:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    assistant_message += content
                                    self.chunk_received.emit(content)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON chunk: {raw_chunk}")
                            continue
            
            # Update chat history
            self.chatbot.chat_history.append({"role": "user", "content": self.message})
            self.chatbot.chat_history.append({"role": "assistant", "content": assistant_message})
            
            self.finished.emit((assistant_message, self.citations))
            
        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            self.finished.emit((f"Error: Connection to LLM server failed - {str(e)}", []))
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            self.finished.emit((f"Error: {str(e)}", []))

class ChatCopilotUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAG Chat Co-Pilot")
        
        screen = QApplication.primaryScreen().geometry()
        self.width = 350
        self.height = screen.height() - 100  # Leave some space at the bottom
        self.setGeometry(screen.width() - self.width, 0, self.width, self.height)
        
        # Use regular window with stay-on-top
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        
        self.chatbot = LlamaChatbot()
        
        # UI Setup
        self.main_widget = QWidget(self)
        self.main_widget.setStyleSheet("background-color: rgba(245, 245, 245, 230);")
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        # Chat display
        self.chat_display = QTextEdit(self)
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Arial", 12))
        self.chat_display.setStyleSheet("background-color: white; border: 1px solid #ccc; border-radius: 5px;")
        
        # Welcome message and database check
        self.chat_display.append("Welcome to RAG Chat Co-Pilot!")
        
        rag_status = self.check_rag_database()
        if rag_status:
            self.chat_display.append(f"RAG database loaded: {rag_status}")
        else:
            self.chat_display.append("RAG database not found or incomplete.")
            self.chat_display.append("You can load a database using the 'Load Database' button.")
        
        self.layout.addWidget(self.chat_display)
        
        # Input field
        self.input_field = QLineEdit(self)
        self.input_field.setFont(QFont("Arial", 12))
        self.input_field.setMinimumHeight(40)
        self.input_field.setPlaceholderText("Type your question here...")
        self.input_field.setStyleSheet("background-color: white; border: 1px solid #ccc; border-radius: 5px;")
        self.input_field.returnPressed.connect(self.send_message_with_rag)
        self.layout.addWidget(self.input_field)
        
        # Button row 1
        button_row1 = QVBoxLayout()
        
        self.send_rag_btn = QPushButton("Send with RAG", self)
        self.send_rag_btn.setMinimumHeight(40)
        self.send_rag_btn.clicked.connect(self.send_message_with_rag)
        self.send_rag_btn.setStyleSheet("background-color: #0078d4; color: white; border-radius: 5px;")
        button_row1.addWidget(self.send_rag_btn)

        self.send_no_rag_btn = QPushButton("Send (Direct)", self)
        self.send_no_rag_btn.setMinimumHeight(40)
        self.send_no_rag_btn.clicked.connect(self.send_message_no_rag)
        self.send_no_rag_btn.setStyleSheet("background-color: #0078d4; color: white; border-radius: 5px;")
        button_row1.addWidget(self.send_no_rag_btn)
        
        # Button row 2
        button_row2 = QVBoxLayout()
        
        self.stop_btn = QPushButton("Stop Generation", self)
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.clicked.connect(self.stop_generation)
        self.stop_btn.setStyleSheet("background-color: #d83b01; color: white; border-radius: 5px;")
        self.stop_btn.setEnabled(False)
        button_row2.addWidget(self.stop_btn)

        self.clear_btn = QPushButton("Clear History", self)
        self.clear_btn.setMinimumHeight(40)
        self.clear_btn.clicked.connect(self.clear_history)
        self.clear_btn.setStyleSheet("background-color: #d83b01; color: white; border-radius: 5px;")
        button_row2.addWidget(self.clear_btn)
        
        self.load_db_btn = QPushButton("Load Database", self)
        self.load_db_btn.setMinimumHeight(40)
        self.load_db_btn.clicked.connect(self.load_database)
        self.load_db_btn.setStyleSheet("background-color: #107c10; color: white; border-radius: 5px;")
        button_row2.addWidget(self.load_db_btn)
        
        self.layout.addLayout(button_row1)
        self.layout.addLayout(button_row2)
        
        # Check LLM server
        self.check_llm_server()
        
        self.worker = None

    def check_rag_database(self):
        """Check if RAG database files exist and return a status string or None"""
        try:
            index_exists = os.path.exists(DB_CONFIG["index_path"])
            chunks_exists = os.path.exists(DB_CONFIG["chunks_path"])
            
            if not index_exists or not chunks_exists:
                return None
                
            # Try to get chunk count
            with open(DB_CONFIG["chunks_path"], 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                chunk_count = len(chunks)
            
            # Try to read metadata if available
            if os.path.exists(DB_CONFIG["metadata_path"]):
                with open(DB_CONFIG["metadata_path"], 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    doc_count = metadata.get("document_count", "?")
                    return f"{chunk_count} chunks from {doc_count} documents"
            
            return f"{chunk_count} chunks"
            
        except Exception as e:
            logger.error(f"Error checking RAG database: {e}")
            return None

    def check_llm_server(self):
        """Check if the LLM server is running"""
        try:
            response = requests.get("http://localhost:8080/v1/models", timeout=3)
            if response.status_code == 200:
                self.chat_display.append("LLM server is online and ready.")
            else:
                self.chat_display.append("Warning: LLM server responded with an error.")
        except requests.RequestException:
            self.chat_display.append("Warning: LLM server not responding. Please start the server.")
        except Exception as e:
            self.chat_display.append(f"Error checking server: {str(e)}")

    def send_message_with_rag(self):
        user_input = self.input_field.text().strip()
        if not user_input:
            return
        
        self.chat_display.append(f"\nYou: {user_input}")
        self.input_field.clear()
        
        self.chat_display.append("Assistant: ")  # Start response line
        
        # Check if RAG database is available
        if not os.path.exists(DB_CONFIG["index_path"]) or not os.path.exists(DB_CONFIG["chunks_path"]):
            self.chat_display.append("(RAG database not found, using direct query instead)")
            use_rag = False
        else:
            use_rag = True
        
        self.worker = StreamWorker(self.chatbot, user_input, use_rag=use_rag)
        self.worker.chunk_received.connect(self.append_chunk)
        self.worker.finished.connect(self.stream_finished)
        self.worker.start()
        
        self.stop_btn.setEnabled(True)
        self.send_rag_btn.setEnabled(False)
        self.send_no_rag_btn.setEnabled(False)

    def send_message_no_rag(self):
        user_input = self.input_field.text().strip()
        if not user_input:
            return
        
        self.chat_display.append(f"\nYou: {user_input}")
        self.input_field.clear()
        
        self.chat_display.append("Assistant: ")  # Start response line
        self.worker = StreamWorker(self.chatbot, user_input, use_rag=False)
        self.worker.chunk_received.connect(self.append_chunk)
        self.worker.finished.connect(self.stream_finished)
        self.worker.start()
        
        self.stop_btn.setEnabled(True)
        self.send_rag_btn.setEnabled(False)
        self.send_no_rag_btn.setEnabled(False)

    def append_chunk(self, chunk):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(chunk)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()
        QApplication.processEvents()

    def stream_finished(self, result):
        full_response, citations = result
        
        if full_response.startswith("Error:") or full_response == "Generation stopped by user.":
            self.chat_display.append(f"\n{full_response}")
        elif citations:  # Show citations if available
            self.chat_display.append("\n\nSources:")
            for i, source in enumerate(citations):
                self.chat_display.append(f"[{i+1}] {source}")
        
        # Reset button states
        self.stop_btn.setEnabled(False)
        self.send_rag_btn.setEnabled(True)
        self.send_no_rag_btn.setEnabled(True)
        self.worker = None

    def stop_generation(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()

    def clear_history(self):
        self.chatbot.clear_history()
        self.chat_display.clear()
        self.chat_display.append("Chat history cleared.")
        
        # Re-check RAG status
        rag_status = self.check_rag_database()
        if rag_status:
            self.chat_display.append(f"RAG database loaded: {rag_status}")

    def load_database(self):
        """Open a file dialog to select an index file and load the database"""
        folder = QFileDialog.getExistingDirectory(self, "Select Database Folder")
        if not folder:
            return
            
        # Check for required files
        index_path = os.path.join(folder, "rag_index.bin")
        chunks_path = os.path.join(folder, "chunks.json")
        metadata_path = os.path.join(folder, "metadata.json")
        
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            self.chat_display.append(f"Error: Required database files not found in {folder}")
            return
            
        # Update configuration
        DB_CONFIG["index_path"] = index_path
        DB_CONFIG["chunks_path"] = chunks_path
        DB_CONFIG["metadata_path"] = metadata_path
        
        self.chat_display.append(f"RAG database loaded from: {folder}")
        
        # Show database stats
        rag_status = self.check_rag_database()
        if rag_status:
            self.chat_display.append(f"Database contains: {rag_status}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Arial", 10))
    window = ChatCopilotUI()
    window.show()
    sys.exit(app.exec_())