import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="🧬 HealthMiner",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import io
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models and clients
@st.cache_resource
def load_models():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get MongoDB URL and other settings
    mongo_url = os.getenv('MONGO_URL')
    db_name = os.getenv('DB_NAME', 'clinical_ai')
    
    try:
        # Configure MongoDB client with settings that worked in the test script
        mongo_client = MongoClient(
            mongo_url,
            ssl=True,
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=5000
        )
        
        # Test the connection
        mongo_client.server_info()  # Will raise an exception if connection fails
        print("Successfully connected to MongoDB Atlas!")
        
        # Initialize the database and collection
        db = mongo_client[db_name]
        print(f"Using database: {db_name}")
        print(f"Available collections: {db.list_collection_names()}")
        
        return {
            'embedding_model': SentenceTransformer('all-MiniLM-L6-v2'),
            'groq_api_key': os.getenv('GROQ_API_KEY'),
            'mongo_client': mongo_client,
        }
        
    except Exception as e:
        print(f"\n!!! ERROR CONNECTING TO MONGODB: {str(e)}")
        print("Troubleshooting tips:")
        print("1. Make sure your MongoDB Atlas cluster is running")
        print("2. Check if your IP is whitelisted in MongoDB Atlas")
        print("3. Verify your MongoDB connection string in the .env file")
        print("4. Check your internet connection")
        raise

models = load_models()
db = models['mongo_client'][os.getenv('DB_NAME', 'clinical_ai')]

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'deleted_docs' not in st.session_state:
    st.session_state.deleted_docs = set()

# Helper functions
def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        return "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap) if words[i:i + chunk_size]]

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a_np = np.array(a)
    b_np = np.array(b)
    return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))

def process_document(file):
    try:
        # Read and process the file
        file_content = file.getvalue()
        text = extract_text_from_pdf(file_content)
        if not text:
            st.error("Could not extract text from the PDF")
            return None
            
        # Create document
        doc_id = str(uuid.uuid4())
        chunks = chunk_text(text)
        
        # Generate embeddings
        embeddings = models['embedding_model'].encode(chunks).tolist()
        
        # Save to MongoDB
        doc = {
            '_id': doc_id,
            'title': file.name,
            'content': text,
            'chunks': chunks,
            'embeddings': embeddings,
            'created_at': datetime.utcnow()
        }
        
        db.documents.insert_one(doc)
        st.session_state.documents.append(doc)
        st.success(f"✅ Successfully processed: {file.name}")
        st.success(f"📄 Extracted {len(chunks)} chunks")
        return doc
        
    except Exception as e:
        st.error(f"❌ Error processing document: {str(e)}")
        logger.exception("Document processing failed")
        return None

def query_documents(query: str, top_k: int = 3):
    try:
        print("\n=== Starting query_documents ===")
        print(f"Query: {query}")
        
        # Check if embedding model is loaded
        if 'embedding_model' not in models:
            error_msg = "Embedding model not loaded"
            print(f"Error: {error_msg}")
            return f"Error: {error_msg}", []
            
        # Get relevant documents
        print("Generating query embedding...")
        query_embedding = models['embedding_model'].encode([query])[0].tolist()
        query_embedding_np = np.array(query_embedding)
        
        # Get all documents with their embeddings
        print("Fetching documents from database...")
        try:
            all_docs = list(db.documents.find({}, {'content': 1, 'title': 1, 'embeddings': 1}))
            print(f"Found {len(all_docs)} total documents in database")
            
            # Calculate similarity for each document
            print("Calculating similarities...")
            for doc in all_docs:
                if not doc.get('embeddings') or not doc['embeddings']:
                    doc['similarity'] = -1
                    continue
                    
                # Get the first chunk's embedding (or average if multiple chunks)
                doc_embedding = np.array(doc['embeddings'][0])
                
                # Calculate cosine similarity
                dot_product = np.dot(query_embedding_np, doc_embedding)
                query_norm = np.linalg.norm(query_embedding_np)
                doc_norm = np.linalg.norm(doc_embedding)
                
                if query_norm == 0 or doc_norm == 0:
                    doc['similarity'] = 0
                else:
                    doc['similarity'] = float(dot_product / (query_norm * doc_norm))
            
            # Sort by similarity and get top k
            results = sorted(all_docs, key=lambda x: x.get('similarity', -1), reverse=True)[:top_k]
            results = [doc for doc in results if doc.get('similarity', -1) > 0]  # Filter out negative similarities
            print(f"Found {len(results)} relevant documents after filtering")
            
        except Exception as e:
            error_msg = f"Error querying database: {str(e)}"
            print(error_msg)
            return f"Error: {error_msg}", []
        
        if not results:
            print("No relevant documents found")
            return "I couldn't find any relevant information to answer your question.", []
            
        # Prepare context for the LLM (limit to first 3 chunks to avoid token limits)
        context_parts = []
        for i, doc in enumerate(results):
            content = doc.get('content', '')
            # Take first 2000 chars of content to avoid hitting token limits
            context_parts.append(f"Document {i+1} (Relevance: {doc.get('similarity', 0):.2f}): {content[:2000]}")
        context = "\n\n".join(context_parts)
        
        # Generate response using Groq API
        if not models.get('groq_api_key'):
            error_msg = "Groq API key not found in environment variables"
            print(f"Error: {error_msg}")
            return f"Error: {error_msg}", []
            
        headers = {
            "Authorization": f"Bearer {models['groq_api_key']}",
            "Content-Type": "application/json"
        }
        
        # Truncate context if too long (to avoid token limits)
        max_context_length = 10000  # Adjust based on model's context window
        if len(context) > max_context_length:
            context = context[:max_context_length] + "... [truncated]"
        
        # Use a supported model - check https://console.groq.com/docs/models for available models
        payload = {
            "model": "llama3-70b-8192",  # Updated to a supported model
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant that answers questions about clinical trials and medical research."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ],
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        print("Sending request to Groq API...")
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30  # 30 second timeout
            )
            
            # Log the response status and headers (but not the full content)
            print(f"Groq API response status: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            
            response.raise_for_status()
            
            # Try to parse the response
            try:
                response_data = response.json()
                if 'choices' not in response_data or not response_data['choices']:
                    raise ValueError("No choices in response")
                    
                answer = response_data['choices'][0]['message']['content']
                print("Successfully got response from Groq API")
                
            except (ValueError, KeyError) as e:
                error_msg = f"Error parsing Groq API response: {str(e)}\nResponse: {response.text[:500]}"
                print(error_msg)
                return f"Error: Failed to process API response. {str(e)}", []
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Error calling Groq API: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f"\nStatus Code: {e.response.status_code}\nResponse: {e.response.text[:500]}"
            print(error_msg)
            return f"Error: Failed to get response from AI service. Please try again later.", []
            
        sources = [doc['title'] for doc in results]
        return answer, sources
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Unexpected error in query_documents: {str(e)}\n{error_details}")
        return f"I encountered an error while processing your request: {str(e)}", []

# UI Components
def sidebar():
    with st.sidebar:
        st.title("📚 Document Management")
        
        # File upload
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
        if uploaded_file is not None and st.button("Process Document"):
            with st.spinner("Processing document..."):
                process_document(uploaded_file)
        
        # Document list
        st.subheader("Your Documents")
        docs = list(db.documents.find({}, {"title": 1, "chunks": 1, "created_at": 1, "_id": 1}))
        
        for doc in docs:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"📄 {doc['title']}")
                st.caption(f"{len(doc.get('chunks', []))} chunks • {doc['created_at'].strftime('%b %d, %Y')}")
            with col2:
                if st.button("🗑️", key=f"del_{doc['_id']}"):
                    db.documents.delete_one({"_id": doc['_id']})
                    st.rerun()
            st.divider()

def chat_interface():
    st.title("🧬 HealthMiner")
    st.caption("AI-powered RAG system for exploring clinical trial data")
    
    # Initialize chat history if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Container for chat history
    chat_container = st.container()
    
    # Display all messages from history first
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["query"]:
                with st.chat_message("user"):
                    st.markdown(msg["query"])
            if msg["answer"]:
                with st.chat_message("assistant"):
                    st.markdown(msg["answer"])
                    if msg["sources"]:
                        with st.expander("Sources"):
                            for src in msg["sources"]:
                                st.write(f"• {src}")
    
    # Chat input at the bottom
    if prompt := st.chat_input("Ask a question about clinical trials...", key="chat_input"):
        # Add user message to chat history
        st.session_state.chat_history.append({
            "query": prompt,
            "answer": "",
            "sources": []
        })
        
        # Show the user's message immediately
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Show loading message
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your question..."):
                    response, sources = query_documents(prompt)
                    st.markdown(response)
                    
                    if sources:
                        with st.expander("Sources"):
                            for src in sources:
                                st.write(f"• {src}")
        
        # Update the last message with the response
        if st.session_state.chat_history:
            st.session_state.chat_history[-1].update({
                "answer": response,
                "sources": sources
            })
        
        # Rerun to update the display with the new message in history
        st.rerun()

def main():
    # Apply custom CSS
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        chat_interface()
    
    with col2:
        sidebar()

if __name__ == "__main__":
    main()
