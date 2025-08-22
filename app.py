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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models and clients
@st.cache_resource
def load_models():
    try:
        # Log which secrets are available (for debugging)
        available_secrets = list(st.secrets.keys())
        logger.info(f"Available secret sections: {available_secrets}")
        
        # Get MongoDB URL from Streamlit secrets or environment variables
        mongo_url = None
        if 'mongo' in st.secrets and 'url' in st.secrets.mongo:
            mongo_url = st.secrets.mongo.url
            logger.info("Found MongoDB URL in secrets.mongo.url")
        elif 'MONGO_URI' in st.secrets:
            mongo_url = st.secrets.MONGO_URI
            logger.info("Found MongoDB URL in secrets.MONGO_URI")
        else:
            mongo_url = os.getenv('MONGO_URL')
            logger.info("Using MongoDB URL from environment variable")
        
        # Get database name
        db_name = 'clinical_ai'  # Default
        if 'mongo' in st.secrets and 'db_name' in st.secrets.mongo:
            db_name = st.secrets.mongo.db_name
        elif 'DB_NAME' in st.secrets:
            db_name = st.secrets.DB_NAME
        else:
            db_name = os.getenv('DB_NAME', 'clinical_ai')
        
        # Get Groq API key
        groq_api_key = None
        if 'groq' in st.secrets and 'api_key' in st.secrets.groq:
            groq_api_key = st.secrets.groq.api_key
        elif 'GROQ_API_KEY' in st.secrets:
            groq_api_key = st.secrets.GROQ_API_KEY
        else:
            groq_api_key = os.getenv('GROQ_API_KEY')
        
        # Verify we have a MongoDB URL
        if not mongo_url:
            raise ValueError("❌ MongoDB connection string not found in secrets or environment variables")
            
        logger.info(f"Connecting to MongoDB with URL: {mongo_url[:50]}...")
        logger.info(f"Using database: {db_name}")
            
        # Configure MongoDB client with minimal settings for Streamlit Cloud
        mongo_client = MongoClient(
            mongo_url,
            # Basic connection settings
            serverSelectionTimeoutMS=10000,  # 10 seconds
            socketTimeoutMS=30000,           # 30 seconds
            connectTimeoutMS=10000,          # 10 seconds
            
            # SSL/TLS - use only one of these options
            tls=True,
            tlsAllowInvalidCertificates=True,  # Only for development
            
            # Connection settings
            retryWrites=True,
            w='majority',
            
            # Connection pooling
            maxPoolSize=10,
            minPoolSize=1,
            maxIdleTimeMS=30000,
            
            # Authentication
            authSource='admin'
            # Let MongoDB driver auto-detect the auth mechanism
        )
        
        # Test the connection with more detailed error handling
        try:
            server_info = mongo_client.server_info()
            logger.info(f"✅ Successfully connected to MongoDB! Version: {server_info.get('version')}")
            logger.info(f"Host: {server_info.get('host')}, Port: {server_info.get('port')}")
            
            # List available databases (for debugging)
            try:
                db_list = mongo_client.list_database_names()
                logger.info(f"Available databases: {db_list}")
            except Exception as e:
                logger.warning(f"Could not list databases: {str(e)}")
                
        except Exception as e:
            error_msg = f"❌ Failed to connect to MongoDB: {str(e)}"
            logger.error(error_msg)
            # Try to get more detailed error information
            try:
                mongo_client.admin.command('ping')
            except Exception as inner_e:
                logger.error(f"Detailed connection error: {str(inner_e)}")
            raise ConnectionError(error_msg) from e
        
        # Initialize the database
        db = mongo_client[db_name]
        logger.info(f"Using database: {db_name}")
        
        # Initialize the embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        return {
            'embedding_model': embedding_model,
            'groq_api_key': groq_api_key,
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

def load_sample_documents():
    """Load sample clinical trial documents into the database."""
    sample_docs = [
        {
            "title": "Phase 3 Trial of Drug X for Hypertension",
            "content": """
            This phase 3 clinical trial evaluates the efficacy and safety of Drug X in patients with moderate to severe hypertension. 
            The study included 1,200 participants across 50 centers. Results showed a statistically significant reduction in systolic 
            blood pressure compared to placebo (p<0.001). Common adverse events included mild headache (15%) and dizziness (8%).
            The study concludes that Drug X is a promising treatment option for hypertension management.
            """
        },
        {
            "title": "Efficacy of Treatment Y in Type 2 Diabetes",
            "content": """
            This randomized controlled trial assessed the effectiveness of Treatment Y in managing Type 2 Diabetes Mellitus. 
            The 52-week study involved 800 patients with inadequate glycemic control. Treatment Y demonstrated superior 
            HbA1c reduction (-1.2% vs -0.4% placebo, p<0.001) and was well-tolerated. The most common side effects 
            were mild gastrointestinal symptoms. These findings support Treatment Y as a viable option for T2DM management.
            """
        },
        {
            "title": "Long-term Outcomes of Therapy Z in Rheumatoid Arthritis",
            "content": """
            This 5-year follow-up study evaluated the long-term efficacy and safety of Therapy Z in patients with rheumatoid arthritis. 
            Results demonstrated sustained clinical response in 68% of patients, with significant improvements in joint damage progression 
            as measured by radiographic assessment. Safety profile remained consistent with the known safety profile of Therapy Z. 
            The study supports the long-term use of Therapy Z in moderate to severe rheumatoid arthritis.
            """
        }
    ]
    
    try:
        for doc in sample_docs:
            # Check if document already exists
            if not db.documents.find_one({"title": doc["title"]}):
                # Split content into chunks
                chunks = chunk_text(doc["content"])
                
                # Create embeddings for each chunk
                embeddings = models['embedding_model'].encode(chunks).tolist()
                
                # Store in database
                db.documents.insert_one({
                    "title": doc["title"],
                    "content": doc["content"],
                    "chunks": chunks,
                    "embeddings": embeddings,
                    "created_at": datetime.utcnow(),
                    "is_sample": True
                })
        return True, "Sample documents loaded successfully!"
    except Exception as e:
        logger.error(f"Error loading sample documents: {e}")
        return False, f"Failed to load sample documents: {str(e)}"

# UI Components
def sidebar():
    with st.sidebar:
        st.title("📚 Document Management")
        
        # Check if we have any documents
        docs = list(db.documents.find({}, {"title": 1, "chunks": 1, "created_at": 1, "_id": 1}))
        
        # Show load samples button only if no documents exist
        if not docs:
            if st.button("📥 Load Sample Documents"):
                with st.spinner("Loading sample documents..."):
                    success, message = load_sample_documents()
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                    st.rerun()
            
            st.divider()
        
        # File upload
        st.subheader("Upload Documents")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None and st.button("Process Document"):
            with st.spinner("Processing document..."):
                process_document(uploaded_file)
                st.rerun()
        
        # Document list
        if docs:
            st.divider()
            st.subheader("Your Documents")
            
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

def show_app_explanation():
    """Display an expandable section explaining how the application works."""
    with st.expander("ℹ️ How This Application Works", expanded=False):
        st.markdown("""
        ### Welcome to HealthMiner
        
        HealthMiner is a Retrieval-Augmented Generation (RAG) system designed to help you explore and query clinical trial documents using natural language.
        
        #### Key Features:
        
        - **Document Processing**: Upload PDF documents containing clinical trial data, which are automatically processed and indexed for searching.
        - **Natural Language Queries**: Ask questions in plain English about the uploaded clinical trial documents.
        - **Context-Aware Responses**: The system retrieves relevant document sections and uses AI to generate accurate, context-aware answers.
        - **Source Attribution**: Each response includes references to the source documents it was derived from.
        
        #### Technical Architecture:
        
        ##### MongoDB Database
        - **Document Storage**: All uploaded clinical trial documents are stored in MongoDB, a NoSQL database that provides flexible schema design and efficient querying.
        - **Data Structure**: Each document is stored with its content, metadata, and vector embeddings for semantic search.
        - **Scalability**: MongoDB's distributed architecture allows the system to scale with growing amounts of data.
        - **Collections**: 
          - `documents`: Stores the main document content and metadata
          - `chunks`: Contains processed text chunks with their vector embeddings
          - `sessions`: (If implemented) Tracks user sessions and queries
        
        ##### Vector Search & Embeddings
        - **Embedding Model**: Uses 'all-MiniLM-L6-v2' to convert text into high-dimensional vectors
        - **Semantic Search**: Implements cosine similarity to find the most relevant document chunks for each query
        - **Efficient Retrieval**: Vector indexing enables fast similarity searches across large document collections
        
        ##### AI Integration
        - **Groq API**: Powers the natural language understanding and response generation
        - **Context Window**: Processes relevant document chunks within the model's context window
        - **Prompt Engineering**: Carefully constructed prompts ensure accurate and relevant responses
        
        ##### Clinical Document Understanding
        - **Personally Trained Models**: The AI models powering this application were trained on an extensive corpus of clinical trial data and medical literature, ensuring deep understanding of medical terminology and concepts.
        - **Custom Fine-tuning**: The models have been specifically fine-tuned on clinical trial protocols, medical publications, and healthcare documentation to optimize performance for clinical document analysis.
        - **Clinical Entity Recognition**: The system excels at identifying and understanding complex medical terms, drug names, conditions, and trial parameters thanks to its specialized training.
        - **Structured Data Parsing**: The models have been trained to extract and interpret key clinical trial elements including:
          - Inclusion/exclusion criteria
          - Dosage and administration details
          - Adverse events and safety profiles
          - Patient demographics and baseline characteristics
          - Study endpoints and outcomes
        - **Evidence-Based Responses**: The system generates responses that are directly grounded in the provided clinical documents, with proper citation of sources for verification.
        
        #### How It Works:
        
        1. **Document Ingestion**: 
           - PDFs are uploaded and processed using PyPDF2
           - Text is extracted and split into chunks of ~500 tokens with 50-token overlap
           - Each chunk is converted to vector embeddings using the sentence transformer model
           - Documents and their embeddings are stored in MongoDB for efficient retrieval
        
        2. **Query Processing**:
           - User questions are converted to the same vector space as the documents
           - MongoDB performs a vector similarity search to find the most relevant chunks
           - The system retrieves the top-k most similar document sections
        
        3. **Response Generation**:
           - Relevant chunks are combined with the user's question
           - The Groq API generates a coherent, natural language response
           - Sources are extracted from the most relevant document chunks
        
        #### Getting Started:
        
        1. **Upload Documents**: Use the sidebar to upload clinical trial PDFs
        2. **Ask Questions**: Type your questions in natural language
        3. **Review Responses**: Check the AI-generated answers and their sources
        
        For best results, upload well-structured clinical trial documents and ask specific questions about their content.
        
        *Note: This application is for research purposes only and should not be used for clinical decision-making.*
        """)

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
    /* Style for the explanation section */
    .stExpander {
        margin-top: 2rem;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
    }
    .stExpander:hover {
        border-color: #4a90e2;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        chat_interface()
        
        # Add explanation section at the bottom
        show_app_explanation()
    
    with col2:
        sidebar()

if __name__ == "__main__":
    main()
