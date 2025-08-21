import streamlit as st
import requests
import json
from datetime import datetime
import io
import os

# Configure page
st.set_page_config(
    page_title="🧬 HealthMiner",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend URL - Updated to use local FastAPI server with /api prefix
BACKEND_URL = "http://127.0.0.1:8000/api"

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents' not in st.session_state:
    st.session_state.documents = []

# Helper functions
def fetch_documents():
    try:
        response = requests.get(f"{BACKEND_URL}/documents")
        if response.status_code == 200:
            st.session_state.documents = response.json()
        else:
            st.error(f"Failed to fetch documents: {response.status_code}")
    except Exception as e:
        st.error(f"Error fetching documents: {str(e)}")

def upload_document(file):
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        response = requests.post(f"{BACKEND_URL}/upload", files=files)
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ Successfully uploaded: {result['title']}")
            st.success(f"📄 Processed {len(result['chunks'])} chunks")
            fetch_documents()  # Refresh document list
            return True
        else:
            error_msg = response.json().get('detail', 'Unknown error')
            st.error(f"❌ Upload failed: {error_msg}")
            return False
    except Exception as e:
        st.error(f"❌ Error uploading document: {str(e)}")
        return False

def query_documents(query):
    try:
        with st.spinner("🔍 Analyzing your question..."):
            response = requests.post(
                f"{BACKEND_URL}/query",
                json={"query": query},
                headers={"Content-Type": "application/json"}
            )
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            error_msg = response.json().get('detail', 'Unknown error')
            st.error(f"❌ Query failed: {error_msg}")
            return None
    except Exception as e:
        st.error(f"❌ Error processing query: {str(e)}")
        return None

def initialize_sample_data():
    try:
        with st.spinner("🔄 Loading sample clinical trial data..."):
            response = requests.post(f"{BACKEND_URL}/initialize-sample-data")
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ {result['message']}")
            if 'documents' in result:
                st.info(f"📚 Loaded: {', '.join(result['documents'])}")
            fetch_documents()  # Refresh document list
            return True
        else:
            error_msg = response.json().get('detail', 'Unknown error')
            st.error(f"❌ Failed to initialize sample data: {error_msg}")
            return False
    except Exception as e:
        st.error(f"❌ Error initializing sample data: {str(e)}")
        return False

def delete_document(doc_id):
    try:
        response = requests.delete(f"{BACKEND_URL}/documents/{doc_id}")
        if response.status_code == 200:
            st.success("✅ Document deleted successfully")
            fetch_documents()  # Refresh document list
            return True
        else:
            st.error("❌ Failed to delete document")
            return False
    except Exception as e:
        st.error(f"❌ Error deleting document: {str(e)}")
        return False

# Main app
def main():
    # Add 3D enhanced styles with animated background
    st.markdown("""
    <style>
        /* Enhanced Animated Background */
        .stApp {
            position: relative;
            z-index: 1;
        }
        
        .stApp > div:first-child {
            background: linear-gradient(-45deg, #e6e9ff, #d9deff, #c9d0ff, #b8c1ff);
            background-size: 300% 300%;
            animation: gradientBG 8s ease infinite;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: -1;
        }
        
        .stApp > div:first-child::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at 20% 30%, rgba(255,255,255,0.8) 0%, transparent 25%),
                        radial-gradient(circle at 80% 70%, rgba(200,210,255,0.6) 0%, transparent 25%);
            animation: float 15s ease-in-out infinite;
        }
        
        @keyframes gradientBG {
            0% { 
                background-position: 0% 50%;
                background-color: #e6e9ff;
            }
            25% {
                background-color: #e0e4ff;
            }
            50% { 
                background-position: 100% 50%;
                background-color: #d9deff;
            }
            75% {
                background-color: #e0e4ff;
            }
            100% { 
                background-position: 0% 50%;
                background-color: #e6e9ff;
            }
        }
        /* 3D Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px) perspective(500px) rotateX(5deg); }
            to { opacity: 1; transform: translateY(0) perspective(500px) rotateX(0); }
        }
        
        @keyframes float {
            0% { transform: translateY(0px) perspective(1000px) rotateX(0deg) rotateY(0deg); }
            50% { transform: translateY(-10px) perspective(1000px) rotateX(2deg) rotateY(2deg); }
            100% { transform: translateY(0px) perspective(1000px) rotateX(0deg) rotateY(0deg); }
        }
        
        /* 3D Header */
        .header-container {
            text-align: center;
            padding: 2.5rem 0;
            background: linear-gradient(145deg, #5d73e0 0%, #6a3d9a 100%);
            border-radius: 20px;
            margin: 2rem 0 3rem 0;
            color: white !important;
            box-shadow: 0 15px 35px rgba(0,0,0,0.2), 
                        0 5px 15px rgba(0,0,0,0.1);
            transform-style: preserve-3d;
            transform: perspective(1000px) rotateX(1deg);
            transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
            animation: float 8s ease-in-out infinite;
        }
        
        .header-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 50%, rgba(0,0,0,0.1) 100%);
            pointer-events: none;
        }
        
        .header-container:hover {
            transform: perspective(1000px) rotateX(2deg) translateY(-5px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.3), 
                        0 10px 20px rgba(0,0,0,0.2);
        }
        
        /* 3D Buttons */
        .stButton>button {
            border: none !important;
            border-radius: 12px !important;
            background: linear-gradient(145deg, #667eea, #764ba2) !important;
            color: white !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px;
            transform: perspective(500px) translateZ(0);
            box-shadow: 5px 5px 15px rgba(0,0,0,0.1),
                        -5px -5px 15px rgba(255,255,255,0.1),
                        inset 2px 2px 5px rgba(255,255,255,0.2),
                        inset -2px -2px 5px rgba(0,0,0,0.1) !important;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
            position: relative;
            overflow: hidden;
        }
        
        .stButton>button::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(45deg, rgba(255,255,255,0.2), transparent);
            transform: translateX(-100%);
            transition: 0.5s;
        }
        
        .stButton>button:hover {
            transform: perspective(500px) translateZ(10px) !important;
            box-shadow: 8px 8px 20px rgba(0,0,0,0.2),
                       -8px -8px 20px rgba(255,255,255,0.1),
                       inset 3px 3px 8px rgba(255,255,255,0.3),
                       inset -3px -3px 8px rgba(0,0,0,0.2) !important;
        }
        
        .stButton>button:active {
            transform: perspective(500px) translateZ(-5px) !important;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1),
                       -2px -2px 8px rgba(255,255,255,0.1),
                       inset 1px 1px 3px rgba(0,0,0,0.2) !important;
        }
        
        .stButton>button:hover::before {
            transform: translateX(100%);
        }
        
        /* 3D Text Areas */
        .stTextArea>div>div>textarea {
            border-radius: 12px !important;
            min-height: 150px !important;
            border: none !important;
            background: #f8f9ff !important;
            box-shadow: inset 3px 3px 8px rgba(0,0,0,0.1),
                        inset -3px -3px 8px rgba(255,255,255,0.8) !important;
            transition: all 0.3s ease !important;
            transform: perspective(500px) translateZ(0);
        }
        
        .stTextArea>div>div>textarea:focus {
            outline: none !important;
            box-shadow: inset 5px 5px 10px rgba(0,0,0,0.1),
                        inset -5px -5px 10px rgba(255,255,255,0.8),
                        0 0 0 2px rgba(102, 126, 234, 0.5) !important;
            transform: perspective(500px) translateZ(5px);
        }
        
        /* 3D AI Answer section */
        .ai-answer {
            animation: fadeIn 0.8s ease-out;
            background: linear-gradient(145deg, #f0f2ff, #e8ecff);
            border-radius: 16px;
            padding: 1.8rem 2rem;
            border: none;
            margin: 1.5rem 0;
            transform-style: preserve-3d;
            transform: perspective(1000px) rotateX(1deg);
            box-shadow: 8px 8px 20px rgba(0,0,0,0.1),
                       -8px -8px 20px rgba(255,255,255,0.8);
            transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
        }
        
        .ai-answer::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 5px;
            background: linear-gradient(90deg, #667eea, #764ba2);
        }
        
        .ai-answer:hover {
            transform: perspective(1000px) rotateX(0.5deg) translateY(-5px);
            box-shadow: 12px 12px 30px rgba(0,0,0,0.15),
                       -12px -12px 30px rgba(255,255,255,0.8);
        }
        
        /* 3D Document cards */
        .document-card {
            border-radius: 16px !important;
            border: none !important;
            background: #f8f9ff;
            transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
            margin: 1.2rem 0;
            padding: 1.5rem;
            transform: perspective(1000px) translateZ(0);
            box-shadow: 6px 6px 15px rgba(0,0,0,0.1),
                       -6px -6px 15px rgba(255,255,255,0.8);
            position: relative;
            overflow: hidden;
        }
        
        .document-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: linear-gradient(to bottom, #667eea, #764ba2);
            transition: all 0.4s ease;
        }
        
        .document-card:hover {
            transform: perspective(1000px) translateZ(10px) rotateX(1deg);
            box-shadow: 10px 10px 25px rgba(0,0,0,0.15),
                       -10px -10px 25px rgba(255,255,255,0.8);
        }
        
        .document-card:hover::before {
            width: 6px;
            box-shadow: 2px 0 10px rgba(102, 126, 234, 0.5);
        }
        
        /* 3D Sidebar */
        .css-1d391kg {
            background: linear-gradient(145deg, #f0f2ff, #e8ecff) !important;
            box-shadow: 5px 0 20px rgba(0,0,0,0.1) !important;
            transform-style: preserve-3d;
            transform: perspective(1000px) rotateY(-2deg) translateX(-5px);
        }
        
        /* Text colors and typography */
        .stMarkdown {
            color: #2d3748 !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Animated Header with new styling
    st.markdown("""
    <div class="header-container">
        <h1 style="margin: 0; font-size: 2.5rem;">🧬 HealthMiner</h1>
        <p style="font-size: 1.2rem; opacity: 0.95; margin: 0.5rem 0 0;">AI-powered RAG system for exploring clinical trial data</p>
        <p style="opacity: 0.85; margin: 0.5rem 0 0; font-size: 0.95rem;">Upload PDFs, ask questions, and get intelligent answers from your research documents</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📚 Document Management")
        
        # Fetch documents on first load
        if not st.session_state.documents:
            fetch_documents()
        
        # Initialize sample data button
        st.subheader("🎯 Quick Start")
        if st.button("🔄 Load Sample Clinical Trials", use_container_width=True):
            initialize_sample_data()
        
        st.markdown("---")
        
        # File upload
        st.subheader("📤 Upload Documents")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload clinical trial PDFs to expand your knowledge base"
        )
        
        if uploaded_file is not None:
            if st.button("🚀 Upload & Process", use_container_width=True):
                upload_document(uploaded_file)
        
        st.markdown("---")
        
        # Document list
        st.subheader(f"📋 Documents ({len(st.session_state.documents)})")
        
        if st.session_state.documents:
            for doc in st.session_state.documents:
                with st.container():
                    st.write(f"📄 **{doc['title'][:40]}{'...' if len(doc['title']) > 40 else ''}**")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.caption(f"📊 {len(doc.get('chunks', []))} chunks")
                    with col2:
                        if st.button("🗑️", key=f"del_{doc['id']}", help="Delete document"):
                            delete_document(doc['id'])
                            st.rerun()
                    st.markdown("---")
        else:
            st.info("📝 No documents uploaded yet. Use sample data or upload PDFs to get started.")
        
        # Refresh button
        if st.button("🔄 Refresh List", use_container_width=True):
            fetch_documents()
            st.rerun()

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💭 Ask Questions")
        
        # Query input
        query = st.text_area(
            "What would you like to know about the clinical trials?",
            placeholder="Ask about treatments, side effects, patient outcomes, trial phases, or any other aspect of the research data...",
            height=100,
            help="Enter your question about the clinical trial data"
        )
        
        # Query button
        if st.button("🔍 Search Knowledge Base", use_container_width=True, type="primary"):
            if query.strip():
                result = query_documents(query)
                
                if result:
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'query': query,
                        'answer': result['answer'],
                        'sources': result['sources'],
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    # Display answer
                    st.markdown("---")
                    st.markdown("<h2 style='color: black;'>🤖 AI Answer</h2>", unsafe_allow_html=True)
                    
                    # AI Answer with enhanced styling
                    st.markdown(
                        f'<div class="ai-answer"><div style="color: #1a1a1a; line-height: 1.7;">{result["answer"]}</div></div>', 
                        unsafe_allow_html=True
                    )
                    
                    # Sources
                    if result['sources']:
                        st.subheader("📚 Sources")
                        for source in result['sources']:
                            st.badge(source)
                    
                    st.markdown("---")
            else:
                st.warning("⚠️ Please enter a question")
    
    with col2:
        st.header("💬 Recent Queries")
        
        if st.session_state.chat_history:
            # Reverse to show most recent first
            for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):
                with st.expander(f"Q: {chat['query'][:50]}{'...' if len(chat['query']) > 50 else ''}", expanded=(i == 0)):
                    st.write("**Answer:**")
                    st.write(chat['answer'][:200] + "..." if len(chat['answer']) > 200 else chat['answer'])
                    st.caption(f"🕒 {chat['timestamp']}")
                    if chat['sources']:
                        st.caption(f"📚 Sources: {', '.join(chat['sources'])}")
        else:
            st.info("💡 Your recent questions and answers will appear here")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 1rem;">
        <p>🚀 Powered by Groq (Mixtral) + HuggingFace Sentence Transformers + MongoDB</p>
        <p>Built with FastAPI + Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()