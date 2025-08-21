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
if 'deleted_docs' not in st.session_state:
    st.session_state.deleted_docs = set()

# Helper functions
def fetch_documents():
    try:
        response = requests.get(f"{BACKEND_URL}/documents")
        if response.status_code == 200:
            # Filter out any documents that have been marked as deleted
            all_docs = response.json()
            st.session_state.documents = [
                doc for doc in all_docs 
                if str(doc.get('id', doc.get('_id', ''))) not in st.session_state.deleted_docs
            ]
    except Exception as e:
        st.error(f"Error fetching documents: {str(e)}")
        st.session_state.documents = []

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
    # Create a placeholder for the loading message
    loading_placeholder = st.empty()
    
    try:
        # Show loading message
        loading_placeholder.markdown("<p style='color: black;'>🔍 Analyzing your question...</p>", unsafe_allow_html=True)
        
        # Make the API call
        response = requests.post(
            f"{BACKEND_URL}/query",
            json={"query": query},
            headers={"Content-Type": "application/json"}
        )
        
        # Clear the loading message
        loading_placeholder.empty()
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            error_msg = response.json().get('detail', 'Unknown error')
            st.error(f"❌ Query failed: {error_msg}")
            return None
    except Exception as e:
        # Clear the loading message in case of error
        loading_placeholder.empty()
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
        # Add to deleted docs set first to prevent re-fetching
        st.session_state.deleted_docs.add(str(doc_id))
        
        # Try to delete from backend
        response = requests.delete(f"{BACKEND_URL}/documents/{doc_id}")
        
        # Update the UI immediately
        if 'documents' in st.session_state:
            st.session_state.documents = [
                doc for doc in st.session_state.documents 
                if str(doc.get('id', doc.get('_id', ''))) != str(doc_id)
            ]
            
        return True
    except Exception as e:
        # Even if there's an error, we've already marked it as deleted
        return True

# Main app
def main():
    # Custom CSS for animations and styling
    st.markdown("""
    <style>
        /* Base styles */
        .stApp {
            background-color: #f8f9fa !important;
        }
        
        /* Sidebar styles */
        section[data-testid="stSidebar"] {
            background-color: white !important;
            /* Custom scrollbar for Webkit browsers (Chrome, Safari, etc.) */
            scrollbar-width: thin;
            scrollbar-color: #666666 #f0f0f0;
        }
        
        section[data-testid="stSidebar"] > div:first-child {
            background-color: white !important;
            padding: 2rem 1.5rem !important;
        }
        
        /* Make the top navigation bar white */
        .stApp > header {
            background-color: white !important;
        }
        
        /* Remove the black border from the navigation bar */
        .stApp > header::before {
            display: none !important;
        }
        
        /* Style the navigation menu items */
        .stApp > header .stAppHeader {
            background: white !important;
            color: #1e293b !important;
        }
        
        /* Custom scrollbar for Webkit browsers (Chrome, Safari, etc.) */
        /* Main scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        /* Track */
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 5px;
        }
        
        /* Handle */
        ::-webkit-scrollbar-thumb {
            background: #555;
            border-radius: 5px;
        }
        
        /* Handle on hover */
        ::-webkit-scrollbar-thumb:hover {
            background: #333;
        }
        
        /* Sidebar scrollbar */
        section[data-testid="stSidebar"]::-webkit-scrollbar {
            width: 8px;
        }
        
        section[data-testid="stSidebar"]::-webkit-scrollbar-track {
            background: #f0f0f0;
            border-radius: 4px;
        }
        
        section[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
            background-color: #666666;
            border-radius: 4px;
        }
        
        section[data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover {
            background-color: #555555;
        }
        
        /* Main content area */
        .main .block-container {
            background: white;
            border-radius: 12px;
            padding: 2rem;
            margin: 1rem auto;
            max-width: 1200px;
        }
        
        /* Alert/Success/Error boxes */
        .stAlert, .stAlert p, .stAlert div[data-testid="stMarkdownContainer"] {
            color: #1f2937 !important;  /* Dark gray for better readability */
        }
        
        .stAlert a {
            color: #1d4ed8 !important;  /* Blue for links */
            text-decoration: underline;
        }
        
        .stAlert .stAlert-icon {
            color: #1f2937 !important;
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
        
        /* Light Header */
        .header-container {
            text-align: center;
            padding: 1.5rem 0;
            background: white !important;
            border-radius: 0;
            margin: 0 0 1.5rem 0;
            color: #1e293b !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
            border: none !important;
            border-bottom: 1px solid #e2e8f0 !important;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        /* Remove gradient overlay */
        .header-container::before {
            display: none;
        }
        
        .header-container:hover {
            transform: none;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
        }
        
        /* Light Buttons */
        .stButton>button {
            border: 1px solid #e2e8f0 !important;
            border-radius: 10px !important;
            background: white !important;
            color: #475569 !important;
            font-weight: 500 !important;
            letter-spacing: 0.3px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            transition: all 0.2s ease !important;
            padding: 0.5rem 1.25rem !important;
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
        
        /* 3D Text Areas - Question Box with Gradient */
        .stTextArea>div>div>textarea {
            border-radius: 12px !important;
            min-height: 150px !important;
            border: none !important;
            background: linear-gradient(145deg, #ffffff, #f8f9ff) !important;
            box-shadow: 
                6px 6px 12px rgba(0,0,0,0.08),
                -6px -6px 12px rgba(255,255,255,0.8),
                inset 1px 1px 2px rgba(255,255,255,0.8) !important;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
            padding: 1.5rem !important;
            color: #1f2937 !important;
            font-size: 1rem !important;
            line-height: 1.6 !important;
            transform: perspective(1000px) rotateX(1deg) translateZ(0);
            position: relative;
            z-index: 1;
        }
        
        /* Add 3D edge effect */
        .stTextArea>div>div>textarea::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.7);
            pointer-events: none;
            z-index: -1;
        }
        
        .stTextArea>div>div>textarea:focus {
            outline: none !important;
            box-shadow: inset 5px 5px 10px rgba(0,0,0,0.1),
                        inset -5px -5px 10px rgba(255,255,255,0.8),
                        0 0 0 2px rgba(102, 126, 234, 0.5) !important;
            transform: perspective(500px) translateZ(5px);
        }
        
        /* Light AI Answer section */
        .ai-answer {
            background: white;
            position: relative;
            overflow: hidden;
        }
        
        /* Question input area */
        .stTextArea textarea {
            border: none !important;
            border-radius: 8px;
            padding: 12px 0;
            box-shadow: none !important;
            transition: all 0.3s ease;
            caret-color: #1e293b; /* Match title color */
            cursor: text;
            font-size: 1.1rem;
            color: #1e293b !important; /* Match title color */
            border-bottom: 2px solid #e2e8f0 !important;
            background: transparent !important;
        }
        
        /* Ensure the text cursor is visible in the input */
        .stTextArea textarea:focus {
            border-bottom-color: #4f46e5 !important;
            box-shadow: none !important;
            outline: none;
        }
        
        /* Remove the container border and shadow */
        .stTextArea>div>div {
            border: none !important;
            box-shadow: none !important;
            background: transparent !important;
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
        
        /* Light Document cards */
        .document-card {
            border-radius: 10px !important;
            border: 1px solid #e2e8f0 !important;
            background: white;
            transition: all 0.2s ease;
            margin: 0.75rem 0;
            padding: 1.25rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.03);
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
        
        /* Light 3D Sidebar */
        .css-1d391kg {
            background: linear-gradient(145deg, #ffffff, #f8fafc) !important;
            border-right: 1px solid rgba(203, 213, 225, 0.5) !important;
            box-shadow: 8px 0 30px -10px rgba(0, 0, 0, 0.05),
                       -2px 0 10px -8px rgba(0, 0, 0, 0.02) !important;
            transform: perspective(1500px) rotateY(-5deg) translateX(-10px);
            transform-origin: right center;
            transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
            border-radius: 0 20px 20px 0 !important;
            padding: 1rem 0.5rem 1rem 1rem !important;
            backdrop-filter: blur(8px);
            border-left: 1px solid rgba(255, 255, 255, 0.7) !important;
        }
        
        /* Sidebar hover effect */
        .css-1d391kg:hover {
            transform: perspective(1500px) rotateY(0deg) translateX(0);
            box-shadow: 15px 0 40px -15px rgba(0, 0, 0, 0.08),
                       -5px 0 15px -8px rgba(0, 0, 0, 0.03) !important;
        }
        
        /* Sidebar header */
        .css-1d391kg .css-1a32fsj {
            color: #334155 !important;
            font-weight: 600 !important;
            margin-bottom: 1.5rem !important;
            text-align: center;
            padding: 0.5rem 1rem;
            background: rgba(255, 255, 255, 0.7);
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
            border: 1px solid rgba(203, 213, 225, 0.3);
        }
        
        /* Sidebar widgets */
        .st-eb, .st-ec, .st-ed {
            background: rgba(255, 255, 255, 0.8) !important;
            border-radius: 12px !important;
            padding: 0.75rem 1rem !important;
            margin: 0.75rem 0 !important;
            border: 1px solid rgba(203, 213, 225, 0.5) !important;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.03) !important;
            transition: all 0.3s ease !important;
        }
        
        .st-eb:hover, .st-ec:hover, .st-ed:hover {
            transform: translateX(5px) !important;
            box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.05) !important;
            border-color: #cbd5e1 !important;
        }
        
        /* Sidebar buttons */
        .st-eb button, .st-ec button, .st-ed button {
            background: white !important;
            color: #475569 !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
        }
        
        .st-eb button:hover, .st-ec button:hover, .st-ed button:hover {
            background: #f8fafc !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
        }
        
        /* Text colors and typography */
        body, .stMarkdown {
            color: #334155 !important;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #1e293b !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em;
        }
        
        /* Light theme form elements */
        .stTextInput>div>div>input, 
        .stSelectbox>div>div>div>div {
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            background: black !important;
        }
        
        /* Recent Queries Section - Make everything visible */
        [data-testid="stExpander"] {
            background-color: #1a1a1a !important;
            border: 1px solid #1e90ff !important;
            border-radius: 8px !important;
            margin: 8px 0 !important;
            padding: 8px !important;
        }
        
        [data-testid="stExpander"] > div {
            background: transparent !important;
        }
        
        [data-testid="stExpander"] * {
            color: white !important;
            background: transparent !important;
        }
        
        [data-testid="stExpander"]:hover {
            box-shadow: 0 0 0 1px #1e90ff !important;
        }
        
        /* Make sure the expander content is visible */
        [data-testid="stExpander"][aria-expanded="true"] {
            background-color: #1a1a1a !important;
        }
        
        /* Style the info message */
        .stAlert {
            background-color: #1a1a1a !important;
            color: white !important;
        }
        
        /* 3D Document Cards - Recent Queries */
        .document-card {
            background: linear-gradient(145deg, #ffffff, #f8f9ff) !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 1.25rem 1.5rem !important;
            margin: 0.75rem 0 !important;
            box-shadow: 
                4px 4px 8px rgba(0,0,0,0.05),
                -4px -4px 8px rgba(255,255,255,0.8) !important;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
            transform: perspective(1000px) rotateX(1deg) translateZ(0);
            position: relative;
            border: 1px solid rgba(255,255,255,0.7) !important;
        }
        
        .document-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.8);
            pointer-events: none;
            z-index: 1;
        }
        
        .document-card:hover {
            transform: perspective(1000px) translateZ(10px) rotateX(0.5deg) !important;
            box-shadow: 
                6px 6px 16px rgba(0,0,0,0.08),
                -6px -6px 16px rgba(255,255,255,0.9) !important;
            background: linear-gradient(145deg, #ffffff, #f0f4ff) !important;
        }
        
        /* Make sure all text in cards is visible */
        .document-card p, 
        .document-card div, 
        .document-card span {
            color: #1f2937 !important;
            text-shadow: 0 1px 1px rgba(255,255,255,0.8) !important;
        }
        
        .stButton>button:hover {
            background: #f8fafc !important;
            border-color: #cbd5e1 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05) !important;
        }
        
        .stTextArea>div>div>textarea:focus {
            transform: perspective(1000px) rotateX(0.5deg) translateZ(10px) !important;
            box-shadow: 
                8px 8px 16px rgba(0,0,0,0.1),
                -8px -8px 16px rgba(255,255,255,0.9),
                inset 2px 2px 4px rgba(0,0,0,0.05),
                0 0 0 2px rgba(99, 102, 241, 0.3) !important;
            outline: none !important;
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

    # Document Management Section in Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        '>
            <h3 style='color: #1f2937; margin: 0 0 1rem 0;'>📚 Document Management</h3>
        """, unsafe_allow_html=True)
        
        # Fetch documents on first load
        if not st.session_state.documents:
            fetch_documents()
        
        # Initialize sample data button
        if st.button("🔄 Load Sample Clinical Trials", use_container_width=True):
            initialize_sample_data()
        
        st.markdown("<hr style='margin: 1rem 0;' />", unsafe_allow_html=True)
        
        # File upload
        st.markdown("<h4 style='color: #1f2937; margin-bottom: 0.5rem;'>📤 Upload Documents</h4>", unsafe_allow_html=True)
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
                    doc_id = doc.get('id', doc.get('_id', ''))  # Handle both 'id' and '_id' fields
                    doc_title = doc.get('title', 'Untitled Document')
                    
                    st.write(f"📄 **{doc_title[:40]}{'...' if len(doc_title) > 40 else ''}**")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.caption(f"📊 {len(doc.get('chunks', []))} chunks")
                    with col2:
                        if st.button("🗑️", key=f"del_{doc_id}", help="Delete document"):
                            if delete_document(doc_id):
                                # Force a rerun to update the UI
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
                    st.markdown("<h2 style='color: white;'>🤖 AI Answer</h2>", unsafe_allow_html=True)
                    
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
            st.markdown("""
            <style>
                .stAlert {
                    color: white !important;
                }
                .stAlert div[data-testid="stMarkdownContainer"] p {
                    color: white !important;
                }
            </style>
            """, unsafe_allow_html=True)
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