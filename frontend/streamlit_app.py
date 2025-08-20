import streamlit as st
import requests
import json
from datetime import datetime
import io

# Configure page
st.set_page_config(
    page_title="🧬 Clinical Trial Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend URL
BACKEND_URL = "https://clinical-finder-1.preview.emergentagent.com/api"

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
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 1rem; margin-bottom: 2rem; color: white;">
        <h1>🧬 Clinical Trial Explorer</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">AI-powered RAG system for exploring clinical trial data</p>
        <p style="opacity: 0.8;">Upload PDFs, ask questions, and get intelligent answers from your research documents</p>
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
                    st.subheader("🤖 AI Answer")
                    
                    # Answer in a nice container
                    with st.container():
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                                    padding: 1.5rem; border-radius: 0.75rem; border-left: 4px solid #0ea5e9;">
                            <p style="margin: 0; line-height: 1.7; color: #1e293b;">{result['answer']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Sources
                    if result['sources']:
                        st.subheader("📚 Sources")
                        for source in result['sources']:
                            st.badge(source, outline=True)
                    
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