from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime
import PyPDF2
import io
from sentence_transformers import SentenceTransformer
import numpy as np
from groq import Groq
import re


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Initialize models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
groq_client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Models
class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    chunks: List[str]
    embeddings: List[List[float]]
    created_at: datetime = Field(default_factory=datetime.utcnow)

class DocumentResponse(BaseModel):
    id: str
    title: str
    content: str
    chunks: List[str]
    created_at: datetime

class DocumentCreate(BaseModel):
    title: str
    content: str

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    relevant_chunks: List[str]

class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    query: str
    answer: str
    sources: List[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Helper functions
def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        logging.error(f"Error extracting PDF text: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a_np = np.array(a)
    b_np = np.array(b)
    return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))

# Sample clinical trial data
SAMPLE_CLINICAL_TRIALS = [
    {
        "title": "Phase II Trial of Pembrolizumab in Advanced Melanoma",
        "content": """
        Background: Pembrolizumab is an anti-PD-1 monoclonal antibody that has shown promising results in melanoma treatment.
        
        Objective: To evaluate the efficacy and safety of pembrolizumab in patients with advanced melanoma who have not received prior immunotherapy.
        
        Methods: This is a multi-center, single-arm, open-label Phase II clinical trial. Patients with unresectable Stage III or Stage IV melanoma were enrolled. Primary endpoint was overall response rate (ORR). Secondary endpoints included progression-free survival (PFS), overall survival (OS), and safety.
        
        Results: 150 patients were enrolled. The overall response rate was 42% (95% CI: 34-50%). Median progression-free survival was 8.2 months. Most common adverse events were fatigue (65%), skin rash (45%), and diarrhea (38%). Grade 3-4 immune-related adverse events occurred in 15% of patients.
        
        Conclusions: Pembrolizumab demonstrated significant antitumor activity in advanced melanoma with a manageable safety profile. These results support further investigation in combination therapy approaches.
        
        Trial Registration: NCT02345678
        """
    },
    {
        "title": "Phase III Randomized Trial of Trastuzumab in HER2-Positive Breast Cancer",
        "content": """
        Background: HER2-positive breast cancer accounts for approximately 20% of all breast cancers and has historically been associated with poor prognosis.
        
        Objective: To compare the efficacy of trastuzumab plus chemotherapy versus chemotherapy alone in patients with HER2-positive early breast cancer.
        
        Methods: This randomized, controlled, phase III trial enrolled 3,222 women with HER2-positive early breast cancer. Patients were randomly assigned to receive either standard chemotherapy plus trastuzumab for one year or chemotherapy alone. Primary endpoint was disease-free survival (DFS).
        
        Results: At a median follow-up of 4.1 years, disease-free survival was significantly improved in the trastuzumab group (HR 0.64, 95% CI 0.54-0.76, P<0.001). The 4-year disease-free survival rate was 86% in the trastuzumab group versus 77% in the control group. Cardiac toxicity was observed in 2.2% of trastuzumab-treated patients.
        
        Conclusions: The addition of one year of trastuzumab to standard chemotherapy significantly improves outcomes for women with HER2-positive early breast cancer. The benefit outweighs the cardiac risk.
        
        Trial Registration: NCT00045032
        """
    },
    {
        "title": "Phase I Dose-Escalation Study of CAR-T Cell Therapy in Relapsed B-cell Lymphoma",
        "content": """
        Background: Chimeric antigen receptor T-cell (CAR-T) therapy represents a novel immunotherapy approach for hematologic malignancies.
        
        Objective: To determine the maximum tolerated dose (MTD) and assess the safety and preliminary efficacy of CD19-targeted CAR-T cells in patients with relapsed or refractory B-cell lymphoma.
        
        Methods: This phase I dose-escalation study enrolled patients with CD19-positive B-cell lymphoma who had failed at least two prior therapies. Patients received lymphodepletion chemotherapy followed by a single infusion of CAR-T cells at escalating dose levels.
        
        Results: 25 patients were treated across four dose levels. Dose-limiting toxicities included cytokine release syndrome (CRS) and neurotoxicity. The MTD was established at 2×10^6 CAR-T cells/kg. Overall response rate was 68%, with 52% achieving complete response. Severe CRS (Grade 3-4) occurred in 24% of patients.
        
        Conclusions: CD19 CAR-T cell therapy showed promising efficacy in heavily pretreated B-cell lymphoma patients. CRS and neurotoxicity are the main safety concerns that require careful monitoring and management.
        
        Trial Registration: NCT02348216
        """
    }
]

# Routes
@api_router.get("/")
async def root():
    return {"message": "RAG Clinical Trial Explorer API"}

@api_router.post("/upload", response_model=Document)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    content = await file.read()
    text = extract_text_from_pdf(content)
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")
    
    # Clean the text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Chunk the text
    chunks = chunk_text(text)
    
    # Generate embeddings for each chunk
    embeddings = embedding_model.encode(chunks).tolist()
    
    # Create document
    document = Document(
        title=file.filename,
        content=text,
        chunks=chunks,
        embeddings=embeddings
    )
    
    # Save to MongoDB
    await db.documents.insert_one(document.dict())
    
    return document

@api_router.post("/initialize-sample-data")
async def initialize_sample_data():
    """Initialize the database with sample clinical trial data"""
    try:
        # Initialize sample documents list
        sample_documents = []
        
        # Try to use MongoDB if available
        try:
            # Check if sample data already exists in MongoDB
            existing_count = await db.documents.count_documents({"title": {"$regex": "Phase.*Trial"}})
            if existing_count > 0:
                return {"message": f"Sample data already exists in database ({existing_count} documents)"}
                
            # Process and save sample data to MongoDB
            documents_created = []
            
            for trial_data in SAMPLE_CLINICAL_TRIALS:
                # Clean the text
                text = re.sub(r'\s+', ' ', trial_data['content']).strip()
                
                # Chunk the text
                chunks = chunk_text(text)
                
                # Generate embeddings for each chunk
                embeddings = embedding_model.encode(chunks).tolist()
                
                # Create document
                document = Document(
                    title=trial_data['title'],
                    content=text,
                    chunks=chunks,
                    embeddings=embeddings
                )
                
                # Add to MongoDB
                await db.documents.insert_one(document.dict())
                documents_created.append(document.title)
            
            return {"message": f"Successfully created {len(documents_created)} sample documents in MongoDB", "documents": documents_created}
            
        except Exception as db_error:
            logging.warning(f"MongoDB operation failed, falling back to in-memory sample data: {db_error}")
            
            # If MongoDB fails, create sample documents in memory
            for trial_data in SAMPLE_CLINICAL_TRIALS:
                text = re.sub(r'\s+', ' ', trial_data['content']).strip()
                chunks = chunk_text(text)
                
                document = DocumentResponse(
                    id=str(uuid.uuid4()),
                    title=trial_data['title'],
                    content=text,
                    chunks=chunks,
                    created_at=datetime.utcnow()
                )
                sample_documents.append(document)
            
            # Store in the in-memory list (for this session only)
            if not hasattr(initialize_sample_data, 'in_memory_documents'):
                initialize_sample_data.in_memory_documents = sample_documents
            
            return {
                "message": "Using in-memory sample data (MongoDB not available)",
                "documents": [doc.title for doc in sample_documents]
            }
    
    except Exception as e:
        logging.error(f"Error initializing sample data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize sample data: {str(e)}")

# Store in-memory documents for the get_documents function
in_memory_documents = []

@api_router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        # Get query embedding
        query_embedding = embedding_model.encode([request.query])[0].tolist()
        
        # Retrieve all documents
        documents = await db.documents.find().to_list(1000)
        
        if not documents:
            raise HTTPException(status_code=404, detail="No documents found. Please upload some clinical trial documents first.")
        
        # Find most relevant chunks
        relevant_chunks = []
        chunk_sources = []
        
        for doc in documents:
            for i, chunk_embedding in enumerate(doc['embeddings']):
                similarity = cosine_similarity(query_embedding, chunk_embedding)
                relevant_chunks.append({
                    'chunk': doc['chunks'][i],
                    'similarity': similarity,
                    'source': doc['title']
                })
        
        # Sort by similarity and take top 5
        relevant_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        top_chunks = relevant_chunks[:5]
        
        # Build context for Groq
        context = "\n\n".join([chunk['chunk'] for chunk in top_chunks])
        sources = list(set([chunk['source'] for chunk in top_chunks]))
        
        # Generate response using Groq
        prompt = f"""You are an expert clinical researcher. Based on the following clinical trial information, answer the user's question accurately and comprehensively.

Context from clinical trials:
{context}

Question: {request.query}

Please provide a detailed answer based on the clinical trial data provided. If the information is insufficient to answer the question, please state that clearly."""

        response = groq_client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        
        # Save chat message
        chat_message = ChatMessage(
            session_id=str(uuid.uuid4()),
            query=request.query,
            answer=answer,
            sources=sources
        )
        await db.chat_messages.insert_one(chat_message.dict())
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            relevant_chunks=[chunk['chunk'] for chunk in top_chunks]
        )
    
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")

@api_router.get("/documents", response_model=List[DocumentResponse])
async def get_documents():
    try:
        # Try to get documents from MongoDB
        documents = await db.documents.find({}, {"embeddings": 0, "_id": 0}).to_list(1000)
        if documents:
            return [DocumentResponse(**doc) for doc in documents]
    except Exception as e:
        print(f"MongoDB error: {e}")
    
    # Return sample data if MongoDB is not available
    return [
        DocumentResponse(
            id="1",
            title="Sample Clinical Trial 1",
            content="This is a sample clinical trial document.",
            chunks=["Sample chunk 1", "Sample chunk 2"],
            created_at=datetime.utcnow()
        ),
        DocumentResponse(
            id="2",
            title="Sample Clinical Trial 2",
            content="Another sample document for testing.",
            chunks=["Another test chunk"],
            created_at=datetime.utcnow()
        )
    ]

@api_router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    result = await db.documents.delete_one({"id": document_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted successfully"}

@api_router.get("/chat-history", response_model=List[ChatMessage])
async def get_chat_history():
    messages = await db.chat_messages.find().sort("created_at", -1).limit(50).to_list(50)
    return [ChatMessage(**msg) for msg in messages]

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()