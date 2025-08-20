import React, { useState, useEffect } from "react";
import "./App.css";
import axios from "axios";
import { Upload, FileText, Search, MessageCircle, Brain, Database, Trash2, RefreshCw } from "lucide-react";
import { Button } from "./components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "./components/ui/card";
import { Input } from "./components/ui/input";
import { Textarea } from "./components/ui/textarea";
import { Badge } from "./components/ui/badge";
import { toast, Toaster } from "sonner";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [documents, setDocuments] = useState([]);
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [relevantChunks, setRelevantChunks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [initializingData, setInitializingData] = useState(false);

  useEffect(() => {
    fetchDocuments();
    fetchChatHistory();
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await axios.get(`${API}/documents`);
      setDocuments(response.data);
    } catch (error) {
      console.error("Error fetching documents:", error);
      toast.error("Failed to fetch documents");
    }
  };

  const fetchChatHistory = async () => {
    try {
      const response = await axios.get(`${API}/chat-history`);
      setChatHistory(response.data);
    } catch (error) {
      console.error("Error fetching chat history:", error);
    }
  };

  const initializeSampleData = async () => {
    setInitializingData(true);
    try {
      const response = await axios.post(`${API}/initialize-sample-data`);
      toast.success(response.data.message);
      fetchDocuments();
    } catch (error) {
      console.error("Error initializing sample data:", error);
      toast.error("Failed to initialize sample data");
    } finally {
      setInitializingData(false);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.name.endsWith('.pdf')) {
      toast.error("Please upload a PDF file");
      return;
    }

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      toast.success(`Document "${response.data.title}" uploaded successfully!`);
      fetchDocuments();
      event.target.value = '';
    } catch (error) {
      console.error("Error uploading file:", error);
      toast.error(error.response?.data?.detail || "Failed to upload document");
    } finally {
      setUploading(false);
    }
  };

  const handleQuery = async () => {
    if (!query.trim()) {
      toast.error("Please enter a question");
      return;
    }

    setLoading(true);
    setAnswer("");
    setSources([]);
    setRelevantChunks([]);

    try {
      const response = await axios.post(`${API}/query`, { query });
      setAnswer(response.data.answer);
      setSources(response.data.sources);
      setRelevantChunks(response.data.relevant_chunks);
      fetchChatHistory();
      toast.success("Query processed successfully!");
    } catch (error) {
      console.error("Error processing query:", error);
      toast.error(error.response?.data?.detail || "Failed to process query");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleQuery();
    }
  };

  const deleteDocument = async (documentId) => {
    try {
      await axios.delete(`${API}/documents/${documentId}`);
      toast.success("Document deleted successfully");
      fetchDocuments();
    } catch (error) {
      console.error("Error deleting document:", error);
      toast.error("Failed to delete document");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        
        {/* Header */}
        <header className="text-center mb-12">
          <div className="inline-flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl shadow-lg">
              <Brain className="h-8 w-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Clinical Trial Explorer
            </h1>
          </div>
          <p className="text-xl text-slate-600 max-w-3xl mx-auto">
            AI-powered RAG system for exploring clinical trial data. Upload PDFs, ask questions, and get intelligent answers from your clinical research documents.
          </p>
        </header>

        <div className="grid lg:grid-cols-3 gap-8">
          
          {/* Left Column - Upload & Documents */}
          <div className="lg:col-span-1 space-y-6">
            
            {/* Sample Data Initialization */}
            <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Database className="h-5 w-5 text-blue-600" />
                  Sample Data
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-slate-600 mb-4">
                  Start exploring with pre-loaded clinical trial data from real research studies.
                </p>
                <Button 
                  onClick={initializeSampleData}
                  disabled={initializingData}
                  className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-medium rounded-xl h-11 shadow-lg"
                >
                  {initializingData ? (
                    <>
                      <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                      Loading...
                    </>
                  ) : (
                    <>
                      <Database className="h-4 w-4 mr-2" />
                      Load Sample Trials
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            {/* File Upload */}
            <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Upload className="h-5 w-5 text-green-600" />
                  Upload Documents
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <p className="text-sm text-slate-600">
                    Upload PDF files containing clinical trial data to expand your knowledge base.
                  </p>
                  <div className="relative">
                    <input
                      type="file"
                      accept=".pdf"
                      onChange={handleFileUpload}
                      disabled={uploading}
                      className="hidden"
                      id="file-upload"
                    />
                    <label
                      htmlFor="file-upload"
                      className={`flex items-center justify-center w-full h-32 border-2 border-dashed rounded-xl cursor-pointer transition-all ${
                        uploading 
                          ? 'border-gray-300 bg-gray-50 cursor-not-allowed' 
                          : 'border-green-300 bg-green-50 hover:bg-green-100 hover:border-green-400'
                      }`}
                    >
                      {uploading ? (
                        <div className="text-center">
                          <RefreshCw className="h-8 w-8 text-gray-400 mx-auto mb-2 animate-spin" />
                          <p className="text-gray-500">Processing...</p>
                        </div>
                      ) : (
                        <div className="text-center">
                          <Upload className="h-8 w-8 text-green-600 mx-auto mb-2" />
                          <p className="text-green-700 font-medium">Drop PDF files here or click to upload</p>
                          <p className="text-xs text-green-600 mt-1">Only PDF files are supported</p>
                        </div>
                      )}
                    </label>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Document List */}
            <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <FileText className="h-5 w-5 text-blue-600" />
                  Documents ({documents.length})
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3 max-h-80 overflow-y-auto">
                  {documents.length === 0 ? (
                    <p className="text-slate-500 text-sm text-center py-4">
                      No documents uploaded yet. Use the sample data or upload PDFs to get started.
                    </p>
                  ) : (
                    documents.map((doc) => (
                      <div key={doc.id} className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-slate-900 truncate">{doc.title}</p>
                          <p className="text-xs text-slate-500">{doc.chunks?.length || 0} chunks</p>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => deleteDocument(doc.id)}
                          className="text-red-500 hover:text-red-700 hover:bg-red-50"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    ))
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Right Column - Query Interface */}
          <div className="lg:col-span-2 space-y-6">
            
            {/* Query Input */}
            <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Search className="h-5 w-5 text-purple-600" />
                  Ask Questions
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <Textarea
                    placeholder="Ask questions about clinical trials, treatments, side effects, patient outcomes, or any other aspect of the research data..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyPress={handleKeyPress}
                    rows={3}
                    className="resize-none border-slate-200 focus:ring-2 focus:ring-purple-500 focus:border-purple-500 rounded-xl"
                  />
                  <Button 
                    onClick={handleQuery} 
                    disabled={loading || !query.trim()}
                    className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-medium rounded-xl h-12 shadow-lg disabled:from-gray-400 disabled:to-gray-500"
                  >
                    {loading ? (
                      <>
                        <RefreshCw className="h-5 w-5 mr-2 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Search className="h-5 w-5 mr-2" />
                        Search Knowledge Base
                      </>
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Answer Display */}
            {answer && (
              <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <MessageCircle className="h-5 w-5 text-green-600" />
                    AI Answer
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="prose prose-slate max-w-none">
                    <div className="bg-gradient-to-r from-green-50 to-blue-50 p-6 rounded-xl border border-green-200">
                      <p className="text-slate-800 leading-relaxed whitespace-pre-wrap">{answer}</p>
                    </div>
                    
                    {sources.length > 0 && (
                      <div className="mt-6">
                        <h4 className="font-semibold text-slate-800 mb-3">Sources:</h4>
                        <div className="flex flex-wrap gap-2">
                          {sources.map((source, index) => (
                            <Badge key={index} variant="outline" className="bg-blue-50 text-blue-800 border-blue-200">
                              {source}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Recent Chat History */}
            {chatHistory.length > 0 && (
              <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <MessageCircle className="h-5 w-5 text-slate-600" />
                    Recent Queries
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4 max-h-64 overflow-y-auto">
                    {chatHistory.slice(0, 5).map((chat, index) => (
                      <div key={chat.id || index} className="p-4 bg-slate-50 rounded-lg">
                        <p className="text-sm font-medium text-slate-800 mb-2">{chat.query}</p>
                        <p className="text-xs text-slate-600 truncate">{chat.answer}</p>
                        <p className="text-xs text-slate-400 mt-2">
                          {new Date(chat.created_at).toLocaleString()}
                        </p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
      
      <Toaster position="top-right" richColors />
    </div>
  );
}

export default App;