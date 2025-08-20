import requests
import sys
import json
import time
from datetime import datetime

class ClinicalTrialAPITester:
    def __init__(self, base_url="https://clinical-finder-1.preview.emergentagent.com/api"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.created_document_ids = []

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}" if endpoint else self.base_url
        headers = {'Content-Type': 'application/json'} if not files else {}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                if files:
                    response = requests.post(url, files=files)
                else:
                    response = requests.post(url, json=data, headers=headers)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response: {json.dumps(response_data, indent=2)[:200]}...")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text}")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_root_endpoint(self):
        """Test the root API endpoint"""
        return self.run_test("Root Endpoint", "GET", "", 200)

    def test_initialize_sample_data(self):
        """Test sample data initialization"""
        success, response = self.run_test(
            "Initialize Sample Data",
            "POST",
            "initialize-sample-data",
            200
        )
        if success:
            print(f"   Sample data message: {response.get('message', 'No message')}")
            if 'documents' in response:
                print(f"   Created documents: {response['documents']}")
        return success

    def test_get_documents(self):
        """Test getting all documents"""
        success, response = self.run_test(
            "Get Documents",
            "GET",
            "documents",
            200
        )
        if success and isinstance(response, list):
            print(f"   Found {len(response)} documents")
            for doc in response[:2]:  # Show first 2 documents
                print(f"   - {doc.get('title', 'No title')} (ID: {doc.get('id', 'No ID')})")
                if doc.get('id'):
                    self.created_document_ids.append(doc['id'])
        return success

    def test_query_system(self):
        """Test the RAG query system"""
        test_queries = [
            "What is pembrolizumab used for?",
            "What are the side effects of trastuzumab?",
            "Tell me about CAR-T cell therapy results"
        ]
        
        all_passed = True
        for query in test_queries:
            print(f"\n   Testing query: '{query}'")
            success, response = self.run_test(
                f"Query: {query[:30]}...",
                "POST",
                "query",
                200,
                data={"query": query}
            )
            
            if success:
                answer = response.get('answer', '')
                sources = response.get('sources', [])
                chunks = response.get('relevant_chunks', [])
                
                print(f"   Answer length: {len(answer)} characters")
                print(f"   Sources: {sources}")
                print(f"   Relevant chunks: {len(chunks)}")
                
                if len(answer) < 50:
                    print(f"   ⚠️  Warning: Answer seems too short")
                    all_passed = False
                    
                if not sources:
                    print(f"   ⚠️  Warning: No sources returned")
                    all_passed = False
            else:
                all_passed = False
                
            # Add delay between queries to avoid rate limiting
            time.sleep(2)
        
        return all_passed

    def test_chat_history(self):
        """Test chat history endpoint"""
        return self.run_test("Get Chat History", "GET", "chat-history", 200)

    def test_delete_document(self):
        """Test document deletion"""
        if not self.created_document_ids:
            print("   No documents to delete, skipping test")
            return True
            
        # Try to delete the first document
        doc_id = self.created_document_ids[0]
        success, response = self.run_test(
            f"Delete Document {doc_id}",
            "DELETE",
            f"documents/{doc_id}",
            200
        )
        return success

    def test_query_without_documents(self):
        """Test query when no documents exist (should handle gracefully)"""
        # First delete all documents if any exist
        for doc_id in self.created_document_ids[1:]:  # Skip first one as it's already deleted
            self.run_test(f"Delete Document {doc_id}", "DELETE", f"documents/{doc_id}", 200)
        
        # Now try to query
        success, response = self.run_test(
            "Query Without Documents",
            "POST",
            "query",
            404,  # Should return 404 when no documents exist
            data={"query": "What is pembrolizumab?"}
        )
        return success

def main():
    print("🧪 Starting Clinical Trial API Tests")
    print("=" * 50)
    
    tester = ClinicalTrialAPITester()
    
    # Test sequence
    tests = [
        ("Root Endpoint", tester.test_root_endpoint),
        ("Initialize Sample Data", tester.test_initialize_sample_data),
        ("Get Documents", tester.test_get_documents),
        ("RAG Query System", tester.test_query_system),
        ("Chat History", tester.test_chat_history),
        ("Delete Document", tester.test_delete_document),
        ("Query Without Documents", tester.test_query_without_documents),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            test_func()
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
            tester.tests_run += 1
        
        # Add delay between test groups
        time.sleep(1)
    
    # Print final results
    print(f"\n{'='*50}")
    print(f"📊 Final Results:")
    print(f"   Tests Run: {tester.tests_run}")
    print(f"   Tests Passed: {tester.tests_passed}")
    print(f"   Success Rate: {(tester.tests_passed/tester.tests_run)*100:.1f}%")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())