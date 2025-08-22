import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_groq_connection():
    # Get API key from environment
    api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        print("❌ Error: GROQ_API_KEY not found in environment variables")
        print("Please make sure you have a .env file with GROQ_API_KEY=")
        return False
    
    print(f"🔑 Found GROQ_API_KEY: {api_key[:5]}...{api_key[-5:]}")
    
    # Test API endpoint
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Just say 'Hello, World!'"}
        ],
        "temperature": 0.7,
        "max_tokens": 10
    }
    
    try:
        print("\n🔌 Testing connection to Groq API...")
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=10
        )
        
        print(f"\n📡 Response Status Code: {response.status_code}")
        print("📋 Response Headers:")
        for key, value in response.headers.items():
            print(f"   {key}: {value}")
        
        if response.status_code == 200:
            print("\n✅ Successfully connected to Groq API!")
            print("📄 Response:", response.json())
            return True
        else:
            print(f"\n❌ Error from Groq API (Status {response.status_code}):")
            print(response.text)
            
            # Check for common issues
            if response.status_code == 401:
                print("\n🔒 Authentication failed. Please check your API key.")
            elif response.status_code == 404:
                print("\n🔍 Endpoint not found. The API URL may have changed.")
            elif response.status_code == 429:
                print("\n⚠️  Rate limit exceeded. You may have reached your API quota.")
            
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Failed to connect to Groq API: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Check your internet connection")
        print("2. Verify the API endpoint is correct")
        print("3. Check if Groq service is up (https://status.groq.com/)")
        print("4. Make sure your API key is valid and has sufficient quota")
        return False

if __name__ == "__main__":
    test_groq_connection()
