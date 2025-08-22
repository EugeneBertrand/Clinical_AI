import os
import sys
from dotenv import load_dotenv

# Redirect stdout to a file for debugging
sys.stdout = open('debug_output.txt', 'w')

# Print environment variables
print("=== Environment Variables ===")
for key, value in os.environ.items():
    if 'MONGO' in key or 'PYTHON' in key or 'PATH' in key or 'STREAMLIT' in key:
        print(f"{key}: {value}")
print("==========================\n")

# Now run the Streamlit app
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.system('streamlit run app.py')
