import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Get the Hugging Face token from the environment variable
hf_token = os.getenv('HUGGINGFACE_TOKEN')

# Check if the token is loaded correctly
if hf_token:
    print("Hugging Face token loaded successfully!")
    print(f"Token: {hf_token}")
else:
    print("Failed to load Hugging Face token. Check your .env file.")