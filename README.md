# Katy ISD Chatbot – High School Passion Project

## Project Overview
Welcome! This project is a GenAI-powered chatbot designed to help answer questions about the Katy ISD website. It uses Retrieval-Augmented Generation (RAG) to give accurate, document-based answers. I created this as a high school passion project to explore modern AI and make school information more accessible for everyone.

## Getting Started

### Use the Right App File
**Please use `app.py` for all development and running the chatbot!**  
This version has the best user interface, more features, and a cleaner design.

### Project Structure

- **Main App (`app.py`)**  
  - Stylish header with colors and borders
  - Sidebar with the Katy ISD logo, a project description, and example questions
  - “Clear chat” button
  - Modern chat interface that’s easy to use (`st.chat_input()`)
  - Proper file encoding to avoid text issues
  - Chat history shows up in order
  - Detailed error handling for a smoother experience

## Tech Stack
- **AI Model:** Mistral-7B-Instruct-v0.1 (from Hugging Face)
- **Embeddings Model:** BAAI/bge-base-en-v1.5
- **Vector Store:** FAISS
- **Frontend:** Streamlit
- **Document Handling:** LangChain

## How to Set Up

1. **Install Requirements**
   Navigate (cd) to the folder of where requirements.txt is located.
   ```bash
   pip install -r ./requirements.txt
   ```

3. **Set Up Your Environment**
   - Make a `.env` file in your project folder:
     ```
     HUGGINGFACE_TOKEN=your_hf_token_here
     ```

4. **Prepare Your Documents**
   These files are scrped from the katyisd domain.
   - Place all PDF files in `website_content/documents/`
   - Place any `.txt` or `.docx` files in the `website_content/` root directory

6. **Add the Logo**
   - Include `katyisd.jpg` for the sidebar logo

7. **Run the Chatbot**
   ```bash
   streamlit run app_mis1.py
   ```

## What Can It Do?
- Answers questions specifically about Katy ISD, using real documents
- Supports PDF, TXT, and DOCX files
- Keeps a persistent vector store for fast, accurate answers
- Attractive, student-friendly UI with a sidebar for guidance and examples
- Tracks which documents are being used as sources
- Uses GPU acceleration if available for faster answers

## Developer Notes
- The vector store will update itself if you swap out the embedding model
- Prompts are formatted for Mistral with `[INST]` tags
- Documents are chunked for better searching (1000 characters with 150-character overlap)
- Retrieves the 10 most relevant chunks for each question
- Response creativity is balanced (temperature: 0.7), up to 512 tokens per answer

## Best Practices

- **Use `app_mis1.py` whenever possible:**  
  It’s the most complete and user-friendly version.
- **Avoid `app_mistral.py` and `app.py`:**  
  They’re older versions and might not have all the new features or the best interface.

## Ideas for the Future
- Clickable citations that link directly to the source document
- Remember conversations across sessions
- Drag-and-drop for uploading new files
- Even friendlier error messages
- Admin panel for managing documents
- Ways to rate and give feedback on chatbot answers

---

**Note:**  
This is a high school-led project to show how modern AI (like RAG, vector databases, and transformer models) can help make school info easier to find and use. Thank you for checking it out!

---

Let me know if you want this tailored to a specific audience (teachers, students, judges) or need other edits!
