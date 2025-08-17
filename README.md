# Katy ISD Chatbot - High School Passion Project 🎓🤖

## Project Overview
This is a GenAI-powered chatbot that answers questions about the Katy ISD website using Retrieval-Augmented Generation (RAG). Built as a high school passion project to demonstrate modern AI capabilities.

## 🎯 **IMPORTANT: Use `app_mis1.py` for development**

**Primary File:** `app_mis1.py` - This is the preferred version with better UI/UX and styling.

## File Structure & Comparison

### Main Application Files

#### ✅ **`app_mis1.py`** - **USE THIS FILE**
**The preferred, improved version with:**
- Professional styled header with colored backgrounds and borders
- Complete sidebar with:
  - Logo image (`katyisd.jpg`)
  - About section explaining it's a high school project
  - Example questions for users
  - Clear chat button functionality
- Modern chat interface using `st.chat_input()` (more intuitive)
- Better file encoding (`utf-8-sig`) for handling BOM
- Cleaner, more streamlined code structure
- Chat history displayed in chronological order
- Enhanced error handling

#### ❌ **`app_mistral.py`** - **DON'T USE**
**Original version with basic features:**
- Simple centered title with minimal styling
- No sidebar - missing user guidance and features
- Form-based input with `st.form()` and submit button (less user-friendly)
- Basic file encoding (`utf-8`)
- Chat history displayed in reverse order (confusing)
- More verbose code structure
- Basic error handling

#### ❌ **`app.py`** - **DON'T USE**
**Incomplete version using Google Gemma model (broken)**

### Key Technical Differences

| Feature | app_mis1.py ✅ | app_mistral.py ❌ |
|---------|---------------|-------------------|
| **UI Style** | Professional with styled divs, colors, padding | Basic HTML styling |
| **Chat Interface** | `st.chat_input()` - modern | `st.form()` with submit button |
| **Sidebar** | Complete with logo, about, examples, clear button | None |
| **File Encoding** | `utf-8-sig` (handles BOM better) | `utf-8` (basic) |
| **Chat Order** | Chronological (intuitive) | Reversed (confusing) |
| **Code Quality** | Clean, streamlined, ~100 lines | Verbose, ~150 lines |
| **User Experience** | Professional, guided | Basic, minimal |

## Technical Stack
- **Model:** Mistral-7B-Instruct-v0.1 (Hugging Face)
- **Embeddings:** BAAI/bge-base-en-v1.5
- **Vector Store:** FAISS
- **Framework:** Streamlit
- **Document Processing:** LangChain

## Setup Instructions

1. **Install Dependencies:**
   ```bash
   pip install streamlit langchain-community langchain transformers torch faiss-cpu python-dotenv langchain-huggingface
   ```

2. **Environment Variables:**
   Create a `.env` file:
   ```
   HUGGINGFACE_TOKEN=your_hf_token_here
   ```

3. **Document Structure:**
   ```
   website_content/
   ├── documents/          # PDF files go here
   ├── *.txt              # Text files in root
   └── *.docx             # Word documents in root
   ```

4. **Assets:**
   - `katyisd.jpg` - Logo image for sidebar

5. **Run the Application:**
   ```bash
   streamlit run app_mis1.py
   ```

## Features
- 🎓 Katy ISD-specific knowledge base
- 🤖 AI-powered question answering with context
- 📚 Multi-format document support (PDF, TXT, DOCX)
- 💾 Persistent FAISS vector store with metadata caching
- 🎨 Professional UI with styled components
- 📋 Sidebar with user guidance and examples
- 💬 Modern chat interface with chronological history
- 🧠 Retrieval-Augmented Generation (RAG) for accurate responses
- ⚡ GPU acceleration when available
- 🔍 Source document tracking for transparency

## Development Notes
- Vector store automatically rebuilds if embedder model changes
- Uses proper Mistral prompt formatting with `[INST]` tags
- Implements document chunking (1000 chars, 150 overlap)
- Retrieves top 10 most relevant chunks per query
- Temperature set to 0.7 for balanced creativity/accuracy
- Max 512 new tokens per response

## File Usage Guidelines

**✅ DO:** Work exclusively with `app_mis1.py`
- Better user experience
- Professional appearance
- More maintainable code
- Enhanced functionality

**❌ DON'T:** Use `app_mistral.py` or `app.py`
- Outdated interfaces
- Missing features
- Less polished UX

## Future Improvements
- [ ] Add clickable citation links to source documents
- [ ] Implement conversation memory across sessions
- [ ] Add drag-and-drop file upload functionality
- [ ] Enhanced error handling with user-friendly messages
- [ ] Admin panel for document management
- [ ] Response quality metrics and feedback system

---
**Note:** This is a high school passion project demonstrating practical applications of modern AI/ML technologies including RAG, vector databases, and transformer models.