# USFDA Document Question-Answering System

This system processes PDF files from the USFDA folder, converts them to a vector store, and creates a retrieval-based QA system using LangChain. The system uses a ChatRIA-style research assistant format to provide detailed, well-cited responses to pharmaceutical regulatory questions.

## Features

- Automatically processes all PDF files in the USFDA folder
- Splits documents into manageable chunks for better retrieval
- Creates a FAISS vector store for efficient similarity search
- Provides a simple interactive interface for asking questions
- Returns answers with source information (file name and page number)
- Uses a professional research assistant format with proper citations
- Includes section headers and markdown formatting for better readability
- Provides executive summaries for comprehensive answers

## Requirements

The following packages are required:
```
langchain
python-dotenv
langchain-openai
langchain-community
langchain-core
pypdf
faiss-cpu
streamlit
```

## Setup

1. Make sure you have all the required packages installed:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. Place your FDA PDF documents in the `USFDA` folder.

## Usage

### Command Line Interface

Run the command line version:
```
python usfda_qa.py
```

The script will:
1. Process all PDF files in the USFDA folder
2. Create a vector store (or load an existing one)
3. Start an interactive Q&A session in the terminal

Type 'exit', 'quit', or 'q' to exit the interactive session.

### Streamlit Web Interface

For a more user-friendly experience, run the Streamlit version:
```
streamlit run usfda_qa_streamlit.py
```

This will:
1. Start a local web server
2. Open a browser window with the web interface
3. Allow you to interact with the system through a modern UI

The Streamlit interface provides:
- A list of available documents
- Example questions you can click on
- Progress indicators during processing
- Better formatting of answers and sources

## Example Questions

- What are the indications for OPDIVO?
- What are the common side effects of YERVOY?
- What is the recommended dosage for TECENTRIQ?
- What contraindications exist for BRAFTOVI?
- How should TAFINLAR be administered?

## Notes

- The first run will take some time as it processes all PDFs and creates the vector store.
- Subsequent runs will be faster as they load the existing vector store.
- To force reprocessing of all PDFs, use the "Force reload all documents" checkbox in the Streamlit interface, or modify the code to set `force_reload=True` when calling `create_vector_store()`.
- The system uses `allow_dangerous_deserialization=True` when loading the vector store. This is safe as long as you're loading vector stores that you created yourself and trust the source of the data.