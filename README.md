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
- Implements advanced text preprocessing to improve PDF text extraction
- Uses Maximum Marginal Relevance (MMR) for better diversity in search results
- Optimized chunking strategy for better context preservation
- Includes an answer verification system that automatically detects and fills in missing information
- Performs targeted drug-specific searches when drug names are identified in missing information

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
- Advanced options for tuning retrieval parameters
- A "Force Reprocess" button to regenerate the vector store
- Answer verification toggle to enable/disable automatic enhancement of answers

## Example Questions

### Single Document Questions
- What are the indications for OPDIVO?
- What are the common side effects of YERVOY?
- What is the recommended dosage for TECENTRIQ?
- What contraindications exist for BRAFTOVI?
- How should TAFINLAR be administered?

### Cross-Document Questions
- Compare the mechanism of action between OPDIVO and YERVOY.
- What are the common adverse reactions shared by TECENTRIQ and PROLEUKIN?
- How do the dosing recommendations differ between BRAFTOVI and MEKTOVI?
- Compare the contraindications of TAFINLAR and COTELLIC.
- What are the similarities and differences in patient monitoring requirements for immune checkpoint inhibitors?

## Notes

- The first run will take some time as it processes all PDFs and creates the vector store.
- Subsequent runs will be faster as they load the existing vector store.
- To force reprocessing of all PDFs, use the "Force Reprocess All Documents" button in the Streamlit interface, or modify the code to set `force_reload=True` when calling `create_vector_store()`.
- The system uses `allow_dangerous_deserialization=True` when loading the vector store. This is safe as long as you're loading vector stores that you created yourself and trust the source of the data.
- The text preprocessing step improves the quality of text extraction from PDFs by fixing common OCR errors and formatting issues.
- The MMR search algorithm helps ensure diversity in the retrieved documents, which is especially important for cross-document questions.
- The answer verification system automatically detects phrases like "not specified in the provided context" and performs a secondary search to find the missing information, enhancing the answer with any new relevant details found.
- When drug names are identified in missing information, the system performs targeted searches directly in the corresponding drug's PDF document to ensure relevant information is found.