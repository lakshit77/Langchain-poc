import os
import glob
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class USFDADocumentProcessor:
    def __init__(self, pdf_folder="USFDA", vector_store_path="usfda_vectorstore"):
        """
        Initialize the USFDA document processor.
        
        Args:
            pdf_folder: Path to the folder containing PDF files
            vector_store_path: Path to save/load the vector store
        """
        self.pdf_folder = pdf_folder
        self.vector_store_path = vector_store_path
        self.vector_store = None
        
    def get_pdf_files(self) -> List[str]:
        """Get all PDF files from the specified folder."""
        pdf_pattern = os.path.join(self.pdf_folder, "*.pdf")
        pdf_files = glob.glob(pdf_pattern)
        logger.info(f"Found {len(pdf_files)} PDF files in {self.pdf_folder}")
        return pdf_files
    
    def load_and_process_pdfs(self) -> List:
        """Load and process all PDF files into documents."""
        pdf_files = self.get_pdf_files()
        all_docs = []
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing {pdf_file}...")
                loader = PyPDFLoader(pdf_file)
                docs = loader.load()
                
                # Add metadata about source file
                for doc in docs:
                    doc.metadata["source_file"] = os.path.basename(pdf_file)
                
                all_docs.extend(docs)
                logger.info(f"Processed {len(docs)} pages from {pdf_file}")
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
        
        logger.info(f"Total documents loaded: {len(all_docs)}")
        return all_docs
    
    def split_documents(self, documents, chunk_size=1000, chunk_overlap=100):
        """Split documents into chunks for better processing."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        splits = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(splits)} chunks")
        return splits
    
    def create_vector_store(self, force_reload=False):
        """Create or load a vector store from the documents."""
        # Check if vector store already exists
        if os.path.exists(self.vector_store_path) and not force_reload:
            logger.info(f"Loading existing vector store from {self.vector_store_path}")
            self.vector_store = FAISS.load_local(
                self.vector_store_path, 
                OpenAIEmbeddings(),
                allow_dangerous_deserialization=True
            )
            return self.vector_store
        
        # Load and process documents
        documents = self.load_and_process_pdfs()
        splits = self.split_documents(documents)
        
        # Create vector store
        logger.info("Creating new vector store...")
        self.vector_store = FAISS.from_documents(splits, OpenAIEmbeddings())
        
        # Save vector store
        logger.info(f"Saving vector store to {self.vector_store_path}")
        self.vector_store.save_local(self.vector_store_path)
        
        return self.vector_store
    
    def create_qa_chain(self):
        """Create a question-answering chain using the vector store."""
        if not self.vector_store:
            self.create_vector_store()
        
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Create custom prompt template
        template = """
        You are ChatRIA, a helpful research assistant trained by Vivpro AI. You assist pharmaceutical companies in conducting research for various regulatory agencies.
        Your answer should be informed by the provided context from FDA documents.
        Write an accurate, detailed, and comprehensive response to the user's queries directed at regulatory documents.
        Your answer must be precise, of high quality, and written by an expert using an unbiased and scientific tone. 
        You must give the user an executive summary as a conclusion at the end.

        You should prepare a detailed answer using relevant section/subsection headers with proper formatting when needed. Do not mention any irrelevant information.

        You MUST cite the source documents that answer the query. Do not mention any irrelevant results.
        You MUST ADHERE to the following instructions for citing sources:

        - To cite a source document, include the document name and page number in brackets at the end of the corresponding sentence, for example "Opdivo is indicated for treatment of melanoma[OPDIVO.pdf, p.5]."
        - NO SPACE between the last word and the citation, and ALWAYS use brackets. Only use this format to cite sources.
        - DO NOT use reference IDs like [Reference ID: 5368236]. Always use the PDF filename and page number.

        You MUST ADHERE to the following formatting instructions:
        - Use markdown to format paragraphs, lists, tables, and quotes whenever possible.
        - Use headings at levels 2 and 3 to separate sections of your response, like "## Header," but NEVER start an answer with a heading or title.
        - Use single new lines for lists and double new lines for paragraphs.

        Context:
        {context}

        Question: {question}

        Answer:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain
    
    def answer_question(self, question):
        """Answer a question using the QA chain."""
        qa_chain = self.create_qa_chain()
        result = qa_chain({"query": question})
        
        answer = result["result"]
        source_docs = result["source_documents"]
        
        # Extract source information
        sources = []
        for doc in source_docs:
            source_file = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            if source_file and {"file": source_file, "page": page} not in sources:
                sources.append({"file": source_file, "page": page})
        
        # Format the answer to include source information at the end
        formatted_answer = answer
        
        # Add source information at the end
        if sources:
            formatted_answer += "\n\n## Sources\n"
            for source in sources:
                formatted_answer += f"- {source['file']}, Page {source['page']}\n"
        
        return {
            "question": question,
            "answer": formatted_answer,
            "sources": sources
        }

def main():
    """Main function to demonstrate the USFDA document processor."""
    processor = USFDADocumentProcessor()
    
    # Create or load vector store
    processor.create_vector_store()
    
    # Interactive Q&A loop
    print("\nUSFDA Document Question-Answering System")
    print("Type 'exit' to quit\n")
    
    while True:
        question = input("\nEnter your question: ")
        if question.lower() in ["exit", "quit", "q"]:
            break
        
        result = processor.answer_question(question)
        
        print("\nAnswer:")
        print(result["answer"])
        
        print("\nSources:")
        for source in result["sources"]:
            print(f"- {source['file']}, Page {source['page']}")
        
        print("\n" + "-"*50)

if __name__ == "__main__":
    main() 