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
import re
import time

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
                    # Clean and preprocess the text
                    doc.page_content = self.preprocess_text(doc.page_content)
                
                all_docs.extend(docs)
                logger.info(f"Processed {len(docs)} pages from {pdf_file}")
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
        
        logger.info(f"Total documents loaded: {len(all_docs)}")
        return all_docs
    
    def preprocess_text(self, text):
        """Clean and preprocess text from PDF documents."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors
        text = text.replace('l', 'l').replace('|', 'I')
        
        # Remove page numbers and headers/footers (common in FDA documents)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove reference markers like [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # Fix hyphenated words that span lines
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Remove URLs and email addresses
        text = re.sub(r'https?://\S+|www\.\S+|\S+@\S+\.\S+', '', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        
        return text.strip()
    
    def split_documents(self, documents, chunk_size=500, chunk_overlap=150):
        """Split documents into chunks for better processing."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
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
        
        # Create retriever with MMR search for better diversity in results
        retriever = self.vector_store.as_retriever(
            search_type="mmr",  # Use Maximum Marginal Relevance for better diversity
            search_kwargs={"k": 10, "fetch_k": 50}  # Retrieve more documents for better coverage
        )
        
        # Create custom prompt template
        template = """
        You are ChatRIA, a helpful research assistant trained by Vivpro AI. You assist pharmaceutical companies in conducting research for various regulatory agencies.
        Your answer should be informed by the provided context from FDA documents.
        Write an accurate, detailed, and comprehensive response to the user's queries directed at regulatory documents.
        Your answer must be precise, of high quality, and written by an expert using an unbiased and scientific tone. 
        You must give the user an executive summary as a conclusion at the end.

        You should prepare a detailed answer using relevant section/subsection headers with proper formatting when needed. Do not mention any irrelevant information.

        IMPORTANT INSTRUCTIONS:
        1. NEVER say "Not specified in the provided context" or similar phrases. If information seems missing, try to infer from related details or state what IS known about the topic.
        2. If specific details aren't available, provide general information about the topic based on the context.
        3. DO NOT include any citations within the text of your answer.
        4. Thoroughly analyze all provided context before answering.
        5. When comparing medications or discussing multiple drugs, be sure to extract and synthesize information from all relevant documents.
        
        You MUST ADHERE to the following formatting instructions:
        - Use markdown to format paragraphs, lists, tables, and quotes whenever possible.
        - Use headings at levels 2 and 3 to separate sections of your response, like "## Header," but NEVER start an answer with a heading or title.
        - Use single new lines for lists and double new lines for paragraphs.
        - Include an executive summary at the end before the sources are added.

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
    
    def answer_question(self, question, enable_verification=True):
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
        
        # Verify and enhance the answer using a secondary search if enabled
        if enable_verification:
            print("\nVerifying and enhancing answer...")
            enhanced_answer = self.verify_and_enhance_answer(question, answer, source_docs)
            formatted_answer = enhanced_answer
        else:
            formatted_answer = answer
        
        return {
            "question": question,
            "answer": formatted_answer,
            "sources": sources
        }
    
    def verify_and_enhance_answer(self, question, initial_answer, source_docs):
        """Verify the answer and enhance it with additional information if needed."""
        # Check if the answer contains phrases indicating missing information
        missing_info_phrases = [
            "not specified in the provided context",
            "not mentioned in the context",
            "not provided in the context",
            "information is not available",
            "no information about",
            "cannot be determined from",
            "not found in the provided",
            "not clear from the context",
            "not stated in the",
            "no data on",
            "no details about"
        ]
        
        # Extract topics that might need additional searching
        potential_missing_topics = []
        
        # Check for missing information phrases
        has_missing_info = False
        for phrase in missing_info_phrases:
            if phrase.lower() in initial_answer.lower():
                has_missing_info = True
                # Extract the sentence containing the phrase
                sentences = re.split(r'(?<=[.!?])\s+', initial_answer)
                for sentence in sentences:
                    if phrase.lower() in sentence.lower():
                        # Extract potential topics from the sentence
                        # This is a simple approach - we're looking for capitalized words
                        # that might represent drug names or medical terms
                        potential_topics = re.findall(r'\b[A-Z][A-Za-z]+\b', sentence)
                        potential_missing_topics.extend(potential_topics)
        
        # Also check for drug names in the question that might not be in the answer
        drug_names_in_question = re.findall(r'\b[A-Z][A-Za-z]+\b', question)
        for drug_name in drug_names_in_question:
            if drug_name not in initial_answer and len(drug_name) > 3:  # Avoid short acronyms
                potential_missing_topics.append(drug_name)
        
        # If no missing info is detected, return the original answer
        if not has_missing_info and "not specified" not in initial_answer.lower() and not potential_missing_topics:
            print("✓ No missing information detected in the answer.")
            return initial_answer
        
        logger.info("Detected potentially missing information. Performing secondary search.")
        print("⚠ Detected potentially missing information in the answer.")
        
        if potential_missing_topics:
            print(f"Identified potential topics for further search: {', '.join(potential_missing_topics)}")
        
        # Create a more specific search query based on the question and missing topics
        search_query = question
        
        # Check if we have drug names in the potential topics
        drug_names = self.identify_drug_names(potential_missing_topics)
        
        if drug_names:
            # Prioritize drug names in the search query
            from collections import Counter
            search_query = f"{question} {' '.join(drug_names)}"
            print(f"Enhanced search query with drug names: '{search_query}'")
            
            # Perform targeted searches for each drug name
            all_additional_docs = []
            
            for drug_name in drug_names:
                print(f"Performing targeted search for {drug_name}...")
                # Create a drug-specific query
                drug_query = f"{drug_name} {question}"
                
                # Use a different retrieval strategy for the drug-specific search
                drug_retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5, "filter": {"source_file": f"{drug_name}.pdf"}}
                )
                
                try:
                    # Try to get documents specifically from this drug's PDF
                    drug_docs = drug_retriever.get_relevant_documents(drug_query)
                    if drug_docs:
                        print(f"Found {len(drug_docs)} documents specifically for {drug_name}")
                        all_additional_docs.extend(drug_docs)
                except Exception as e:
                    print(f"Error in targeted search for {drug_name}: {str(e)}")
            
            # If we found documents in the targeted search, use them
            if all_additional_docs:
                additional_docs = all_additional_docs
            else:
                # Fall back to the general search
                secondary_retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 15}
                )
                additional_docs = secondary_retriever.get_relevant_documents(search_query)
        else:
            # If no drug names, use the original approach with most common topics
            if potential_missing_topics:
                from collections import Counter
                topic_counts = Counter(potential_missing_topics)
                most_common_topics = [topic for topic, _ in topic_counts.most_common(3)]
                search_query = f"{question} {' '.join(most_common_topics)}"
                print(f"Enhanced search query: '{search_query}'")
            
            # Use a different retrieval strategy for the secondary search
            secondary_retriever = self.vector_store.as_retriever(
                search_type="similarity",  # Use similarity search instead of MMR for this search
                search_kwargs={"k": 15}  # Retrieve more documents
            )
            
            # Get additional documents
            additional_docs = secondary_retriever.get_relevant_documents(search_query)
        
        # Combine with original source docs, removing duplicates
        all_doc_contents = set(doc.page_content for doc in source_docs)
        unique_additional_docs = []
        
        for doc in additional_docs:
            if doc.page_content not in all_doc_contents:
                unique_additional_docs.append(doc)
                all_doc_contents.add(doc.page_content)
        
        # If we found new information, enhance the answer
        if unique_additional_docs:
            logger.info(f"Found {len(unique_additional_docs)} additional relevant documents.")
            print(f"✓ Found {len(unique_additional_docs)} additional relevant documents.")
            
            # Display the sources of additional documents
            print("Additional sources:")
            for doc in unique_additional_docs:
                source_file = doc.metadata.get("source_file", "Unknown")
                page = doc.metadata.get("page", "Unknown")
                print(f"- {source_file}, Page {page}")
            
            # Create a new prompt for enhancing the answer
            enhance_template = """
            You are ChatRIA, a helpful research assistant trained by Vivpro AI. You assist pharmaceutical companies in conducting research for various regulatory agencies.
            
            You have previously provided an answer to a question, but some information was missing or incomplete.
            
            Original Question: {question}
            
            Your Previous Answer: {initial_answer}
            
            Now you have been provided with additional context that might contain the missing information.
            
            Additional Context:
            {additional_context}
            
            Please enhance your previous answer by incorporating any relevant new information from the additional context.
            If the additional context doesn't provide the missing information, maintain your original answer.
            
            IMPORTANT INSTRUCTIONS:
            1. DO NOT say phrases like "Based on the additional context" or "With this new information".
            2. Simply integrate the new information seamlessly into a comprehensive answer.
            3. DO NOT include any citations within the text of your answer.
            4. Maintain the same formatting style as your original answer.
            5. If the additional context contradicts your original answer, prioritize the information from the additional context.
            
            Enhanced Answer:
            """
            
            enhance_prompt = PromptTemplate(
                template=enhance_template,
                input_variables=["question", "initial_answer", "additional_context"]
            )
            
            # Combine the additional documents into a single context
            additional_context = "\n\n".join([doc.page_content for doc in unique_additional_docs])
            
            # Use the same LLM to enhance the answer
            llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
            
            print("Enhancing answer with additional information...")
            
            # Get the enhanced answer
            enhance_result = llm.invoke(
                enhance_prompt.format(
                    question=question,
                    initial_answer=initial_answer,
                    additional_context=additional_context
                )
            )
            
            print("✓ Answer enhanced with additional information.")
            
            return enhance_result.content
        
        # If no new information was found, return the original answer
        print("No additional relevant information found. Returning original answer.")
        
        return initial_answer
    
    def identify_drug_names(self, potential_topics):
        """Identify drug names from potential topics by checking against PDF filenames."""
        pdf_files = self.get_pdf_files()
        pdf_basenames = [os.path.basename(pdf).split('.')[0] for pdf in pdf_files]
        
        # Create a list of drug names from the PDF filenames
        drug_names = []
        for topic in potential_topics:
            # Check if the topic matches or is contained in any PDF filename
            for pdf_name in pdf_basenames:
                if topic.upper() == pdf_name.upper() or topic.upper() in pdf_name.upper():
                    drug_names.append(topic)
                    break
        
        # If we didn't find any matches, check if any topic contains common drug name patterns
        if not drug_names:
            for topic in potential_topics:
                # Many drug names are in ALL CAPS
                if topic.isupper() and len(topic) > 3:
                    drug_names.append(topic)
        
        return list(set(drug_names))  # Remove duplicates

def main():
    """Main function to demonstrate the USFDA document processor."""
    processor = USFDADocumentProcessor()
    
    # Create or load vector store
    processor.create_vector_store()
    
    # Interactive Q&A loop
    print("\nUSFDA Document Question-Answering System")
    print("Type 'exit' to quit\n")
    
    # Example questions that search across multiple documents
    print("Example questions that search across multiple documents:")
    print("1. Compare the mechanism of action between OPDIVO and YERVOY.")
    print("2. What are the common adverse reactions shared by TECENTRIQ and PROLEUKIN?")
    print("3. How do the dosing recommendations differ between BRAFTOVI and MEKTOVI?")
    print("4. Compare the contraindications of TAFINLAR and COTELLIC.")
    print("5. What are the similarities and differences in patient monitoring requirements for immune checkpoint inhibitors?")
    print()
    
    # Add option to enable/disable verification
    enable_verification = True
    print("Answer verification is ENABLED. Type 'toggle verification' to turn it on/off.")
    print()
    
    while True:
        question = input("\nEnter your question: ")
        if question.lower() in ["exit", "quit", "q"]:
            break
        
        # Check if user wants to toggle verification
        if question.lower() == "toggle verification":
            enable_verification = not enable_verification
            status = "ENABLED" if enable_verification else "DISABLED"
            print(f"\nAnswer verification is now {status}.")
            continue
        
        # Process the question
        result = processor.answer_question(question, enable_verification)
        
        # Display answer
        print("\nAnswer:")
        print(result["answer"])
        
        # No longer displaying sources separately
        print("\n" + "-"*50)

if __name__ == "__main__":
    main() 