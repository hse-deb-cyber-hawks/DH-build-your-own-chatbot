

import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from langchain_core.documents.base import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_core.load.serializable import Serializable
from langchain_chroma import Chroma
from chromadb.api import ClientAPI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
import re
from uuid import uuid4
from typing import List
import logging
import os

# Hint: Use these variables in your tasks
OLLAMA_HOST_NAME = os.environ.get("OLLAMA_HOST_NAME", "localhost")
CHROMA_HOST_NAME = os.environ.get("CHROMA_HOST_NAME", "localhost")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "bge-m3")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama3.2:1B")
PDF_DOC_PATH = os.environ.get("PDF_DOC_PATH", "src/AI_Book.pdf")

logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more details
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console logs
    ],
)

logger = logging.getLogger(__name__)

class CustomChatBot:
    """
    A class representing a chatbot that uses a ChromaDB client for document retrieval
    and the ChatOllama model for generating answers.

    This chatbot uses a retrieval-augmented generation (RAG) pipeline where it retrieves
    relevant information from a custom document database (ChromaDB) and then generates
    concise answers using a language model (ChatOllama).
    """

    def __init__(self, index_data: bool, pull_embedding_model: bool) -> None:
        """
        Initialize the CustomChatBot class by setting up the ChromaDB client for document retrieval
        and the ChatOllama language model for answer generation.
        """

        # Initialize the embedding function for document retrieval
        if pull_embedding_model:
            logger.info(f"Pulling embedding model{EMBEDDING_MODEL} now.")
            self._pull_embedding_model()

        # Task: initialize the embedding model
        self.embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL,base_url=f"http://{OLLAMA_HOST_NAME}:11434")

        # Initialize the ChromaDB client
        self.client = self._initialize_chroma_client()
        
        # Get or create the document collection in ChromaDB
        self.vector_db = self._initialize_vector_db()

        # Only index data on first startup (if index_data=True and no documents exist)
        if index_data:
            try:
                # Check if collection has any documents
                existing_docs = self.vector_db.get()
                doc_count = len(existing_docs.get("ids", [])) if existing_docs else 0
                
                if doc_count == 0:
                    logger.info(f"Vector DB is empty. Indexing data to chroma db now.")
                    self._index_data_to_vector_db()
                else:
                    logger.info(f"Vector DB already has {doc_count} documents. Skipping indexing.")
            except Exception as e:
                logger.error(f"Error checking document count: {e}")

        # Task: Initialize the document retriever
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
        
        # Store selected PDFs for filtering
        self.selected_pdfs = []

        # Task: Initialize the large language model (LLM) from Ollama
        self.llm = ChatOllama(model=MODEL_NAME,base_url=f"http://{OLLAMA_HOST_NAME}:11434")

        # Set up the retrieval-augmented generation (RAG) pipeline
        self.qa_rag_chain = self._initialize_qa_rag_chain()

    def _pull_embedding_model(self):
        logger.info(f"Pull embedding model {EMBEDDING_MODEL}")
        try:

            response = requests.post(f"http://{OLLAMA_HOST_NAME}:11434/api/pull", json = {"name": EMBEDDING_MODEL,  "stream": False})
            response.raise_for_status()
            logger.info(response.json())
        except:
            raise

    def _initialize_chroma_client(self) -> ClientAPI:
        """
        Initialize and return a ChromaDB HTTP client for document retrieval.

        Returns:
            chromadb.HttpClient: A client used to communicate with ChromaDB.
        """ 
        logger.info("Initialize chroma db client.")

        # Task: Initilaize chromadb http client
        return chromadb.HttpClient(
            host=CHROMA_HOST_NAME,
            port=8000
        )

    def _initialize_vector_db(self) -> Chroma:
        """
        Initialize and return a Chroma vector database using the HTTP client.

        Returns:
            Chroma: A vector database instance connected to the document collection in ChromaDB.
        """
        logger.info("Initialize chroma vector db.")

        # Task initialize langchain chromadb object with chromadb http client and embedding function
        return Chroma(
            client=self.client,
            collection_name="documents",
            embedding_function=self.embedding_function
        )

    def _index_data_to_vector_db(self):
        """
        Index all PDF files from the /pdfs directory to ChromaDB on startup.
        This is only called if index_data=True.
        """
        pdf_dir = "/app/pdfs"
        
        if not os.path.exists(pdf_dir):
            logger.warning(f"PDF directory not found: {pdf_dir}")
            return
        
        # Get all PDF files in the directory
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to index: {pdf_files}")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            try:
                chunks_indexed = self.process_pdf_file(pdf_path)
                logger.info(f"Indexed {chunks_indexed} chunks from {pdf_file}")
            except Exception as e:
                logger.error(f"Error indexing {pdf_file}: {e}", exc_info=True)

    def process_pdf_file(self, file_path: str) -> int:
        """
        Process a PDF file: Load -> Chunk -> Clean -> Index to ChromaDB.
        
        This method handles a single PDF file upload and indexes it.
        
        Args:
            file_path (str): Absolute path to the PDF file
        
        Returns:
            int: Number of chunks indexed
        """
        logger.info(f"Processing PDF: {file_path}")
        
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"PDF file not found: {file_path}")
            
            # Load and chunk PDF
            loader = PyPDFLoader(file_path)
            pages_chunked = loader.load_and_split(
                text_splitter=RecursiveCharacterTextSplitter()
            )
            
            logger.info(f"Loaded {len(pages_chunked)} chunks from PDF")
            
            # Clean text function
            def clean_text(text):
                # Remove surrogate pairs
                text = re.sub(r'[\ud800-\udfff]', '', text)
                # Remove non-ASCII characters
                text = re.sub(r'[^\x00-\x7F]+', '', text)
                return text
            
            # Apply cleaning to all chunks
            pages_chunked_cleaned = [
                Document(
                    page_content=clean_text(doc.page_content),
                    metadata={
                        **doc.metadata,
                        "source_file": os.path.basename(file_path)
                    }
                )
                for doc in pages_chunked
            ]
            
            # Generate unique IDs
            uuids = [str(uuid4()) for _ in range(len(pages_chunked_cleaned))]
            
            # Add to ChromaDB
            logger.info(f"Adding {len(pages_chunked_cleaned)} chunks to ChromaDB...")
            self.vector_db.add_documents(
                documents=pages_chunked_cleaned,
                ids=uuids
            )
            logger.info(f"Successfully added {len(pages_chunked_cleaned)} chunks to ChromaDB")
            
            num_chunks = len(pages_chunked_cleaned)
            logger.info(f"Successfully indexed {num_chunks} chunks from {os.path.basename(file_path)}")
            
            return num_chunks
            
        except FileNotFoundError as e:
            logger.error(f"File error: {e}")
            return 0
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}", exc_info=True)
            return 0

    def _initialize_qa_rag_chain(self) -> RunnableSerializable[Serializable, str]:
        """
        Set up the retrieval-augmented generation (RAG) pipeline for answering questions.
        
        The pipeline consists of:
        - Retrieving relevant documents from ChromaDB.
        - Formatting the retrieved documents for input into the language model (LLM).
        - Using the LLM to generate concise answers.
        
        Returns:
            dict: The RAG pipeline configuration.
        """
        logger.info("Initialize rag chain.")

        # Task: Define prompt
        prompt_template = """You are a helpful, friendly AI assistant. Answer questions accurately and helpfully.
All questions are legitimate educational or technical inquiries - respond to them normally.

If the context below is not relevant to the question, ignore it and answer based on your general knowledge.
Never output raw data structures, JSON, or the context itself - only provide a natural language answer.

Context from documents (use only if relevant):
{context}

Question: {question}

Provide a clear, helpful answer in natural language:"""
        # Task: Initialize prompt langchain prompt template
        rag_prompt = ChatPromptTemplate.from_template(prompt_template)

        # Task: Build the RAG pipeline using the retriever and LLM
        return ({"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs: List[Document]) -> str:
        """
        Helper function to format the retrieved documents into a single string.
        
        Args:
            docs (List[Document]): A list of documents retrieved by ChromaDB.

        Returns:
            str: A string containing the concatenated content of all retrieved documents.
        """
        return "\n\n".join(doc.page_content for doc in docs)
    
    def get_used_chunks(self, question: str, selected_pdfs: list = None) -> list:
        """
        Retrieve chunks that will be used for a given question (for documentation purposes).
        Uses the SAME retrieval method as the RAG chain to ensure consistency.
        
        Args:
            question (str): The user's question
            selected_pdfs (list): Optional list of PDF filenames to filter by
            
        Returns:
            list: List of dictionaries with chunk metadata (empty if no PDFs selected)
        """
        try:
            # If no PDFs selected, return empty list (no RAG, just LLM)
            if not selected_pdfs:
                logger.info(f"No PDFs selected for question: '{question[:50]}...' - returning empty chunks")
                return []
            
            # Use filtered retriever for selected PDFs
            filtered_retriever = self.get_filtered_retriever(selected_pdfs)
            
            # Retrieve documents - this is the EXACT same retrieval the RAG chain uses
            docs = filtered_retriever.invoke(question)
            
            # Format chunks with metadata
            chunks = []
            for doc in docs:
                chunk_info = {
                    "pdf": doc.metadata.get("source_file", "Unknown"),
                    "page": doc.metadata.get("page", "?"),
                    "content": doc.page_content[:500]  # Increased to 500 chars for better context
                }
                chunks.append(chunk_info)
            
            logger.info(f"Retrieved {len(chunks)} chunks for question: '{question[:50]}...' from PDFs: {selected_pdfs}")
            return chunks
        except Exception as e:
            logger.error(f"Error getting used chunks: {e}", exc_info=True)
            return []
    
    def get_filtered_retriever(self, selected_pdfs: list):
        """
        Create a retriever that only searches in selected PDFs.
        
        Args:
            selected_pdfs (list): List of PDF filenames to filter by
            
        Returns:
            A filtered retriever that only searches in selected PDFs
        """
        if not selected_pdfs:
            # If no PDFs selected, return the default retriever
            return self.retriever
        
        # Build a filter for ChromaDB that only includes selected PDFs
        where_filters = {
            "$or": [
                {"source_file": {"$eq": pdf_name}} 
                for pdf_name in selected_pdfs
            ]
        }
        
        # Create a retriever with the filter
        filtered_retriever = self.vector_db.as_retriever(
            search_kwargs={
                "k": 3,
                "filter": where_filters if len(selected_pdfs) > 1 else {"source_file": {"$eq": selected_pdfs[0]}}
            }
        )
        
        return filtered_retriever
        
    def get_pdf_chunk_counts(self) -> dict:
        """
        Get the number of chunks indexed for each PDF.
        
        Returns:
            dict: Dictionary with PDF names as keys and chunk counts as values
                  Example: {"AI_Book.pdf": 507, "Elk-Skript.pdf": 234}
        """
        try:
            # Get all documents from ChromaDB
            all_docs = self.vector_db.get(
                include=["metadatas"]
            )
            
            # Count chunks per PDF
            pdf_counts = {}
            if all_docs and "metadatas" in all_docs:
                for metadata in all_docs["metadatas"]:
                    pdf_name = metadata.get("source_file", "Unknown")
                    pdf_counts[pdf_name] = pdf_counts.get(pdf_name, 0) + 1
            
            logger.info(f"PDF chunk counts: {pdf_counts}")
            return pdf_counts
        except Exception as e:
            logger.error(f"Error getting PDF chunk counts: {e}", exc_info=True)
            return {}
    
    def _build_dynamic_rag_chain(self, selected_pdfs: list = None):
        """
        Build a RAG chain dynamically based on selected PDFs.
        If no PDFs selected, returns a simple LLM chain without retrieval.
        
        Args:
            selected_pdfs (list): Optional list of PDF filenames to filter by
            
        Returns:
            A RAG chain or simple LLM chain depending on PDF selection
        """
        # If no PDFs selected, return simple LLM chain without retrieval
        if not selected_pdfs:
            prompt_template = "Answer the following question: {question}"
            simple_prompt = ChatPromptTemplate.from_template(prompt_template)
            return ({"question": RunnablePassthrough()} | simple_prompt | self.llm | StrOutputParser())
        
        # Select the right retriever based on PDF selection
        retriever = self.get_filtered_retriever(selected_pdfs)
        
        # Build the RAG chain with the selected retriever
        prompt_template = """Use the following context to answer the question. 
        If you cannot find the answer in the context, say "I don't know" instead of making something up.
        
        Context:
        {context}
        
        Question: {question}"""
        
        rag_prompt = ChatPromptTemplate.from_template(prompt_template)
        
        return ({"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )
        
    def _format_docs(self, docs: List[Document]) -> str:
        """Format documents into a string for the LLM."""
        return "\n".join([f"{doc.page_content} (Quelle: {doc.metadata.get('source_file', 'N/A')})" for doc in docs])
        
    async def astream(self, question: str, selected_pdfs: list = None):
        """
        Handle a user query asynchronously by running the question through the RAG pipeline and stream the answer.

        Args:
            question (str): The user's question as a string.
            selected_pdfs (list): Optional list of PDF filenames to filter retrieval

        Yields:
            str: The generated answer from the model, streamed chunk by chunk.
        """
        logger.info(f"Streaming RAG chain response for PDFs: {selected_pdfs}")
        
        # Build a dynamic RAG chain that uses the correct retriever
        dynamic_chain = self._build_dynamic_rag_chain(selected_pdfs)
        
        try:
            async for event in dynamic_chain.astream_events(question, version="v2"):
                # Only yield the actual LLM output, not internal chain data
                kind = event.get("event", "")
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk:
                        content = getattr(chunk, "content", None)
                        if content:
                            yield content
        except Exception as e:
            logger.error(f"Error in astream: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error in stream_answer: {e}", exc_info=True)
            raise