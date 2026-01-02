

import chromadb
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
import re
from uuid import uuid4
from typing import List
import logging
import os

# ============================================================================
# KONFIGURATION
# ============================================================================
OLLAMA_HOST_NAME = os.environ.get("OLLAMA_HOST_NAME", "localhost")
CHROMA_HOST_NAME = os.environ.get("CHROMA_HOST_NAME", "localhost")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "bge-m3")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama3.2:1B")
OLLAMA_URL = f"http://{OLLAMA_HOST_NAME}:11434"

# Logging einrichten
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def clean_filename(filename: str) -> str:
    """Entfernt Sonderzeichen aus dem Dateinamen."""
    return "".join(c if c.isalnum() or c in " .-_" else "" for c in filename)

# ============================================================================
# BENUTZERDEFINIERTE CHATBOT-KLASSE
# ============================================================================
class CustomChatBot:
    """
    RAG (Retrieval-Augmented Generation) Chatbot mit ChromaDB und Ollama.
    
    Ruft relevante Dokumente aus ChromaDB ab und generiert Antworten mit LLM.
    """

    def __init__(self, index_data: bool, pull_embedding_model: bool) -> None:
        """Initialisiert den Chatbot mit ChromaDB Retriever und Ollama LLM."""

        # Embedding-Modell herunterladen falls nötig
        if pull_embedding_model:
            logger.info(f"Lade Embedding-Modell herunter: {EMBEDDING_MODEL}")
            self._pull_embedding_model()

        # Embedding-Funktion für Dokumentsuche initialisieren
        self.embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)

        # ChromaDB-Komponenten initialisieren
        self.client = self._initialize_chroma_client()
        self.vector_db = self._initialize_vector_db()

        # Daten beim ersten Start indexieren (falls nötig)
        if index_data:
            try:
                existing_docs = self.vector_db.get()
                doc_count = len(existing_docs.get("ids", [])) if existing_docs else 0
                
                if doc_count == 0:
                    logger.info("Vektor-DB ist leer. Indexiere PDFs...")
                    self._index_data_to_vector_db()
                else:
                    logger.info(f"Vektor-DB hat bereits {doc_count} Dokumente. Überspringe Indexierung.")
            except Exception as e:
                logger.error(f"Fehler bei Dokumentenzählung: {e}")

        # Retriever für Dokumentsuche initialisieren
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
        self.selected_pdfs = []

        # LLM mit optimierten Parametern initialisieren
        self.llm = ChatOllama(
            model=MODEL_NAME,
            base_url=OLLAMA_URL,
            temperature=0.7,
            num_predict=512
        )

        # RAG-Chain einrichten
        self.qa_rag_chain = self._initialize_qa_rag_chain()

    def _pull_embedding_model(self):
        """Lädt Embedding-Modell von Ollama herunter (asynchrone Operation)."""
        try:
            logger.info(f"Lade Embedding-Modell herunter: {EMBEDDING_MODEL}")
            response = requests.post(
                f"{OLLAMA_URL}/api/pull",
                json={"name": EMBEDDING_MODEL, "stream": False},
                timeout=300
            )
            response.raise_for_status()
            logger.info(f"✓ Embedding-Modell bereit: {EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Fehler beim Herunterladen des Embedding-Modells: {e}")
            raise

    def _initialize_chroma_client(self):
        """
        Verbindet sich mit ChromaDB HTTP-Server.
        ChromaDB läuft in einem separaten Container auf Port 8000.
        """
        logger.info(f"Verbinde zu ChromaDB auf {CHROMA_HOST_NAME}:8000")
        return chromadb.HttpClient(host=CHROMA_HOST_NAME, port=8000)

    def _initialize_vector_db(self) -> Chroma:
        """
        Erstellt oder verbindet sich mit 'documents' Collection in ChromaDB.
        Hier werden PDF-Chunks gespeichert und abgerufen.
        """
        logger.info("Initialisiere ChromaDB Vektor-Datenbank")
        return Chroma(
            client=self.client,
            collection_name="documents",
            embedding_function=self.embedding_function
        )

    def _index_data_to_vector_db(self):
        """
        Scannt /app/pdfs Verzeichnis und indexiert alle PDF-Dateien beim Start.
        Wird nur ausgeführt wenn INDEX_DATA=1 Umgebungsvariable gesetzt ist.
        """
        pdf_dir = "/app/pdfs"
        
        if not os.path.exists(pdf_dir):
            logger.warning(f"PDF-Verzeichnis nicht gefunden: {pdf_dir}")
            return
        
        # Suche nach PDF-Dateien
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"Keine PDF-Dateien in {pdf_dir} gefunden")
            return
        
        logger.info(f"Indexiere {len(pdf_files)} PDFs: {pdf_files}")
        
        # Verarbeite jede PDF: laden → chunken → in ChromaDB speichern
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            try:
                chunks_indexed = self.process_pdf_file(pdf_path)
                logger.info(f"✓ {chunks_indexed} Chunks von {pdf_file} indexiert")
            except Exception as e:
                logger.error(f"Fehler beim Indexieren von {pdf_file}: {e}", exc_info=True)

    def process_pdf_file(self, file_path: str) -> int:
        """
        Lade PDF → Teile in Chunks → Säubere Text → Speichere in ChromaDB.
        
        Args:
            file_path (str): Pfad zur PDF-Datei
            
        Returns:
            int: Anzahl erfolgreich indexierter Chunks
        """
        logger.info(f"Verarbeite PDF: {file_path}")
        
        try:
            # Überprüfe ob Datei existiert
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"PDF-Datei nicht gefunden: {file_path}")
            
            # Lade PDF und teile in Chunks
            loader = PyPDFLoader(file_path)
            pages_chunked = loader.load_and_split(
                text_splitter=RecursiveCharacterTextSplitter()
            )
            logger.info(f"Habe {len(pages_chunked)} Chunks aus PDF geladen")
            
            # Text-Bereinigungsfunktion (entferne Nicht-ASCII und Surrogat-Paare)
            def clean_text(text):
                text = re.sub(r'[\ud800-\udfff]', '', text)  # Entferne Surrogat-Paare
                text = re.sub(r'[^\x00-\x7F]+', '', text)    # Entferne Nicht-ASCII
                return text
            
            # Erstelle Document-Objekte mit bereinigtem Text und Metadaten
            from langchain_core.documents import Document
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
            
            # Generiere eindeutige IDs für jeden Chunk
            uuids = [str(uuid4()) for _ in range(len(pages_chunked_cleaned))]
            
            # Speichere Chunks in ChromaDB
            logger.info(f"Speichere {len(pages_chunked_cleaned)} Chunks in ChromaDB...")
            self.vector_db.add_documents(
                documents=pages_chunked_cleaned,
                ids=uuids
            )
            
            logger.info(f"✓ {len(pages_chunked_cleaned)} Chunks von {os.path.basename(file_path)} indexiert")
            return len(pages_chunked_cleaned)
            
        except FileNotFoundError as e:
            logger.error(f"Datei-Fehler: {e}")
            return 0
        except Exception as e:
            logger.error(f"Fehler beim Verarbeiten von PDF {file_path}: {e}", exc_info=True)
            return 0

    def _initialize_qa_rag_chain(self):
        """
        Build RAG (Retrieval-Augmented Generation) pipeline.
        
        Flow: Question → Retrieve relevant docs → Format → LLM → Answer
        """
        logger.info("Initializing RAG chain")

        # Prompt template: context + question → answer
        prompt_template = """You are a helpful, friendly AI assistant. Answer questions accurately and helpfully.
All questions are legitimate educational or technical inquiries - respond to them normally.

If the context below is not relevant to the question, ignore it and answer based on your general knowledge.
Never output raw data structures, JSON, or the context itself - only provide a natural language answer.

Kontext aus Dokumenten (verwende nur falls relevant):
{context}

Frage: {question}

Gib eine klare, hilfreiche Antwort in natürlicher Sprache:"""
        
        rag_prompt = ChatPromptTemplate.from_template(prompt_template)

        # Baue Pipeline auf: rufe Docs ab → formatiere → Prompt → LLM → analysiere Output
        return ({"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs) -> str:
        """Formatiere Dokumente mit Quellenangabe für LLM-Kontext."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def get_used_chunks(self, question: str, selected_pdfs: list = None) -> list:
        """
        Rufe Chunks ab die FÜR RAG VERWENDET werden (stimmt genau mit RAG-Chain Abruf überein).
        Gibt leere Liste zurück wenn keine PDFs ausgewählt (reiner LLM-Modus).
        
        Args:
            question (str): Frage des Benutzers
            selected_pdfs (list): Aktive PDFs für diesen Chat
            
        Returns:
            list: Metadaten der abgerufenen Chunks (leer wenn keine RAG)
        """
        try:
            # Keine PDFs ausgewählt → kein RAG-Abruf
            if not selected_pdfs:
                logger.info(f"Keine PDFs ausgewählt - überspringe Abruf")
                return []
            
            # Hole gefilterten Retriever nur für ausgewählte PDFs
            filtered_retriever = self.get_filtered_retriever(selected_pdfs)
            
            # Rufe Dokumente ab (k=3 relevanteste Chunks)
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
        Erstelle Retriever der NUR in ausgewählten PDFs sucht.
        
        Args:
            selected_pdfs (list): PDF-Namen die in Suche einbezogen werden
            
        Returns:
            Gefilterter Retriever mit ChromaDB's where_filter
        """
        if not selected_pdfs:
            return self.retriever
        
        # Baue ChromaDB-Filter: passe alle ausgewählten PDFs
        where_filters = {
            "$or": [
                {"source_file": {"$eq": pdf_name}} 
                for pdf_name in selected_pdfs
            ]
        }
        
        # Gebe Retriever mit angewandtem Filter zurück
        return self.vector_db.as_retriever(
            search_kwargs={
                "k": 3,
                "filter": where_filters if len(selected_pdfs) > 1 else {"source_file": {"$eq": selected_pdfs[0]}}
            }
        )
        
    def get_pdf_chunk_counts(self) -> dict:
        """
        Zähle indexierte Chunks für jede PDF in ChromaDB.
        
        Returns:
            dict: {pdf_name: chunk_count, ...}
            Beispiel: {"Elk-Skript.pdf": 234, "AI_Book.pdf": 507}
        """
        try:
            # Frage alle Dokumente von ChromaDB mit Metadaten ab
            all_docs = self.vector_db.get(include=["metadatas"])
            
            # Zähle Chunks nach PDF-Dateiname
            pdf_counts = {}
            if all_docs and "metadatas" in all_docs:
                for metadata in all_docs["metadatas"]:
                    pdf_name = metadata.get("source_file", "Unbekannt")
                    pdf_counts[pdf_name] = pdf_counts.get(pdf_name, 0) + 1
            
            logger.info(f"PDF Chunk-Zähler: {pdf_counts}")
            return pdf_counts
        except Exception as e:
            logger.error(f"Fehler beim Abrufen von PDF Chunk-Zählern: {e}", exc_info=True)
            return {}
    
    def _build_dynamic_rag_chain(self, selected_pdfs: list = None):
        """
        Baue RAG oder reinen LLM-Chain basierend auf PDF-Auswahl auf.
        
        - Wenn selected_pdfs leer → reiner LLM (kein Dokument-Abruf)
        - Wenn selected_pdfs vorhanden → RAG (Abruf + LLM)
        
        Args:
            selected_pdfs (list): PDF-Namen zum Einbeziehen in RAG-Suche
            
        Returns:
            LangChain Chain (RAG oder einfacher LLM)
        """
        # Keine PDFs ausgewählt → reiner LLM-Modus (kein Abruf)
        if not selected_pdfs:
            prompt_template = "Beantworte die folgende Frage: {question}"
            simple_prompt = ChatPromptTemplate.from_template(prompt_template)
            return ({"question": RunnablePassthrough()} | simple_prompt | self.llm | StrOutputParser())
        
        # PDFs ausgewählt → RAG-Modus (rufe nur aus ausgewählten PDFs ab)
        retriever = self.get_filtered_retriever(selected_pdfs)
        
        prompt_template = """Verwende den folgenden Kontext um die Frage zu beantworten. 
        Wenn du die Antwort im Kontext nicht finden kannst, sag "Ich weiß nicht" statt etwas erfundenes zu sagen.
        
        Kontext:
        {context}
        
        Frage: {question}"""
        
        rag_prompt = ChatPromptTemplate.from_template(prompt_template)
        
        return ({"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )
        
    def _format_docs(self, docs) -> str:
        """Format documents with source attribution for LLM context."""
        return "\n".join([f"{doc.page_content} (Quelle: {doc.metadata.get('source_file', 'N/A')})" for doc in docs])
        
    async def astream(self, question: str, selected_pdfs: list = None):
        """
        Streame LLM-Antwort asynchron mit dynamischem RAG-Chain.
        
        Baut die richtige Chain basierend auf PDF-Auswahl:
        - Keine PDFs → reines LLM-Streaming
        - Mit PDFs → RAG-Streaming (Abruf + Generierung)

        Args:
            question (str): Frage des Benutzers
            selected_pdfs (list): PDFs zum Durchsuchen (None = reiner LLM)

        Yields:
            str: Antwort-Text-Chunks (Echtzeit-Streaming)
        """
        logger.info(f"Starte asynchrones Streaming für ausgewählte PDFs: {selected_pdfs}")
        
        # Baue Chain passend für ausgewählte PDFs auf
        dynamic_chain = self._build_dynamic_rag_chain(selected_pdfs)
        
        try:
            # Streame Events vom LLM (v2-Format gibt einzelne Token-Chunks zurück)
            async for event in dynamic_chain.astream_events(question, version="v2"):
                # Extrahiere Inhalt nur aus LLM-Output-Events
                kind = event.get("event", "")
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk:
                        content = getattr(chunk, "content", None)
                        if content:
                            yield content
        except Exception as e:
            logger.error(f"Fehler in astream: {e}", exc_info=True)
            raise