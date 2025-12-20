import streamlit as st
import asyncio
import logging
from langchain.schema import ChatMessage
from src.chatbot import CustomChatBot
import os
import json
from datetime import datetime
from pathlib import Path

INDEX_DATA = os.environ.get("INDEX_DATA", "0")
PULL_EMBEDDING_MODEL = os.environ.get("PULL_EMBEDDING_MODEL", "0")

# Configure logger
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more details
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console logs
    ],
)

logger = logging.getLogger(__name__)

# Create exports folder if it doesn't exist
EXPORT_FOLDER = Path(os.environ.get("EXPORT_PATH", "/app/exports"))
EXPORT_FOLDER.mkdir(exist_ok=True, parents=True)

# Create PDFs folder if it doesn't exist
PDF_FOLDER = Path("/app/pdfs")
PDF_FOLDER.mkdir(exist_ok=True, parents=True)

def upload_and_index_pdf(uploaded_file):
    """Upload a PDF and index it to ChromaDB."""
    if uploaded_file is None:
        return
    
    # Validate PDF format
    if uploaded_file.type != "application/pdf":
        st.error("❌ Bitte lade nur PDF-Dateien hoch!")
        return
    
    # Clean filename
    safe_filename = "".join(c if c.isalnum() or c in " .-_" else "" for c in uploaded_file.name)
    file_path = PDF_FOLDER / safe_filename
    
    with st.spinner(f"📥 Lade PDF hoch und indexiere '{safe_filename}'..."):
        try:
            # Save PDF file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            logger.info(f"PDF saved to {file_path}")
            
            # Process and index PDF
            num_chunks = st.session_state["bot"].process_pdf_file(str(file_path))
            
            st.success(f"✅ PDF erfolgreich importiert!\n\n**Datei:** {safe_filename}\n**Chunks:** {num_chunks}")
            logger.info(f"PDF indexed with {num_chunks} chunks")
            
        except Exception as e:
            st.error(f"❌ Fehler beim Hochladen: {e}")
            logger.error(f"PDF upload error: {e}")

def delete_pdf_from_chromadb(pdf_name: str):
    """Delete all chunks of a PDF from ChromaDB."""
    try:
        # Get all documents from ChromaDB and filter by source_file metadata
        # Use ChromaDB's delete function with a filter
        st.session_state["bot"].vector_db.delete(
            where={"source_file": {"$eq": pdf_name}}
        )
        logger.info(f"Deleted all chunks for PDF: {pdf_name}")
        return True
    except Exception as e:
        logger.error(f"Error deleting PDF from ChromaDB: {e}")
        return False

def export_chat_history():
    """Export current chat history with smart naming based on topic."""
    if not st.session_state.messages:
        st.warning("No messages to export!")
        return
    
    with st.spinner("📊 Analyzing chat topic..."):
        # Create export data (ohne summary speichern)
        export_data = {
            "export_date": datetime.now().isoformat(),
            "message_count": len(st.session_state.messages),
            "selected_pdfs": st.session_state.get("selected_pdfs", []),
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "used_chunks": msg.metadata.get("used_chunks", []) if hasattr(msg, "metadata") else []
                }
                for msg in st.session_state.messages
            ]
        }
        
        # Generate topic only (not stored, just for filename)
        chat_content = "\n".join([f"{msg.role.upper()}: {msg.content}" for msg in st.session_state.messages])
        
        topic_prompt = f"""Extract the main topic from this chat in exactly 1-3 words. 
Reply with ONLY the topic words, nothing else. No explanation, no sentences.

Examples of correct answers:
- 3D Drucker
- Machine Learning
- Generative AI
- Python Programming

Chat:
{chat_content[:1000]}

Topic (1-3 words only):"""
        
        try:
            logger.info("Analyzing chat topic for filename...")
            topic_response = st.session_state["bot"].llm.invoke(topic_prompt).content
            # Clean response - remove quotes, colons, etc.
            topic_cleaned = topic_response.strip().strip('"\'').strip()
            # Extract first 1-3 words only
            words = topic_cleaned.split()[:3]
            topic = " ".join(words) if words else "Chat"
            logger.info(f"Extracted topic: {topic}")
        except Exception as e:
            logger.error(f"Error analyzing topic: {e}")
            topic = "Chat"
        
        # Generate filename with topic and formatted date/time
        now = datetime.now()
        date_str = now.strftime("%d.%m.%Y")
        time_str = now.strftime("%H%M%S")
        
        # Clean topic for filename (remove special characters)
        safe_topic = "".join(c if c.isalnum() or c in " -_" else "" for c in topic).strip()
        if not safe_topic:
            safe_topic = "Chat"
        
        filename = f"{safe_topic}-{date_str}-{time_str}.json"
        filepath = EXPORT_FOLDER / filename
        
        # Save to file
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Chat exported successfully to {filepath}")
            st.success(f"✅ Chat exported successfully!\n\n**Datei:** {filename}")
        except Exception as e:
            logger.error(f"Error exporting chat: {e}")
            st.error(f"❌ Error exporting chat: {e}")

def import_and_summarize_chat(uploaded_file):
    """Import a chat from JSON file and create a summary."""
    try:
        # Read the JSON file
        import_data = json.load(uploaded_file)
        logger.info(f"Imported chat with {import_data.get('message_count', 0)} messages")
        
        # Get messages from imported data
        imported_messages = import_data.get("messages", [])
        imported_pdfs = import_data.get("selected_pdfs", [])
        
        if not imported_messages:
            st.warning("No messages found in the imported file!")
            return
        
        # Restore selected PDFs
        st.session_state["selected_pdfs"] = imported_pdfs
        
        # Convert to ChatMessage objects and load into session
        st.session_state.messages = [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in imported_messages
        ]
        
        # Create summary from the imported chat
        chat_content = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in imported_messages])
        
        # Extract main topics from the conversation
        summary_prompt = f"""Analysiere diesen Chat-Verlauf und erstelle eine strukturierte Zusammenfassung:

{chat_content}

Bitte antworte in folgendem Format:
**Thema:** [Kurze Beschreibung worum es ging]
**Wichtigste Erkenntnisse:**
- [Punkt 1]
- [Punkt 2]
- [Punkt 3]
- [Punkt 4]
- [Punkt 5]"""
        
        logger.info("Generating summary with LLM...")
        
        # Generate summary using the LLM
        summary = ""
        with st.spinner("📊 Generating summary from imported chat..."):
            try:
                summary = st.session_state["bot"].llm.invoke(summary_prompt).content
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                summary = f"Error generating summary: {str(e)}"
        
        # Add summary as assistant message to the chat with better formatting
        export_date = import_data.get('export_date', 'unknown')
        summary_message = f"""**📋 Zusammenfassung des importierten Chats:**

Zusammengefasst haben wir im vorherigen Chat über verschiedene Themen geredet. Die wichtigsten Erkenntnisse waren:

{summary}"""
        st.session_state.messages.append(ChatMessage(role="assistant", content=summary_message))
        
        st.success(f"✅ Chat imported successfully! Loaded {len(imported_messages)} messages.")
        st.rerun()
        
    except json.JSONDecodeError:
        st.error("❌ Invalid JSON file!")
        logger.error("Invalid JSON file uploaded")
    except Exception as e:
        st.error(f"❌ Error importing chat: {e}")
        logger.error(f"Error importing chat: {e}")

def get_exported_chats():
    """Get list of exported chat files."""
    try:
        files = sorted(EXPORT_FOLDER.glob("*.json"), reverse=True)
        return [(f.name, f) for f in files]
    except Exception as e:
        logger.error(f"Error listing exports: {e}")
        return []

# Initialize chatbot instance (avoid reloading)
@st.cache_resource
def get_chatbot():
    """Initialize and cache the chatbot instance to avoid reloading."""
    return CustomChatBot(index_data=bool(int(INDEX_DATA)), pull_embedding_model=bool(int(PULL_EMBEDDING_MODEL)))

if "bot" not in st.session_state:
    st.session_state["bot"] = get_chatbot()

# Streamlit UI setup
st.set_page_config(page_title="ChatDoc", page_icon="📄")
st.header("Chat with your Document")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

# Initialize selected PDFs (keine standardmäßig aktiv - User muss bewusst auswählen)
if "selected_pdfs" not in st.session_state:
    st.session_state["selected_pdfs"] = []

# Sidebar buttons
if st.sidebar.button("Clear message history", key="clear_btn"):
    st.session_state["messages"].clear()
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]
    st.rerun()

st.sidebar.markdown("---")

# Export Chat Section
if st.sidebar.button("📥 Export Chat", key="export_btn"):
    export_chat_history()

# Import Chat Section
if st.sidebar.button("📤 Import & Summarize", key="import_browse_btn"):
    st.session_state.show_imports = True

if st.session_state.get("show_imports", False):
    st.sidebar.subheader("Available Exports")
    
    exported_chats = get_exported_chats()
    
    if not exported_chats:
        st.sidebar.warning("No exported chats found!")
        if st.sidebar.button("Cancel", key="cancel_import"):
            st.session_state.show_imports = False
    else:
        chat_options = [name for name, _ in exported_chats]
        selected_chat = st.sidebar.selectbox("Select a chat to import:", chat_options, key="chat_select")
        
        # Find the selected file path
        selected_path = None
        for name, path in exported_chats:
            if name == selected_chat:
                selected_path = path
                break
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("✅ Load", key="confirm_import"):
                if selected_path:
                    with open(selected_path, "r", encoding="utf-8") as f:
                        import_and_summarize_chat(f)
                    st.session_state.show_imports = False
        
        with col2:
            if st.button("❌ Cancel", key="cancel_import2"):
                st.session_state.show_imports = False

# PDF Management Section
st.sidebar.markdown("---")
st.sidebar.subheader("📄 PDF Management")

uploaded_pdf = st.sidebar.file_uploader(
    "PDF hochladen",
    type=["pdf"],
    label_visibility="collapsed"
)

if uploaded_pdf:
    if st.sidebar.button("➕ Indexieren", key="upload_pdf_btn"):
        upload_and_index_pdf(uploaded_pdf)

# Show indexed PDFs with checkboxes for selection
if PDF_FOLDER.exists():
    pdfs = sorted([f.name for f in PDF_FOLDER.glob("*.pdf")])
    if pdfs:
        st.sidebar.markdown("**📋 PDFs verwenden:**")
        
        # Get chunk counts
        pdf_chunk_counts = st.session_state["bot"].get_pdf_chunk_counts()
        
        for pdf_name in pdfs:
            col1, col2 = st.sidebar.columns([3, 1])
            
            with col1:
                # Show PDF name with chunk count
                chunk_count = pdf_chunk_counts.get(pdf_name, 0)
                display_text = f"{pdf_name} ({chunk_count} chunks)"
                
                is_selected = st.checkbox(
                    display_text,
                    value=(pdf_name in st.session_state.get("selected_pdfs", [])),
                    key=f"pdf_select_{pdf_name}"
                )
                
                # Update session state and trigger rerun if changed
                old_selection = pdf_name in st.session_state["selected_pdfs"]
                if is_selected and not old_selection:
                    st.session_state["selected_pdfs"].append(pdf_name)
                    st.rerun()
                elif not is_selected and old_selection:
                    st.session_state["selected_pdfs"].remove(pdf_name)
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"delete_{pdf_name}", help="PDF und Chunks löschen"):
                    with st.spinner(f"Lösche {pdf_name}..."):
                        # Delete from ChromaDB
                        delete_pdf_from_chromadb(pdf_name)
                        
                        # Delete file
                        try:
                            (PDF_FOLDER / pdf_name).unlink()
                            logger.info(f"Deleted PDF file: {pdf_name}")
                            
                            # Remove from selected if it was selected
                            if pdf_name in st.session_state["selected_pdfs"]:
                                st.session_state["selected_pdfs"].remove(pdf_name)
                            
                            st.success(f"✅ {pdf_name} gelöscht!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Fehler beim Löschen: {e}")
                            logger.error(f"Error deleting PDF file: {e}")

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

# Handle user input
if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.session_state.messages.append(ChatMessage(role="user", content=user_query))
    logger.info(f"Write user message in session state {user_query}")
    st.chat_message("user").write(user_query)

    async def handle_user_query(user_query):
        container = st.empty()
        answer = ""
        try:
            # Pass selected PDFs to retriever filter (empty list if none selected = no RAG)
            selected_pdfs = st.session_state.get("selected_pdfs", [])
            
            # Get chunks used for this question (empty if no PDFs selected)
            used_chunks = st.session_state["bot"].get_used_chunks(user_query, selected_pdfs=selected_pdfs)
            
            async for chunk in st.session_state["bot"].astream(user_query, selected_pdfs=selected_pdfs):
                if chunk:
                    answer+=chunk
                    container.markdown(answer)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            container.error("An error occurred while processing your request.")

        # Store assistant response in session state
        if answer:
            logger.info(f"Write assistant message in session state {user_query}")
            # Create ChatMessage with metadata (used_chunks stored but not displayed)
            msg = ChatMessage(role="assistant", content=answer)
            msg.metadata = {"used_chunks": used_chunks}
            st.session_state.messages.append(msg)

    with st.chat_message("assistant"):
        with st.spinner("Searching for information in your documents and generation response..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(handle_user_query(user_query))