import streamlit as st
import asyncio
import logging
from langchain.schema import ChatMessage
from src.chatbot import CustomChatBot
import os
import json
from datetime import datetime
from pathlib import Path

# ============================================================================
# STREAMLIT SEITEN-KONFIGURATION (muss erste Anweisung sein)
# ============================================================================
st.set_page_config(page_title="ChatDoc", page_icon="📄")

# ============================================================================
# KONFIGURATION & SETUP
# ============================================================================
INDEX_DATA = os.environ.get("INDEX_DATA", "0")
PULL_EMBEDDING_MODEL = os.environ.get("PULL_EMBEDDING_MODEL", "0")
EXPORT_FOLDER = Path(os.environ.get("EXPORT_PATH", "/app/exports"))
PDF_FOLDER = Path("/app/pdfs")

# Logging einrichten
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Erforderliche Ordner erstellen
EXPORT_FOLDER.mkdir(exist_ok=True, parents=True)
PDF_FOLDER.mkdir(exist_ok=True, parents=True)

# ============================================================================
# HILFSFUNKTIONEN
# ============================================================================
def clean_filename(filename: str) -> str:
    """Entfernt Sonderzeichen aus dem Dateinamen."""
    return "".join(c if c.isalnum() or c in " .-_" else "" for c in filename)

# ============================================================================
# PDF-VERWALTUNGSFUNKTIONEN
# ============================================================================
def upload_and_index_pdf(uploaded_file):
    """Lädt eine PDF hoch und indexiert sie in ChromaDB."""
    if uploaded_file is None or uploaded_file.type != "application/pdf":
        st.error("❌ Bitte lade nur PDF-Dateien hoch!")
        return
    
    safe_filename = clean_filename(uploaded_file.name)
    file_path = PDF_FOLDER / safe_filename
    
    with st.spinner(f"📥 Lade PDF hoch und indexiere '{safe_filename}'..."):
        try:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            logger.info(f"PDF gespeichert unter {file_path}")
            
            num_chunks = st.session_state["bot"].process_pdf_file(str(file_path))
            st.success(f"✅ PDF erfolgreich importiert!\n\n**Datei:** {safe_filename}\n**Chunks:** {num_chunks}")
            logger.info(f"PDF mit {num_chunks} Chunks indexiert")
            
        except Exception as e:
            st.error(f"❌ Fehler beim Hochladen: {e}")
            logger.error(f"PDF Hochlad-Fehler: {e}")

def delete_pdf_from_chromadb(pdf_name: str) -> bool:
    """Löscht alle Chunks einer PDF aus ChromaDB."""
    try:
        st.session_state["bot"].vector_db.delete(where={"source_file": {"$eq": pdf_name}})
        logger.info(f"Alle Chunks gelöscht für PDF: {pdf_name}")
        return True
    except Exception as e:
        logger.error(f"Fehler beim Löschen der PDF aus ChromaDB: {e}")
        return False

def import_and_summarize_chat(uploaded_file):
    """Importiert einen Chat aus JSON und generiert eine Zusammenfassung."""
    try:
        import_data = json.load(uploaded_file)
        imported_messages = import_data.get("messages", [])
        
        if not imported_messages:
            st.warning("Keine Nachrichten in der importierten Datei gefunden!")
            return
        
        # Stelle PDFs und Nachrichten wieder her
        st.session_state["selected_pdfs"] = import_data.get("selected_pdfs", [])
        st.session_state.messages = [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in imported_messages
        ]
        
        # Generiere Zusammenfassung
        chat_content = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in imported_messages])
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
        
        logger.info("Generiere Zusammenfassung mit LLM...")
        with st.spinner("📊 Generiere Zusammenfassung aus importiertem Chat..."):
            try:
                summary = st.session_state["bot"].llm.invoke(summary_prompt).content
            except Exception as e:
                logger.error(f"Fehler beim Generieren der Zusammenfassung: {e}")
                summary = f"Fehler: {str(e)}"
        
        summary_message = f"""**📋 Zusammenfassung des importierten Chats:**

Zusammengefasst haben wir im vorherigen Chat über verschiedene Themen geredet. Die wichtigsten Erkenntnisse waren:

{summary}"""
        
        st.session_state.messages.append(ChatMessage(role="assistant", content=summary_message))
        st.success(f"✅ Chat erfolgreich importiert! {len(imported_messages)} Nachrichten geladen.")
        st.rerun()
        
    except json.JSONDecodeError:
        st.error("❌ Ungültige JSON-Datei!")
        logger.error("Ungültige JSON-Datei hochgeladen")
    except Exception as e:
        st.error(f"❌ Fehler beim Importieren des Chats: {e}")
        logger.error(f"Fehler beim Importieren des Chats: {e}")

def get_exported_chats():
    """Ruft die Liste der exportierten Chat-Dateien ab."""
    try:
        files = sorted(EXPORT_FOLDER.glob("*.json"), reverse=True)
        return [(f.name, f) for f in files]
    except Exception as e:
        logger.error(f"Fehler beim Auflisten der Exporte: {e}")
        return []

# ============================================================================
# CHATBOT-INITIALISIERUNG
# ============================================================================
@st.cache_resource
def get_chatbot():
    """Initialisiert und speichert die Chatbot-Instanz um erneutes Laden zu vermeiden."""
    return CustomChatBot(index_data=bool(int(INDEX_DATA)), pull_embedding_model=bool(int(PULL_EMBEDDING_MODEL)))

if "bot" not in st.session_state:
    st.session_state["bot"] = get_chatbot()

# ============================================================================
# UI LAYOUT & SESSION-STATUS
# ============================================================================
st.header("Chat mit deinem Dokument")

# Initialisiere Session-Status Variablen
if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="Wie kann ich dir helfen?")]

# Initialisiere ausgewählte PDFs (Benutzer muss sie explizit auswählen)
if "selected_pdfs" not in st.session_state:
    st.session_state["selected_pdfs"] = []

# Initialisiere Processing-Flag (verhindert gleichzeitige Anfragen)
if "processing" not in st.session_state:
    st.session_state["processing"] = False

# Verfolge hochgeladene PDFs um die gleiche Datei nicht erneut zu verarbeiten
if "uploaded_pdf_names" not in st.session_state:
    st.session_state["uploaded_pdf_names"] = set()

# ============================================================================
# SEITENLEISTE - STEUERELEMENTE & VERWALTUNG
# ============================================================================
if st.sidebar.button("Nachrichtenverlauf löschen", key="clear_btn"):
    st.session_state["messages"].clear()
    st.session_state["messages"] = [ChatMessage(role="assistant", content="Wie kann ich dir helfen?")]
    st.session_state.processing = False
    st.rerun()

st.sidebar.markdown("---")

# Exportiere Chat Sektion
if st.sidebar.button("📥 Chat Exportieren", key="export_btn"):
    st.session_state.show_export = not st.session_state.get("show_export", False)

# Importiere Chat Sektion
if st.sidebar.button("📤 Importieren & Zusammenfassen", key="import_browse_btn"):
    st.session_state.show_imports = True

if st.session_state.get("show_imports", False):
    st.sidebar.subheader("Verfügbare Exporte")
    
    exported_chats = get_exported_chats()
    
    if not exported_chats:
        st.sidebar.warning("Keine exportierten Chats gefunden!")
        if st.sidebar.button("Abbrechen", key="cancel_import"):
            st.session_state.show_imports = False
    else:
        chat_options = [name for name, _ in exported_chats]
        selected_chat = st.sidebar.selectbox("Wähle einen Chat zum Importieren:", chat_options, key="chat_select")
        
        # Finde den ausgewählten Dateipfad
        selected_path = None
        for name, path in exported_chats:
            if name == selected_chat:
                selected_path = path
                break
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("✅ Laden", key="confirm_import"):
                if selected_path:
                    with open(selected_path, "r", encoding="utf-8") as f:
                        import_and_summarize_chat(f)
                    st.session_state.show_imports = False
        
        with col2:
            if st.button("❌ Abbrechen", key="cancel_import2"):
                st.session_state.show_imports = False

# PDF-Verwaltungs-Sektion
st.sidebar.markdown("---")
st.sidebar.subheader("📄 PDF-Verwaltung")

uploaded_pdf = st.sidebar.file_uploader(
    "PDF hochladen",
    type=["pdf"],
    label_visibility="collapsed"
)

# Automatisch PDF hochladen und indexieren wenn ausgewählt (verhindert doppelte Verarbeitung)
if uploaded_pdf:
    safe_filename = clean_filename(uploaded_pdf.name)
    
    if safe_filename not in st.session_state["uploaded_pdf_names"]:
        upload_and_index_pdf(uploaded_pdf)
        st.session_state["uploaded_pdf_names"].add(safe_filename)

# Zeige verfügbare PDFs mit Auswahlkästchen und Löschen-Buttons
if PDF_FOLDER.exists():
    pdfs = sorted([f.name for f in PDF_FOLDER.glob("*.pdf")])
    if pdfs:
        st.sidebar.markdown("**📋 PDFs verwenden:**")
        
        # Hole Chunk-Zähler für die Anzeige
        pdf_chunk_counts = st.session_state["bot"].get_pdf_chunk_counts()
        
        for pdf_name in pdfs:
            col1, col2 = st.sidebar.columns([3, 1])
            
            with col1:
                # Zeige PDF mit Chunk-Zähler und Auswahlkästchen
                chunk_count = pdf_chunk_counts.get(pdf_name, 0)
                display_text = f"{pdf_name} ({chunk_count} Chunks)"
                
                is_selected = st.checkbox(
                    display_text,
                    value=(pdf_name in st.session_state.get("selected_pdfs", [])),
                    key=f"pdf_select_{pdf_name}"
                )
                
                # Aktualisiere Session-Status und starte erneut wenn geändert
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
                        # Lösche aus ChromaDB
                        delete_pdf_from_chromadb(pdf_name)
                        
                        # Lösche Datei von Festplatte
                        try:
                            (PDF_FOLDER / pdf_name).unlink()
                            logger.info(f"PDF-Datei gelöscht: {pdf_name}")
                            
                            # Entferne auch aus ausgewählten PDFs wenn ausgewählt war
                            if pdf_name in st.session_state["selected_pdfs"]:
                                st.session_state["selected_pdfs"].remove(pdf_name)
                            
                            st.success(f"✅ {pdf_name} gelöscht!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Fehler beim Löschen: {e}")
                            logger.error(f"Fehler beim Löschen der PDF-Datei: {e}")

# ============================================================================
# CHAT-SCHNITTSTELLE - NACHRICHTEN & EINGABE
# ============================================================================
# Zeige alle Nachrichten aus dem Gesprächsverlauf
for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

# Benutzereingabe (immer aktiv aber verarbeitet nur wenn nicht beschäftigt)
user_query = st.chat_input(placeholder="Frag mich etwas!")

if user_query:
    # Verhindere neue Anfragen während Bot die vorherige bearbeitet
    if st.session_state.processing:
        st.warning("⏳ Bitte warten bis die vorherige Antwort fertig ist...")
    else:
        # Setze Processing-Flag um neue Anfragen während Response zu verhindern
        st.session_state.processing = True
        
        st.session_state.messages.append(ChatMessage(role="user", content=user_query))
        logger.info(f"Schreibe Benutzernachricht in Session-Status {user_query}")
        st.chat_message("user").write(user_query)

    async def handle_user_query(user_query):
        container = st.empty()
        answer = ""
        try:
            # Übergebe ausgewählte PDFs an Retriever-Filter (leere Liste wenn keine = kein RAG)
            selected_pdfs = st.session_state.get("selected_pdfs", [])
            
            # Hole Chunks die für diese Frage verwendet werden (leer wenn keine PDFs ausgewählt)
            used_chunks = st.session_state["bot"].get_used_chunks(user_query, selected_pdfs=selected_pdfs)
            
            # Streame Response vom LLM
            async for chunk in st.session_state["bot"].astream(user_query, selected_pdfs=selected_pdfs):
                if chunk:
                    answer+=chunk
                    container.markdown(answer)
        except Exception as e:
            logger.error(f"Fehler bei Anfrageverarbeitung: {e}")
            container.error("Ein Fehler ist bei der Verarbeitung deiner Anfrage aufgetreten.")

        # Speichere Response mit Metadaten (verwendete Chunks zur Referenz)
        if answer:
            logger.info(f"Schreibe Assistentennachricht in Session-Status {user_query}")
            msg = ChatMessage(role="assistant", content=answer)
            msg.metadata = {"used_chunks": used_chunks}
            st.session_state.messages.append(msg)

    # Zeige Assistenten-Response mit Spinner während Verarbeitung
    with st.chat_message("assistant"):
        with st.spinner("Durchsuche Dokumente und generiere Antwort..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(handle_user_query(user_query))
    
    # Reaktiviere Eingabe nach Response-Abschluss
    st.session_state.processing = False
    st.rerun()

# ============================================================================
# EXPORT-CHAT-DIALOG
# ============================================================================
if st.session_state.get("show_export", False):
    st.markdown("---")
    st.subheader("💾 Chat exportieren")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        export_name = st.text_input(
            "Dateiname:",
            placeholder="z.B. Mein Chat",
            help="Der Name ohne .json - wird automatisch hinzugefügt",
            key="export_name_input"
        )
    
    with col2:
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            if st.button("💾 Export", use_container_width=True):
                if export_name.strip():
                    with st.spinner("💾 Exportiere Chat..."):
                        # Erstelle Export-Daten
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
                        
                        # Bereinige Dateinamen
                        safe_name = clean_filename(export_name).strip()
                        if not safe_name:
                            safe_name = "Chat"
                        
                        filename = f"{safe_name}.json"
                        filepath = EXPORT_FOLDER / filename
                        
                        # Speichere in Datei
                        try:
                            with open(filepath, "w", encoding="utf-8") as f:
                                json.dump(export_data, f, indent=2, ensure_ascii=False)
                            logger.info(f"Chat erfolgreich exportiert zu {filepath}")
                            st.success(f"✅ Chat exportiert!\n\n**Datei:** {filename}")
                            st.session_state.show_export = False
                            st.rerun()
                        except Exception as e:
                            logger.error(f"Fehler beim Exportieren des Chats: {e}")
                            st.error(f"❌ Fehler beim Exportieren des Chats: {e}")
                else:
                    st.warning("⚠️ Bitte einen Namen eingeben!")
        
        with export_col2:
            if st.button("❌ Abbrechen", use_container_width=True):
                st.session_state.show_export = False
                st.rerun()