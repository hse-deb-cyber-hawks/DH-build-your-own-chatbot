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
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content
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
        
        if not imported_messages:
            st.warning("No messages found in the imported file!")
            return
        
        # Convert to ChatMessage objects and load into session
        st.session_state.messages = [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in imported_messages
        ]
        
        # Create summary from the imported chat
        chat_content = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in imported_messages])
        
        summary_prompt = f"""Erstelle eine prägnante Zusammenfassung (max. 5 Punkte) der wichtigsten Erkenntnisse 
aus diesem bisherigen Chat-Verlauf. Fasse zusammen, worum es in der Unterhaltung ging:

{chat_content}

Zusammenfassung:"""
        
        logger.info("Generating summary with LLM...")
        
        # Generate summary using the LLM
        summary = ""
        with st.spinner("📊 Generating summary from imported chat..."):
            try:
                summary = st.session_state["bot"].llm.invoke(summary_prompt).content
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                summary = f"Error generating summary: {str(e)}"
        
        # Add summary as assistant message to the chat
        summary_message = f"""**📋 Zusammenfassung des importierten Chats:**\n\n{summary}"""
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
if "bot" not in st.session_state:
    st.session_state["bot"] = CustomChatBot(index_data=bool(int(INDEX_DATA)), pull_embedding_model=bool(int(PULL_EMBEDDING_MODEL)))

# Streamlit UI setup
st.set_page_config(page_title="ChatDoc", page_icon="📄")
st.header("Chat with your Document")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

# Sidebar buttons
if st.sidebar.button("Clear message history", key="clear_btn"):
    st.session_state["messages"].clear()
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]
    st.rerun()

if st.sidebar.button("📥 Export Chat", key="export_btn"):
    export_chat_history()

# Import Chat Section
st.sidebar.markdown("---")

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
            async for chunk in st.session_state["bot"].astream(user_query):
                if chunk:
                    answer+=chunk
                    container.markdown(answer)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            container.error("An error occurred while processing your request.")

        # Store assistant response in session state
        if answer:
            logger.info(f"Write assistant message in session state {user_query}")
            st.session_state.messages.append(ChatMessage(role="assistant", content=answer))

    with st.chat_message("assistant"):
        with st.spinner("Searching for information in your documents and generation response..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(handle_user_query(user_query))