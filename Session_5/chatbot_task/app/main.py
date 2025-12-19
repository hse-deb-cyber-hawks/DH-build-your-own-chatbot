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
    """Export current chat history to a JSON file."""
    if not st.session_state.messages:
        st.warning("No messages to export!")
        return
    
    # Create export data
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
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_export_{timestamp}.json"
    filepath = EXPORT_FOLDER / filename
    
    # Save to file
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Chat exported successfully to {filepath}")
        st.success(f"✅ Chat exported successfully!\nLocation: {filepath}")
    except Exception as e:
        logger.error(f"Error exporting chat: {e}")
        st.error(f"❌ Error exporting chat: {e}")

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