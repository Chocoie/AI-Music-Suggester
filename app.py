import streamlit as st
from main import MusicAssistant

st.set_page_config(page_title="Music Assistant", page_icon="🎵", layout="centered")

# initialize assistant and state
if "assistant" not in st.session_state:
    st.session_state.assistant = MusicAssistant()
if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = st.session_state.assistant.initial_state.copy()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🎵 Music Assistant")
status = "running" if not st.session_state.conversation_state.get("complete") else "complete"
st.caption(f"mode: music-only • status: {status}")

# Display chat history
for msg in st.session_state.chat_history:
    role = msg.get("role", "").lower()
    content = msg.get("content", "")
    if role == "human":
        st.chat_message("user").write(content)
    elif role == "ai":
        st.chat_message("assistant").write(content)

# Input box
disabled = st.session_state.conversation_state.get("complete", False)
prompt = st.chat_input("Ask for music suggestions… (q to quit)", disabled=disabled)

if prompt is not None:
    # Add human message to chat history
    st.session_state.chat_history.append({"role": "human", "content": prompt})
    st.chat_message("user").write(prompt)

    if prompt.strip().lower() in ("q", "quit"):
        st.session_state.conversation_state["complete"] = True
        st.session_state.chat_history.append({"role": "ai", "content": "Goodbye! Thanks for using the Music Assistant."})
        st.chat_message("assistant").write("Goodbye! Thanks for using the Music Assistant.")
        st.rerun()

    with st.spinner("Thinking…"):
        try:
            # Set the streamlitPrompt in the state so userIn can use it
            current_state = st.session_state.conversation_state.copy()
            current_state["streamlitPrompt"] = prompt
            
            # Now invoke the workflow - userIn will use the streamlitPrompt instead of input()
            result_state = st.session_state.assistant.app.invoke(current_state)
            
            # Update the conversation state
            st.session_state.conversation_state = result_state
            
            # Extract the AI response from the messages
            messages = result_state.get("messages", [])
            ai_response = "I'm ready to help you find music!"
            
            # Find the last AI message
            for msg in reversed(messages):
                if hasattr(msg, 'type') and msg.type == "ai":
                    content = msg.content
                    if isinstance(content, str):
                        ai_response = content
                    elif isinstance(content, list) and content:
                        if isinstance(content[0], dict):
                            ai_response = content[0].get('text', str(content[0]))
                        else:
                            ai_response = str(content[0])
                    elif isinstance(content, dict):
                        ai_response = content.get('text', str(content))
                    else:
                        ai_response = str(content)
                    break
            
            # Add AI response to chat history
            st.session_state.chat_history.append({"role": "ai", "content": ai_response})
            st.chat_message("assistant").write(ai_response)
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            st.session_state.chat_history.append({"role": "ai", "content": error_msg})
            st.chat_message("assistant").write(error_msg)

    st.rerun()