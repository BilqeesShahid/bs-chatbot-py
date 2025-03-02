import streamlit as st  # Importing the Streamlit library for building the web app
from langchain_groq import ChatGroq  # Importing ChatGroq for AI chat capabilities
from langchain_core.output_parsers import StrOutputParser  # Importing output parser for processing AI responses
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)  # Importing prompt templates for structuring chat messages

# API key for accessing the Groq service
groq_api_key = "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"

# Display the title and subtitle of the app
st.title("👩‍💻 BS Chatbot")  # Title with an emoji
st.subheader("Hi! I'm  BS Chatbot. How can I assist you today? 💬")  # Subtitle with an emoji

# Sidebar configuration for user settings
# Sidebar configuration for user settings
with st.sidebar:
    st.header("🔧 Chatbot Configuration")  # Header for the sidebar
    selected_model = st.selectbox("Model Used for my chatbot", ["deepseek-r1-distill-llama-70b"])
    st.caption("deepseek-r1-distill-llama-70b:deepseek Release-1,70 billion-parameter distilled Llama(Large Language Model Meta AI)-based model, optimized for efficiency and high performance.")
    st.markdown("### ChatBot Capabilities")  # Section header for capabilities
    capabilities = [
        "💬 General Conversation",
        "🐍 Python Expert",
        "🐞 Debugging Assistant",
        "📝 Code Documentation",
        "💡 Solution Design",
        "🌐 Information Retrieval"
    ]  # List of capabilities
    st.multiselect("Select Capabilities", capabilities, default=capabilities)  # Multiselect for capabilities
    with st.expander("### Quick Tips"):  # Expandable section for tips
        st.markdown("""
        - **Tip 1**: Use the chatbot for general conversation to explore its versatility.
        - **Tip 2**: Leverage the Python expertise for coding help and debugging.
        - **Tip 3**: Utilize information retrieval for quick access to data and facts.
        """)  # Tips for using the chatbot

# Initialize the chat engine with the selected model
llm_engine = ChatGroq(api_key=groq_api_key, model=selected_model, temperature=0.3)

# Function to build the system prompt based on selected capabilities
def build_system_prompt(selected_capabilities):
    capabilities_text = ", ".join(selected_capabilities)  # Join capabilities into a string
    return f"You are a versatile AI chatbot with the following capabilities: {capabilities_text}. Engage in general conversation, provide coding assistance, and offer information on a wide range of topics. Always respond in English."

# Create the system prompt using the selected capabilities
system_prompt = SystemMessagePromptTemplate.from_template(
    build_system_prompt(capabilities)
)

# Initialize session state for message log
if "message_log" not in st.session_state:
    st.session_state.message_log = []  # Initialize message log if not present

# Container for displaying chat messages
chat_container = st.container()

# Display chat messages from the message log
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):  # Display each message with its role
            if "<think>" in message["content"] and "</think>" in message["content"]:
                start_idx = message["content"].find("<think>") + len("<think>")
                end_idx = message["content"].find("</think>")
                think_content = message["content"][start_idx:end_idx].strip()  # Extract "thinking" content
                actual_response = message["content"][end_idx + len("</think>"):].strip()  # Extract actual response
                with st.expander("🤔 AI Thought Process"):  # Expandable section for thought process
                    st.markdown(think_content)
                st.markdown(actual_response)  # Display the actual response
            else:
                st.markdown(message["content"])  # Display message content

# Input field for user queries
user_query = st.chat_input("Type your question or topic here...")

# Function to generate AI response
def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()  # Create processing pipeline
    return processing_pipeline.invoke({})  # Invoke the pipeline to get response

# Function to build the prompt chain for the chat
def build_prompt_chain():
    prompt_sequence = [system_prompt]  # Start with the system prompt
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))  # Add user message
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))  # Add AI message
    return ChatPromptTemplate.from_messages(prompt_sequence)  # Return the complete prompt chain

# Process user query if present
if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})  # Add user query to log
    with st.spinner("🧠 Processing..."):  # Show spinner while processing
        prompt_chain = build_prompt_chain()  # Build the prompt chain
        ai_response = generate_ai_response(prompt_chain)  # Generate AI response
    st.session_state.message_log.append({"role": "ai", "content": ai_response})  # Add AI response to log
    st.rerun()  # Rerun the app to update the display