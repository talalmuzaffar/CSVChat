import textwrap
from datetime import datetime
import tabulate
import pandas as pd
import streamlit as st
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.chat_models import ChatOpenAI

# Page config
st.set_page_config(page_title="CSV Data Chat with AI", layout="wide")

# Initialize session states
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = None
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Please upload a CSV file to begin."}
    ]
if "memory" not in st.session_state:
    st.session_state.memory = []
if "agent" not in st.session_state:
    st.session_state.agent = None

def clear_api_key():
    st.session_state.openai_api_key = None

# Function to format response text
def format_text(text, width=70):
    if isinstance(text, pd.DataFrame):
        return f"```\n{text.to_string()}\n```"
    elif "| " in str(text):  # Detect table-like content
        return f"```\n{text}\n```"
    else:
        wrapped_lines = textwrap.fill(str(text), width=width)
        return wrapped_lines

# Main UI
if not st.session_state.openai_api_key:
    st.title("Welcome to CSV Data Chat with AI")
    
    # Center the API key input using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write("Please enter your OpenAI API key to begin:")
        api_key = st.text_input("OpenAI API Key", type="password", key="api_key_input")
        if st.button("Submit"):
            if api_key:
                st.session_state.openai_api_key = api_key
                st.rerun()
            else:
                st.error("Please enter an API key")
    st.stop()

# Main chat interface after API key is provided
st.title("CSV Data Chat with AI")

# Add current date to memory
current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.session_state.memory.append({"role": "system", "content": f"Current date: {current_date}"})

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load and store the DataFrame in session state
    if "df" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
        st.session_state["current_file"] = uploaded_file.name
        
        with st.expander("View Dataset Information"):
            # Display schema information
            st.subheader("Dataset Schema")
            schema_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isna().sum()
            })
            st.dataframe(schema_df)
            
            # Display sample data
            st.subheader("Sample Data")
            st.dataframe(df.head())
        
        # Create Langchain agent with user's API key
        st.session_state["agent"] = create_pandas_dataframe_agent(
            ChatOpenAI(
                temperature=0.1, 
                model="gpt-3.5-turbo",
                openai_api_key=st.session_state.openai_api_key
            ),
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
            allow_dangerous_code=True
        )

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask a question about your data:"):
        # Display user message in chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            # Generate and display AI response
            response = st.session_state["agent"].invoke(prompt)
            formatted_response = format_text(response["output"])
            
            st.session_state.messages.append(
                {"role": "assistant", "content": formatted_response}
            )
            with st.chat_message("assistant"):
                st.markdown(formatted_response)

            # Update memory with the last few messages
            st.session_state.memory = st.session_state.memory[-4:] + [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": formatted_response},
            ]
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message}
            )
            with st.chat_message("assistant"):
                st.markdown(error_message)

else:
    st.info("👆 Please upload a CSV file to begin.")

# Add a small footer with reset option
with st.sidebar:
    if st.button("Reset API Key"):
        clear_api_key()
        st.rerun()
