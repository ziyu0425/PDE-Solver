"""
Web Interface for Multi-Agent PDE Solver
========================================
A Streamlit-based chatbot interface for solving PDE problems using natural language.
"""

import os
import asyncio
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from multi_agent_orchestrator import MultiAgentOrchestrator

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDE Solver Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize orchestrator
@st.cache_resource
def get_orchestrator():
    """Initialize and cache the orchestrator."""
    return MultiAgentOrchestrator()

orchestrator = get_orchestrator()


async def process_query(message):
    """
    Process a user query through the multi-agent system.
    
    Args:
        message: User's message/query
    
    Returns:
        Dictionary with response message, html_path, and status
    """
    if not message.strip():
        return {"response": "", "html_path": None, "status": "empty"}
    
    try:
        # Call the orchestrator to solve the PDE problem
        result = await orchestrator.solve(message)
        
        # Handle greeting/non-PDE responses
        if result.get("status") == "greeting":
            return {
                "response": result.get("response", ""),
                "html_path": None,
                "data_file": None,
                "status": "greeting"
            }
        
        # Handle non-PDE query responses (validated by LLM)
        if result.get("status") == "not_pde":
            return {
                "response": result.get("response", ""),
                "html_path": None,
                "data_file": None,
                "status": "not_pde"
            }
        
        if "error" in result:
            # Error occurred
            error_msg = result["error"]
            response = f"**Error:** {error_msg}\n\n"
            
            if "pde_params" in result:
                response += "**Parsed parameters (partial):**\n"
                pde_params = result["pde_params"]
                response += f"- PDE Type: {pde_params.pde_type}\n"
                response += f"- Dimension: {pde_params.dim}D\n"
                if pde_params.domain_size:
                    response += f"- Domain: {pde_params.domain_size}\n"
            
            return {
                "response": response,
                "html_path": None,
                "data_file": None,
                "status": "error"
            }
        
        # Success - get summary and visualization path
        summary = result.get("summary", "Simulation completed successfully.")
        html_path = result.get("html_path")
        data_file = result.get("data_file")
        
        response = f"**Simulation Complete**\n\n{summary}\n\n"
        
        if html_path and Path(html_path).exists():
            return {
                "response": response,
                "html_path": html_path,
                "data_file": data_file,
                "status": "success"
            }
        else:
            response += "⚠️ Visualization file not found."
            return {
                "response": response,
                "html_path": None,
                "data_file": data_file,
                "status": "success_no_viz"
            }
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        return {
            "response": f"**Error:** {error_msg}",
            "html_path": None,
            "data_file": None,
            "status": "error"
        }


def run_async(coro):
    """Run async function in sync context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def main():
    """Main Streamlit app."""
    
    # Header
    st.title("🤖 Multi-Agent PDE Solver Chatbot")
    st.markdown("""
    Solve partial differential equations (PDEs) using natural language!
    
    ### Supported Problems:
    - **Heat Equation**: 1D, 2D, 3D transient or steady-state
    - **Elasticity**: 1D, 2D, 3D linear elasticity with stress/strain output
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Clear memory button
        if st.button("🗑️ Clear Memory", type="secondary", use_container_width=True):
            orchestrator.clear_memory()
            st.success("✅ Memory cleared! Previous simulation history has been reset.")
            st.rerun()
        
        st.divider()
        
        st.header("💡 Example Queries")
        example_queries = [
            "Solve 1D heat transfer in a 2 meter rod, left end at 20°C, right end at 0°C",
            "Solve 2D elasticity problem on a 1m x 1m plate with Young's modulus 210 GPa",
            "3D elasticity problem on a 1m x 0.2m x 0.2m cube with gravity",
        ]
        
        for i, example in enumerate(example_queries, 1):
            if st.button(f"Example {i}", key=f"example_{i}", use_container_width=True):
                st.session_state.example_query = example
                st.rerun()
        
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "latest_html" not in st.session_state:
        st.session_state.latest_html = None
    
    # Handle example query - automatically process it through the agent pipeline
    if "example_query" in st.session_state and "processing_example" not in st.session_state:
        query = st.session_state.example_query
        # Mark that we're processing to avoid duplicate processing
        st.session_state.processing_example = True
        
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Process the example query through the agent pipeline (orchestrator.solve)
        # This goes through the same pipeline as user input: parser -> dispatcher -> solver -> plotter
        with st.spinner("Processing example query through agent pipeline..."):
            result = run_async(process_query(query))
            
            # Store assistant response in session state
            message_data = {
                "role": "assistant",
                "content": result["response"]
            }
            if result["html_path"]:
                message_data["html_path"] = result["html_path"]
            if result.get("data_file"):
                message_data["data_file"] = result["data_file"]
            
            st.session_state.messages.append(message_data)
            if result["html_path"]:
                st.session_state.latest_html = result["html_path"]
        
        # Clean up
        del st.session_state.example_query
        del st.session_state.processing_example
        
        # Rerun to display the results
        st.rerun()
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "html_path" in message and message["html_path"]:
                    html_path = message["html_path"]
                    if Path(html_path).exists():
                        # Display the HTML visualization inline
                        with open(html_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                            st.markdown("### 📊 Visualization")
                            components.html(html_content, height=600, scrolling=True)
                        
                        # Download buttons
                        # Download data file if available in message
                        if "data_file" in message and message["data_file"] and Path(message["data_file"]).exists():
                            # Use two columns if we have both files
                            col1, col2 = st.columns(2)
                            with col1:
                                # Download visualization HTML
                                with open(html_path, 'rb') as f:
                                    st.download_button(
                                        label="📥 Download Visualization HTML",
                                        data=f.read(),
                                        file_name=Path(html_path).name,
                                        mime="text/html",
                                        key=f"download_html_history_{idx}",
                                        use_container_width=True
                                    )
                            with col2:
                                data_path = message["data_file"]
                                with open(data_path, 'rb') as f:
                                    st.download_button(
                                        label="📥 Download Data (Pickle)",
                                        data=f.read(),
                                        file_name=Path(data_path).name,
                                        mime="application/octet-stream",
                                        key=f"download_data_history_{idx}",
                                        use_container_width=True
                                    )
                        else:
                            # Only HTML download button
                            with open(html_path, 'rb') as f:
                                st.download_button(
                                    label="📥 Download Visualization HTML",
                                    data=f.read(),
                                    file_name=Path(html_path).name,
                                    mime="text/html",
                                    key=f"download_html_history_{idx}",
                                    use_container_width=True
                                )
    
    # Chat input
    if prompt := st.chat_input("Describe your PDE problem in natural language..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process query and display response
        with st.chat_message("assistant"):
            with st.spinner("Processing your query..."):
                result = run_async(process_query(prompt))
                
                # Display response
                if result["status"] == "greeting":
                    st.info(result["response"])
                elif result["status"] == "not_pde":
                    st.warning(result["response"])
                elif result["status"] == "error":
                    st.error(result["response"])
                elif result["status"] == "success" or result["status"] == "success_no_viz":
                    st.success("✅ Simulation completed!")
                    st.markdown(result["response"])
                    
                    # Display visualization inline if available
                    if result["html_path"] and Path(result["html_path"]).exists():
                        html_path = result["html_path"]
                        st.markdown("### 📊 Visualization")
                        with open(html_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                            components.html(html_content, height=600, scrolling=True)
                        
                        # Download buttons
                        next_idx = len(st.session_state.messages) + 1
                        
                        # Download data file if available
                        if result.get("data_file") and Path(result["data_file"]).exists():
                            # Use two columns if we have both files
                            col1, col2 = st.columns(2)
                            with col1:
                                # Download visualization HTML
                                with open(html_path, 'rb') as f:
                                    st.download_button(
                                        label="📥 Download Visualization HTML",
                                        data=f.read(),
                                        file_name=Path(html_path).name,
                                        mime="text/html",
                                        key=f"download_html_new_{next_idx}",
                                        use_container_width=True
                                    )
                            with col2:
                                data_path = result["data_file"]
                                with open(data_path, 'rb') as f:
                                    st.download_button(
                                        label="📥 Download Data (Pickle)",
                                        data=f.read(),
                                        file_name=Path(data_path).name,
                                        mime="application/octet-stream",
                                        key=f"download_data_new_{next_idx}",
                                        use_container_width=True
                                    )
                        else:
                            # Only HTML download button
                            with open(html_path, 'rb') as f:
                                st.download_button(
                                    label="📥 Download Visualization HTML",
                                    data=f.read(),
                                    file_name=Path(html_path).name,
                                    mime="text/html",
                                    key=f"download_html_new_{next_idx}",
                                    use_container_width=True
                                )
                else:
                    st.info(result["response"])
                
                # Store in session state
                message_data = {
                    "role": "assistant",
                    "content": result["response"]
                }
                if result["html_path"]:
                    message_data["html_path"] = result["html_path"]
                if result.get("data_file"):
                    message_data["data_file"] = result["data_file"]
                
                st.session_state.messages.append(message_data)
                if result["html_path"]:
                    st.session_state.latest_html = result["html_path"]


if __name__ == "__main__":
    # Check if OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        st.warning(
            "⚠️ **Warning:** OPENAI_API_KEY not found in environment variables.\n\n"
            "Please set it in a .env file or export it:\n"
            "`export OPENAI_API_KEY=your_api_key_here`\n\n"
            "The interface will still launch but queries may fail."
        )
    
    main()
