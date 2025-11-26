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

# ---------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="PDE Solver Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Initialize orchestrator (cached)
# ---------------------------------------------------------------------
@st.cache_resource
def get_orchestrator():
    """Initialize and cache the orchestrator."""
    return MultiAgentOrchestrator()


orchestrator = get_orchestrator()


# ---------------------------------------------------------------------
# Core async processing
# ---------------------------------------------------------------------
async def process_query(message: str):
    """
    Process a user query through the multi-agent system.

    Args:
        message: User's message/query

    Returns:
        dict with keys:
          - response (str)
          - html_path (str or None)
          - data_file (str or None)
          - status (str)
    """
    if not message or not message.strip():
        return {
            "response": "",
            "html_path": None,
            "data_file": None,
            "status": "empty",
        }

    try:
        # Call the orchestrator to solve the PDE problem
        result = await orchestrator.solve(message)

        # Handle greeting / non-PDE responses
        if result.get("status") == "greeting":
            return {
                "response": result.get("response", ""),
                "html_path": None,
                "data_file": None,
                "status": "greeting",
            }

        if result.get("status") == "not_pde":
            return {
                "response": result.get("response", ""),
                "html_path": None,
                "data_file": None,
                "status": "not_pde",
            }

        # Error from orchestrator
        if "error" in result:
            error_msg = result["error"]
            response = f"**Error:** {error_msg}\n\n"

            if "pde_params" in result:
                response += "**Parsed parameters (partial):**\n"
                pde_params = result["pde_params"]
                response += f"- PDE Type: {getattr(pde_params, 'pde_type', 'N/A')}\n"
                response += f"- Dimension: {getattr(pde_params, 'dim', 'N/A')}D\n"
                if getattr(pde_params, "domain_size", None):
                    response += f"- Domain: {pde_params.domain_size}\n"

            return {
                "response": response,
                "html_path": None,
                "data_file": None,
                "status": "error",
            }

        # Success
        summary = result.get("summary", "Simulation completed successfully.")
        html_path = result.get("html_path")
        data_file = result.get("data_file")

        response = f"**Simulation Complete**\n\n{summary}\n\n"

        if html_path and Path(html_path).exists():
            return {
                "response": response,
                "html_path": html_path,
                "data_file": data_file,
                "status": "success",
            }
        else:
            response += "⚠️ Visualization file not found."
            return {
                "response": response,
                "html_path": None,
                "data_file": data_file,
                "status": "success_no_viz",
            }

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        return {
            "response": f"**Error:** {error_msg}",
            "html_path": None,
            "data_file": None,
            "status": "error",
        }


def run_async(coro):
    """Run async function in sync Streamlit context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------
# Shared logic: handle one prompt (from Example or chat_input)
# ---------------------------------------------------------------------
def respond_to_prompt(prompt: str):
    """
    给定一个 prompt（可以是 Example，也可以是用户在 chat_input 输入的），
    1. 作为 user 消息写入 session_state.messages
    2. 调用 orchestrator.solve
    3. 将结果作为 assistant 消息写入 session_state.messages
    4. 在界面上显示可视化 & 下载按钮
    """

    # 1. 写入 user 消息到历史
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. 显示 user 消息
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. 处理并显示 assistant 回复
    with st.chat_message("assistant"):
        with st.spinner("Processing your query..."):
            result = run_async(process_query(prompt))

            status = result.get("status")
            resp_text = result.get("response", "")

            if status == "greeting":
                st.info(resp_text)
            elif status == "not_pde":
                st.warning(resp_text)
            elif status == "error":
                st.error(resp_text)
            elif status in ("success", "success_no_viz"):
                st.success("✅ Simulation completed!")
                st.markdown(resp_text)

                html_path = result.get("html_path")
                data_file = result.get("data_file")

                # 有可视化文件就嵌入 + 下载
                if html_path and Path(html_path).exists():
                    st.markdown("### 📊 Visualization")
                    with open(html_path, "r", encoding="utf-8") as f:
                        html_content = f.read()
                        components.html(html_content, height=600, scrolling=True)

                    # 下载按钮 key 需要唯一，这里用当前历史长度
                    next_idx = len(st.session_state.messages) + 1

                    if data_file and Path(data_file).exists():
                        col1, col2 = st.columns(2)
                        with col1:
                            with open(html_path, "rb") as f_html:
                                st.download_button(
                                    label="📥 Download Visualization HTML",
                                    data=f_html.read(),
                                    file_name=Path(html_path).name,
                                    mime="text/html",
                                    key=f"download_html_new_{next_idx}",
                                    use_container_width=True,
                                )
                        with col2:
                            with open(data_file, "rb") as f_data:
                                st.download_button(
                                    label="📥 Download Data (Pickle)",
                                    data=f_data.read(),
                                    file_name=Path(data_file).name,
                                    mime="application/octet-stream",
                                    key=f"download_data_new_{next_idx}",
                                    use_container_width=True,
                                )
                    else:
                        with open(html_path, "rb") as f_html:
                            st.download_button(
                                label="📥 Download Visualization HTML",
                                data=f_html.read(),
                                file_name=Path(html_path).name,
                                mime="text/html",
                                key=f"download_html_new_{next_idx}",
                                use_container_width=True,
                            )
            else:
                st.info(resp_text)

        # 4. 将 assistant 消息写入历史
        message_data = {
            "role": "assistant",
            "content": resp_text,
        }
        if result.get("html_path"):
            message_data["html_path"] = result["html_path"]
        if result.get("data_file"):
            message_data["data_file"] = result["data_file"]

        st.session_state.messages.append(message_data)

        if result.get("html_path"):
            st.session_state.latest_html = result["html_path"]


# ---------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------
def main():
    # Header
    st.title("🤖 Multi-Agent PDE Solver Chatbot")
    st.markdown(
        """
Solve partial differential equations (PDEs) using natural language!

### Supported Problems:
- **Heat Equation**: 1D, 2D, 3D transient or steady-state
- **Elasticity**: 1D, 2D, 3D linear elasticity with stress/strain output
"""
    )

    # ---------------- Sidebar ----------------
    example_clicked = None  # 记录本次运行是否点击了某个 Example

    with st.sidebar:
        st.header("⚙️ Controls")

        # Clear memory
        if st.button("🗑️ Clear Memory", type="secondary", use_container_width=True):
            orchestrator.clear_memory()
            # 同时清空前端对话历史
            if "messages" in st.session_state:
                st.session_state.messages = []
            st.success("✅ Memory cleared! Previous simulation history has been reset.")
            st.rerun()

        st.divider()

        st.header("💡 Example Queries")
        example_queries = [
            "Solve 1D heat transfer in a 2 meter rod, left end at 20°C, right end at 0°C",
            "Solve 2D elasticity problem on a 1m x 1m plate with Young's modulus 210 GPa",
            "3D elasticity problem on a 1m x 0.2m x 0.2m cube with gravity",
        ]

        for i, example in enumerate(example_queries, start=1):
            if st.button(f"Example {i}", key=f"example_{i}", use_container_width=True):
                # 不马上处理，只记录是哪条 example；主区域统一调用 respond_to_prompt
                example_clicked = example

    # ---------------- Session state init ----------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "latest_html" not in st.session_state:
        st.session_state.latest_html = None

    # ---------------- Display chat history ----------------
    chat_container = st.container()
    with chat_container:
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # 如果这条 assistant 消息有可视化 html，就再嵌入一次
                html_path = message.get("html_path")
                data_file = message.get("data_file")

                if html_path and Path(html_path).exists():
                    st.markdown("### 📊 Visualization")
                    with open(html_path, "r", encoding="utf-8") as f:
                        html_content = f.read()
                        components.html(html_content, height=600, scrolling=True)

                    # 下载按钮
                    if data_file and Path(data_file).exists():
                        col1, col2 = st.columns(2)
                        with col1:
                            with open(html_path, "rb") as f_html:
                                st.download_button(
                                    label="📥 Download Visualization HTML",
                                    data=f_html.read(),
                                    file_name=Path(html_path).name,
                                    mime="text/html",
                                    key=f"download_html_history_{idx}",
                                    use_container_width=True,
                                )
                        with col2:
                            with open(data_file, "rb") as f_data:
                                st.download_button(
                                    label="📥 Download Data (Pickle)",
                                    data=f_data.read(),
                                    file_name=Path(data_file).name,
                                    mime="application/octet-stream",
                                    key=f"download_data_history_{idx}",
                                    use_container_width=True,
                                )
                    else:
                        with open(html_path, "rb") as f_html:
                            st.download_button(
                                label="📥 Download Visualization HTML",
                                data=f_html.read(),
                                file_name=Path(html_path).name,
                                mime="text/html",
                                key=f"download_html_history_{idx}",
                                use_container_width=True,
                            )

    # ---------------- New input: Example or chat_input ----------------
    # 底部输入框
    prompt = st.chat_input("Describe your PDE problem in natural language...")

    # 优先处理 Example 点击，其次处理用户输入
    if example_clicked is not None:
        respond_to_prompt(example_clicked)
    elif prompt:
        respond_to_prompt(prompt)


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
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

