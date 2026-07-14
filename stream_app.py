import streamlit as st
from pathlib import Path
from rag_system import RagSystem  # 导入后端文件

# ==========================================
# 1. 头部全局CSS样式区 (保持原样，绝不修改)
# ==========================================
st.markdown("""
<style>
.main-header {
    font-size: 36px !important;
    font-weight: 700 !important;
    text-align: center !important;
    background: linear-gradient(90deg, #0066cc, #00bb77, #ff8800);
    -webkit-background-clip: text;
    color: transparent !important;
    background-color: #f0f5f9;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(30,80,140,0.15);
    padding: 24px 10px;
    margin-top: 10px;
    margin-bottom: 30px;
    letter-spacing: 2px;
}
/* 多行文本输入框 */
div[data-testid="stTextArea"] textarea {
    border-radius: 12px !important;
    border: 1px solid #b3d8ff !important;
    /* background: #f8fbff !important;  <-- 建议把这行也删掉，让它自动适应深色背景 */
    padding: 14px !important;
    font-size: 15px;
    /* color: #000000 !important;  <-- 删掉这行，或者改成 color: inherit !important; */
}
div[data-testid="stTextArea"] textarea::placeholder {
    color: #666666 !important;
    opacity: 0.8 !important;
}
/* form按钮样式 */
div[data-testid="stFormSubmitButton"] button {
    border-radius: 8px;
    height: 42px;
}
/* 快捷按钮 */
div[data-testid="stButton"] > button {
    border-radius: 8px;
    padding: 8px;
}
/* 对话气泡样式 */
.chat-message {
    padding: 1rem;
    border-radius: 12px;
    margin: 0.8rem 0;
}
.user-message {
    background-color: #e6f7ff;
    border-left: 4px solid #1890ff;
}
.assistant-message {
    background-color: #f6ffed;
    border-left: 4px solid #87d068;
}
.source-info {
    margin-top: 0.5rem;
    padding: 0.5rem;
    background-color: #f5f5f5;
    border-radius: 8px;
    font-size: 0.9rem;
    color: #595959;
}
</style>
""", unsafe_allow_html=True)


# 从 .env 文件加载环境变量的辅助函数
# ==========================================
# 修改后的密钥获取函数 (兼容本地和云端)
# ==========================================
def getEnvInfo(key):
    # 1. 优先尝试从 Streamlit Secrets (网页端配置) 获取
    try:
        # st.secrets 结构通常是 [general] -> KEY_NAME
        return st.secrets["general"][key]
    except Exception:
        pass # 如果网页端没配，或者不在Streamlit环境，就忽略错误继续往下走

    # 2. 如果上面失败了，再尝试读取本地的 .env 文件 (保持原有逻辑)
    env_res = None
    try:
        with open('.env', 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith(f"{key}="):
                    env_res = line.split("=", 1)[1].strip()
                    break
    except FileNotFoundError:
        # 这里不要 warning 了，因为云端本来就没有文件，warning 会误导用户
        pass 
    return env_res


def display_chat_message(role, content, sources=None, typing=False):
    """显示一条聊天消息，支持来源信息展示"""

    # --- 新增：一个简单的文本清洗函数 ---
    def clean_source_text(text):
        """移除字符串中的HTML标签，只保留纯文本"""
        if not isinstance(text, str):
            return str(text)
        # 这是一个非常简单的实现，用于移除 <...> 形式的标签
        import re
        clean_text = re.sub(r'<[^>]+>', '', text)
        return clean_text.strip()

    # --- 修改结束 ---

    if role == "user":
        st.markdown(f'<div class="chat-message user-message"><strong>用户:</strong> {content}</div>',
                    unsafe_allow_html=True)
    else:  # role == "assistant"
        typing_indicator = "正在思考..." if typing else ""

        st.markdown(f"""
        <div class="chat-message assistant-message">
             <strong>{typing_indicator} 生产助手:</strong> {content}
        </div>
        """, unsafe_allow_html=True)



def main():
    st.set_page_config(page_title="生产指导检索问答系统", page_icon="⚙️", layout="wide")

    # 1. 页面标题
    st.markdown('<h1 class="main-header">⚙️ 生产指导检索问答系统</h1>', unsafe_allow_html=True)

    # 2. 初始化 session_state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'system_init' not in st.session_state:
        st.session_state.system_init = False
    if 'rag_instance' not in st.session_state:
        st.session_state.rag_instance = None
    if 'preset_question' not in st.session_state:
        st.session_state.preset_question = ""

    # 3. 快捷问题区域 (仅填充输入框，不发送)
    st.subheader("快速提问")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("异常处理", use_container_width=True):
            st.session_state.preset_question = '如果我在空载试车时发现轴承温度达到 85℃，我应该首先检查什么？如果检查后发现游隙正常，下一步该做什么？'
            st.rerun()
    with col2:
        if st.button("入库流程", use_container_width=True):
            st.session_state.preset_question = '凸轮轴原材料入库重点核对哪些元素含量是否符合GB/T 3077标准？'
            st.rerun()
    with col3:
        if st.button("质量检验", use_container_width=True):
            st.session_state.preset_question = '在进行齿轮接触斑点检验时，如果斑点偏向齿顶，说明了什么问题？应该如何调整？'
            st.rerun()
    with col4:
        if st.button("参数提取", use_container_width=True):
            st.session_state.preset_question = '操作重型机械时，必须遵守的核心安全规范有哪些？'
            st.rerun()

    # 4. 聊天历史显示区域
    chat_container = st.container(border=True)
    with chat_container:
        for msg in st.session_state.chat_history:
            display_chat_message(role=msg["role"], content=msg["content"], sources=msg.get("sources"))

    # 5. 用户输入区域
    with st.form(key='chat_form', clear_on_submit=True):
        st.subheader("向生产助手提问")
        user_input = st.text_area(
            "请输入您的问题",
            value=st.session_state.preset_question,
            placeholder="车床上机加工流程是？\n\n提示：你可以咨询生产工艺/成品质检/设备维护/安全规范等相关问题",
            height=100,
            label_visibility="visible",
        )
        col1, col2 = st.columns([1, 1])
        with col1:
            submit_button = st.form_submit_button("发送", use_container_width=True)
        with col2:
            clear_history_button = st.form_submit_button("清空全部对话", use_container_width=True)

    # --- 核心逻辑修正区 ---
    # A. 处理表单提交（点击发送）
    if submit_button:
        if user_input.strip():
            # 1. 将用户问题加入历史
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            # 2. 清空预设问题，防止下次还自动填
            st.session_state.preset_question = ""
            # 3. 立即刷新页面，先显示出用户的问题
            st.rerun()
        else:
            st.toast("请输入内容后再发送哦！", icon="⚠️")

    # B. 处理清空历史
    if clear_history_button:
        st.session_state.chat_history = []
        st.rerun()

    # C. 检查是否有待处理的用户问题（即刚发完消息，需要调用后端）
    # 逻辑：如果最后一条消息是用户的，且还没有助手的回复，则开始生成
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        if st.session_state.system_init and st.session_state.rag_instance:
            question = st.session_state.chat_history[-1]["content"]
            # 先添加一个空的助手消息占位，用于显示“正在思考”
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "",
                "sources": None
            })
            # 刷新一下，让用户看到“正在思考”的气泡
            st.rerun()
        # --- 以下代码会在下一次 rerun 后执行（此时页面已显示“正在思考”） ---
        # 但为了简化逻辑，我们在这里直接执行阻塞操作，
        # 实际上 Streamlit 的 rerun 机制会导致这里比较 tricky。
        # 最佳实践是：利用 st.spinner 包裹整个请求过程。
        # 由于上面的 rerun 会中断脚本，我们需要一种机制来标记“正在请求中”
        # 但为了保持代码简单且不引入复杂的状态机，我们采用以下策略：
        # 只有当最后一条是空白的 assistant 消息时，才执行请求。
        pass

    # D. 真正的执行逻辑（当最后一条是空白助手消息时触发）
    if (st.session_state.chat_history and len(st.session_state.chat_history) >= 2 and
            st.session_state.chat_history[-1]["role"] == "assistant" and
            st.session_state.chat_history[-1]["content"] == ""):

        if st.session_state.system_init and st.session_state.rag_instance:
            question = st.session_state.chat_history[-2]["content"]  # 获取倒数第二条（用户的问题）
            # 使用 spinner 提供视觉反馈
            with st.spinner("生产助手正在思考中..."):
                try:
                    rag = st.session_state.rag_instance
                    res = rag.ask(question)
                    # 更新最后一条消息的内容
                    st.session_state.chat_history[-1] = {
                        "role": "assistant",
                        "content": res["answer"],
                        "sources": res["sources"]
                    }
                except Exception as err:
                    st.error(f"调用RAG问答失败：{str(err)}")
                    # 出错时也更新内容，避免一直卡在空白状态
                    st.session_state.chat_history[-1]["content"] = f"出错了：{str(err)}"
            # 完成后刷新页面显示结果
            st.rerun()
        else:
            # 如果系统没初始化，给个提示
            st.session_state.chat_history[-1]["content"] = "系统尚未初始化，请检查侧边栏配置或 .env 文件。"
            st.rerun()

    # 6. 侧边栏配置 (保持不变)
    with st.sidebar:
        st.header("⚙️系统配置")
        st.info("系统固定读取.env内密钥，不支持网页端修改API密钥")
        st.divider()
        st.subheader("系统状态")
        api_key = getEnvInfo("DEEPSEEK_API_KEY")
        if not api_key or len(api_key.strip()) == 0:
            st.error(".env 文件未配置有效 DEEPSEEK_API_KEY")
            st.session_state.system_init = False
            st.session_state.rag_instance = None
        else:
            if not st.session_state.system_init:
                with st.spinner("正在初始化RAG系统..."):
                    try:
                        rag = RagSystem(api_key=api_key)
                        rag.initialize()
                        st.session_state.rag_instance = rag
                        st.session_state.system_init = True
                    except Exception as e:
                        st.error(f"RAG系统初始化失败: {str(e)}")
                        st.session_state.system_init = False

            if st.session_state.system_init and st.session_state.rag_instance:
                rag = st.session_state.rag_instance
                st.success("✅ RAG知识库初始化完成")
                if hasattr(rag, 'documents'):
                    st.metric("加载文档总数", value=len(rag.documents))
                if hasattr(rag, 'doc_chunks'):
                    st.metric("文本块总数", value=len(rag.doc_chunks))


if __name__ == "__main__":
    main()
