import streamlit as st
from dotenv import load_dotenv
from firebase_client import sign_in, sign_up
from rag import load_guide, embed_texts, retrieve, build_context, generate_answer

load_dotenv()
st.set_page_config(page_title="Lumen Travel Assistant", page_icon="🧭")

# Load and embed the Knowledge Base only once
@st.cache_resource
def init_knowledge_base():
    records = load_guide("knowledge_base.json")
    vectors = embed_texts([f"{r['title']}: {r['text']}" for r in records])
    return records, vectors

# Initialize session_state
if "user" not in st.session_state:
    st.session_state.user = None
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🧭 Lumen Travel City RAG Assistant")

# Sidebar: Login / Signup / Logout
if st.session_state.user is None:
    st.info("Please sign in to access the travel assistant.")
    tab_login, tab_signup = st.tabs(["Sign In", "Sign Up"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In")
            if submitted:
                if not email.strip() or not password.strip():
                    st.warning("Please enter both email and password.")
                else:
                    try:
                        st.session_state.user = sign_in(email.strip(), password)
                        st.session_state.messages = []
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

    with tab_signup:
        with st.form("signup_form"):
            new_email = st.text_input("New Email")
            new_password = st.text_input("New Password (min 6 chars)", type="password")
            reg_submitted = st.form_submit_button("Create Account")
            if reg_submitted:
                if not new_email.strip() or not new_password.strip():
                    st.warning("Please fill in all fields.")
                else:
                    try:
                        sign_up(new_email.strip(), new_password)
                        st.success("Account created! Switch to the Sign In tab to log in.")
                    except Exception as e:
                        st.error(str(e))
    st.stop()

# Chat interface only after login
with st.sidebar:
    st.write(f"**Signed in as:** {st.session_state.user['email']}")
    st.caption(f"UID: `{st.session_state.user['uid']}`")
    if st.button("Sign Out"):
        st.session_state.user = None
        st.session_state.messages = []  # Xóa lịch sử chat khi đăng xuất
        st.rerun()

# Load and cache vector guide
try:
    records, doc_vectors = init_knowledge_base()
except Exception as e:
    st.error(f"Failed to initialize knowledge base (check Ollama/Pinggy URL): {e}")
    st.stop()

# Display chat history for the current session
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("Retrieved Guide Entries & Cosine Scores"):
                for h in msg["sources"]:
                    st.markdown(f"**[{h['id']}] {h['title']}** — Score: `{h['score']:.4f}`")
                    st.caption(h["text"])

# Receive new user input
question = st.chat_input("Ask about attractions, opening hours, or prices...")
if question is not None:
    if not question.strip():
        st.warning("Please enter a non-empty question.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching travel guide and generating answer..."):
                try:
                    hits = retrieve(question, records, doc_vectors, k=3)
                    context = build_context(hits)
                    answer = generate_answer(question, context)
                    st.markdown(answer)
                    with st.expander("Retrieved Guide Entries & Cosine Scores"):
                        for h in hits:
                            st.markdown(f"**[{h['id']}] {h['title']}** — Score: `{h['score']:.4f}`")
                            st.caption(h["text"])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": hits,
                    })
                except Exception as e:
                    st.error(f"Error: {e}")
                    