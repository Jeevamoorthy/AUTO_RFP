import streamlit as st
import os
from rag import (
    build_knowledge_base,
    generate_proposal,
    extract_rfp_text,
    research_competitors,
    extract_emails,
    get_client_name,
    generate_email_body,
    send_real_email
)
from utils import save_to_word

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Proposera AI | Neural Midnight",
    layout="wide",
    page_icon="💠"
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>
.stApp {
    background-color: #0A101E;
    background-image:
        linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
    background-size: 50px 50px;
    color: #E6F1FF;
}

h1 {
    color: #00E0FF;
    text-align: center;
    font-weight: 800;
    text-shadow: 0 0 15px rgba(0,224,255,0.4);
}

.form-card {
    background-color: #121E33;
    border: 1px solid #1F2A44;
    padding: 40px;
    border-radius: 14px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    max-width: 900px;
    margin: 0 auto;
}

.stButton>button {
    background: linear-gradient(90deg, #007BFF, #00E0FF) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 14px !important;
    font-weight: 700 !important;
    width: 100%;
}

.preview-box {
    height: 420px;
    overflow-y: scroll;
    background-color: #0E1627;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #1F2F4A;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #
with st.sidebar:

    st.markdown("## 📬 Dispatch Control")

    sender_mail = st.text_input("Gmail Address")
    app_pass = st.text_input("App Password (16-digit Gmail)", type="password")
    st.caption("Generate at https://myaccount.google.com/apppasswords")

    st.divider()

    st.markdown("## 🔐 LLM Provider Control")

    provider = st.selectbox(
        "Select AI Provider",
        ["Gemini (Default)", "Gemini (Custom Key)", "OpenAI", "Claude"]
    )

    # Version Dropdown
    if "Gemini" in provider:
        model_version = st.selectbox(
            "Gemini Model",
            [ "gemini-2.5-flash", "gemini-2.0-flash","gemini-2.5-pro"]
        )

    elif provider == "OpenAI":
        model_version = st.selectbox(
            "OpenAI Model",
            ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
        )

    elif provider == "Claude":
        model_version = st.selectbox(
            "Claude Model",
            ["claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
        )

    custom_api_key = None
    if provider != "Gemini (Default)":
        custom_api_key = st.text_input("API Key Override", type="password")

    temp = st.slider("Neural Creativity", 0.0, 1.0, 0.3)

# ---------------- HEADER ---------------- #
st.markdown("<h1>Proposera AI</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#9CB3D1;'>Autonomous Enterprise Proposal Engineering</p>",
    unsafe_allow_html=True
)

# Layout Center
_, col_center, _ = st.columns([1,2,1])

with col_center:

    # -------- Knowledge Base -------- #
    with st.expander("⚙️ System Configuration (Knowledge Base)"):
        kb_files = st.file_uploader(
            "Upload Company PDFs",
            accept_multiple_files=True,
            type="pdf"
        )

        if st.button("Optimize Neural Brain"):
            if kb_files:
                os.makedirs("data/knowledge_base", exist_ok=True)

                for f in kb_files:
                    with open(os.path.join("data/knowledge_base", f.name), "wb") as out:
                        out.write(f.getbuffer())

                st.session_state['vectorstore'] = build_knowledge_base()
                st.success("Neural Brain Ready.")
            else:
                st.warning("Upload knowledge files first.")

    # -------- Main Card -------- #
    st.markdown("<div class='form-card'>", unsafe_allow_html=True)
    st.markdown("### Autonomous RFP Intake")

    rfp_files = st.file_uploader(
        "Inbound Client RFPs (Multiple Supported)",
        type="pdf",
        accept_multiple_files=True
    )

    # -------- EXECUTION -------- #
    if st.button("⚡ EXECUTE NEURAL SEQUENCE"):

        if rfp_files and 'vectorstore' in st.session_state:

            # Provider Setup
            if provider == "OpenAI":
                selected_provider = "openai"
                if custom_api_key:
                    os.environ["OPENAI_API_KEY"] = custom_api_key

            elif provider == "Claude":
                selected_provider = "claude"
                if custom_api_key:
                    os.environ["ANTHROPIC_API_KEY"] = custom_api_key

            else:
                selected_provider = "gemini"
                if provider == "Gemini (Custom Key)" and custom_api_key:
                    os.environ["GOOGLE_API_KEY"] = custom_api_key

            with st.status("Processing Pipeline...", expanded=True) as status:

                results = []

                for f in rfp_files:

                    status.write(f"Analyzing: {f.name}")

                    os.makedirs("data/rfp_input", exist_ok=True)
                    path = os.path.join("data/rfp_input", f.name)

                    with open(path, "wb") as out:
                        out.write(f.getbuffer())

                    text = extract_rfp_text(path)
                    client = get_client_name(text, model_version)

                    status.write("Researching market...")
                    web_data = research_competitors(text)

                    status.write("Generating proposal...")
                    prop = generate_proposal(
                        text,
                        st.session_state['vectorstore'],
                        web_data,
                        model_version,
                        temp,
                        client,
                        provider=selected_provider
                    )

                    email_body = generate_email_body(
                        prop,
                        model_version,
                        client,
                        provider=selected_provider
                    )

                    doc_path = save_to_word(
                        prop,
                        f"output/Proposal_{f.name}.docx"
                    )

                    results.append({
                        "file": f.name,
                        "client": client,
                        "email": extract_emails(text),
                        "text": prop,
                        "email_body": email_body,
                        "doc": doc_path
                    })

                status.update(label="Complete.", state="complete")

            st.session_state['batch_results'] = results

        elif 'vectorstore' not in st.session_state:
            st.error("Optimize Neural Brain first.")
        else:
            st.error("Upload RFP files.")

    # -------- Results -------- #
    if 'batch_results' in st.session_state:

        st.divider()

        for res in st.session_state['batch_results']:

            with st.expander(f"{res['client']} | Ready", expanded=True):

                colA, colB = st.columns(2)

                with colA:
                    with open(res['doc'], "rb") as f:
                        st.download_button(
                            "Download .DOCX",
                            f,
                            file_name=res['doc']
                        )

                with colB:
                    if st.button(f"Dispatch to {res['email']}", key=res['file']):
                        if not sender_mail or not app_pass:
                            st.error("Provide Gmail + App Password in sidebar.")
                        else:
                            msg = send_real_email(
                                res['email'],
                                f"Proposal for {res['client']}",
                                res['email_body'],
                                res['doc'],
                                sender_mail,
                                app_pass
                            )

                            if msg is True:
                                st.success("Email Dispatched.")
                            else:
                                st.error(msg)

                st.markdown(
                    f"<div class='preview-box'>{res['text']}</div>",
                    unsafe_allow_html=True
                )

    st.markdown("</div>", unsafe_allow_html=True)
