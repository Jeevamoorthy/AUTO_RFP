import streamlit as st
import os, time, urllib.parse
from rag import build_knowledge_base, generate_proposal, extract_rfp_text, research_competitors, extract_emails, get_client_name, generate_email_body, send_real_email
from utils import save_to_word
import subprocess
import sys

# Hotfix for DuckDuckGo Search error
try:
    from duckduckgo_search import DDGS
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "duckduckgo-search==6.3.0"])

# --- PAGE CONFIG ---
st.set_page_config(page_title="Proposera AI | Neural Midnight", layout="wide", page_icon="💠")

# --- CUSTOM CSS: NEURAL MIDNIGHT BRANDING ---
st.markdown("""
    <style>
        /* 1. Global Background with AI Grid Overlay */
        .stApp {
            background-color: #0A101E;
            background-image: 
                linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
            background-size: 50px 50px;
            color: #E6F1FF;
        }

        /* 2. Header & Branding */
        h1 {
            color: #00E0FF; 
            font-family: 'Inter', sans-serif; 
            text-align: center; 
            font-weight: 800; 
            text-shadow: 0 0 15px rgba(0,224,255,0.4);
            margin-bottom: 0px;
        }
        .subtitle {
            color: #9CB3D1;
            text-align: center;
            font-size: 15px;
            margin-bottom: 40px;
        }

        /* 3. The Central "Neural Card" */
        .form-card {
            background-color: #121E33;
            border: 1px solid #1F2A44;
            padding: 40px;
            border-radius: 14px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            max-width: 850px;
            margin: 0 auto;
        }

        /* 4. Sidebar Styles */
        [data-testid="stSidebar"] {
            background-color: #0A101E;
            border-right: 1px solid #1F2A44;
        }
        [data-testid="stSidebar"] h2 { color: #00E0FF; }

        /* 5. Input Fields & Selectboxes */
        .stTextInput>div>div>input, .stSelectbox>div>div>div, .stSlider>div>div {
            background-color: #0E1627 !important;
            color: #E6F1FF !important;
            border: 1px solid #1F2F4A !important;
            border-radius: 8px;
        }
        .stTextInput>div>div>input:focus {
            border-color: #00B3FF !important;
            box-shadow: 0 0 10px rgba(0,179,255,0.3) !important;
        }

        /* 6. Primary Action Buttons (Neon Gradient) */
        .stButton>button {
            background: linear-gradient(90deg, #007BFF, #00E0FF) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 14px !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
            width: 100%;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(0, 200, 255, 0.5);
        }

        /* 7. Output Preview Scrollbox */
        .preview-box {
            height: 450px; 
            overflow-y: scroll; 
            background-color: #0E1627; 
            padding: 25px; 
            border-radius: 10px; 
            border: 1px solid #1F2F4A; 
            color: #E6F1FF; 
            font-family: 'Inter', sans-serif;
            line-height: 1.6;
        }

        /* Expander Styling */
        .streamlit-expanderHeader {
            background-color: #121E33 !important;
            border: 1px solid #1F2A44 !important;
            border-radius: 8px !important;
            color: #E6F1FF !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR: DISPATCH SETTINGS ---
with st.sidebar:
    st.markdown("## 📬 Dispatch Control")
    sender_mail = st.text_input("Gmail Address", value="jeevamissvmins34@gmail.com")
    app_pass = st.text_input("App Password", type="password", help="Gmail > Security > App Passwords")
    st.divider()
    st.markdown("## 🧠 Reasoning Model")
    model = st.selectbox("Select Intelligence Model", [
            "deep-research-pro-preview-12-2025",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-pro-latest",
            "gemini-2.5-flash"
            ])
    temp = st.slider("Neural Creativity", 0.0, 1.0, 0.3)

# --- MAIN PAGE HEADER ---
st.markdown("<h1>Proposera <span style='color:white'>AI</span></h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Autonomous Enterprise Proposal Engineering • Neural Midnight Engine</p>", unsafe_allow_html=True)

# --- LAYOUT CENTERING ---
_, col_center, _ = st.columns([1, 2, 1])

with col_center:
    # 💎 Knowledge Base Ingestion
    with st.expander("⚙️ System Configuration (Enterprise Brain)"):
        st.caption("Step 1: Upload company context to train the autonomous reasoning engine.")
        kb_files = st.file_uploader("Upload Company PDFs", accept_multiple_files=True, type="pdf")
        if st.button("Optimize Neural Brain"):
            if kb_files:
                os.makedirs("data/knowledge_base", exist_ok=True)
                for f in kb_files:
                    with open(os.path.join("data/knowledge_base", f.name), "wb") as file: file.write(f.getbuffer())
                st.session_state['vectorstore'] = build_knowledge_base()
                st.success("✅ Neural Brain Optimized and Ready.")
            else:
                st.warning("Please upload files first.")

    # ⚡ The Main Workflow Card
    st.markdown("<div class='form-card'>", unsafe_allow_html=True)
    st.markdown("### **Autonomous Mission Intake**")
    rfp_files = st.file_uploader("Inbound Client RFPs (Multiple OK)", type="pdf", accept_multiple_files=True)
    
    if st.button("⚡ EXECUTE NEURAL SEQUENCE"):
        if rfp_files and 'vectorstore' in st.session_state:
            with st.status("🚀 Processing Autonomous Pipeline...", expanded=True) as status:
                results = []
                for f in rfp_files:
                    status.write(f"📂 **Analyzing:** {f.name}")
                    path = os.path.join("data/rfp_input", f.name)
                    os.makedirs("data/rfp_input", exist_ok=True)
                    with open(path, "wb") as out: out.write(f.getbuffer())
                    
                    text = extract_rfp_text(path)
                    client = get_client_name(text, model)
                    
                    status.write(f"🌐 **Researching:** Market trends for {client}...")
                    web_data = research_competitors(text)
                    
                    status.write(f"🧠 **Synthesizing:** Response using {model}...")
                    prop = generate_proposal(text, st.session_state['vectorstore'], web_data, model, temp, client)
                    email_body = generate_email_body(prop, model, client)
                    
                    results.append({
                        "file": f.name, 
                        "client": client, 
                        "email": extract_emails(text), 
                        "text": prop, 
                        "email_body": email_body, 
                        "doc": save_to_word(prop, f"output/Proposal_{f.name}.docx")
                    })
                    
                status.update(label="✅ Sequence Complete. Proposals Generated.", state="complete", expanded=False)
            st.session_state['batch_results'] = results
        elif 'vectorstore' not in st.session_state:
            st.error("Neural Brain not optimized. Please upload knowledge base first.")
        else:
            st.error("No RFP documents uploaded.")

    # 📊 Output Dashboard
    if 'batch_results' in st.session_state:
        st.divider()
        for res in st.session_state['batch_results']:
            with st.expander(f"✨ Result: {res['client']} | Protocol Complete", expanded=True):
                ca, cb = st.columns(2)
                with ca: 
                    with open(res['doc'], "rb") as f:
                        st.download_button("📥 Download .DOCX", f, file_name=f"Proposal_{res['file']}.docx")
                with cb:
                    if st.button(f"🚀 Dispatch to {res['email']}", key=f"btn_{res['file']}"):
                        if not app_pass: 
                            st.error("Enter App Password in Sidebar!")
                        else:
                            with st.spinner("Dispatching via Secure SMTP..."):
                                msg = send_real_email(res['email'], f"Proposal for {res['client']}", res['email_body'], res['doc'], sender_mail, app_pass)
                                if msg is True: 
                                    st.success("✅ Dispatched with Attachment!")
                                else: 
                                    st.error(f"Dispatch Error: {msg}")
                
                # Full Scrollable Preview
                st.markdown(f"<div class='preview-box'>{res['text']}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
