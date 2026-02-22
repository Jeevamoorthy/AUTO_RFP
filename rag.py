import os, re, smtplib, unicodedata, streamlit as st
from email import encoders
from email.header import Header
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from duckduckgo_search import DDGS 

load_dotenv()

def get_llm(model_name, temp, user_key=None):
    if "gpt" in model_name.lower():
        key = user_key if user_key else st.secrets.get("OPENAI_API_KEY")
        return ChatOpenAI(model=model_name, temperature=temp, api_key=key)
    elif "claude" in model_name.lower():
        key = user_key if user_key else st.secrets.get("ANTHROPIC_API_KEY")
        return ChatAnthropic(model=model_name, temperature=temp, api_key=key)
    else:
        key = user_key if user_key else (st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        if not key: raise ValueError("Invalid Gemini API Key")
        return ChatGoogleGenerativeAI(model=model_name, temperature=temp, google_api_key=key, max_output_tokens=4000)

def build_knowledge_base(user_key=None):
    kb_path, docs = "/tmp/knowledge_base", []
    current_key = user_key if user_key else (st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    if not current_key: raise ValueError("No API Key found. Please enter it in the sidebar.")
    
    # Resilient Embedding Hunter
    embeddings_model = None
    for m in ["text-embedding-004", "models/text-embedding-004", "models/embedding-001"]:
        try:
            test = GoogleGenerativeAIEmbeddings(model=m, google_api_key=current_key)
            test.embed_query("ping")
            embeddings_model = test; break
        except: continue
    
    if not embeddings_model: raise ValueError("Invalid Google Key for Embeddings. Ensure the key has Gemini API access.")
    if not os.path.exists(kb_path): return None
    for file in os.listdir(kb_path):
        if file.endswith(".pdf"): docs.extend(PyPDFLoader(os.path.join(kb_path, file)).load())
    if not docs: return None
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    return FAISS.from_documents(documents=splits, embedding=embeddings_model)

# (Keep your other helper functions: extract_emails, get_client_name, research_competitors, generate_proposal, generate_email_body, send_real_email, extract_rfp_text)
def extract_emails(text):
    """Finds client email address from text."""
    email_pattern = r"[\w\.-]+@[\w\.-]+\.\w+"
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else "client@example.com"

def get_client_name(text, model_name, user_key=None):
    llm = get_llm(model_name, 0, user_key)
    prompt = f"Identify the recipient company name from this RFP. Return ONLY the name. No notes. If missing return 'Prospective Client'. \n\n Text: {text[:2000]}"
    try:
        name = llm.invoke(prompt).content.strip()
        return name if len(name.split()) <= 5 else "Prospective Client"
    except: return "Prospective Client"

def research_competitors(rfp_text):
    try:
        with DDGS() as ddgs:
            res = [r for r in ddgs.text(f"Market pricing for {rfp_text[:50]}", max_results=2)]
            return "\n".join([f"{r['title']}: {r['body']}" for r in res])
    except: return "Market data unavailable."

def generate_proposal(rfp_text, vectorstore, web_data, model_name, temp, client_name, user_key=None):
    llm = get_llm(model_name, temp, user_key)
    system_prompt = (f"Autonomous AI. Draft B2B proposal for {client_name}. STRICT: NEVER use 'Your Organization'. ALWAYS use {client_name}. Format with Markdown. \n\nContext: {{context}} \nWeb Data: {{web_data}}")
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "Draft for:\n{input}")])
    chain = ({"context": vectorstore.as_retriever() | (lambda ds: "\n\n".join(d.page_content for d in ds)), "web_data": lambda x: web_data, "input": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    return chain.invoke(rfp_text)

def generate_email_body(proposal_text, model_name, client_name, user_key=None):
    llm = get_llm(model_name, 0.5, user_key)
    prompt = f"Write a professional 3-sentence email to {client_name} about the attached proposal. \n\n Proposal: {proposal_text[:500]}"
    try: return str(llm.invoke(prompt).content)
    except: return f"Dear {client_name}, please find our proposal attached."

def send_real_email(recipient_email, subject, body, attachment_path, sender_email, app_password):
    try:
        def clean(t): return "".join(c for c in unicodedata.normalize('NFKD', str(t)) if ord(c) < 128)
        msg = MIMEMultipart()
        msg['From'], msg['To'], msg['Subject'] = sender_email, recipient_email, Header(clean(subject), 'utf-8')
        msg.attach(MIMEText(clean(body), 'plain'))
        with open(attachment_path, "rb") as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read()); encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(attachment_path)}"')
            msg.attach(part)
        s = smtplib.SMTP('smtp.gmail.com', 587); s.starttls(); s.login(sender_email, app_password)
        s.sendmail(sender_email, recipient_email, msg.as_string()); s.quit()
        return True
    except Exception as e: return str(e)

def extract_rfp_text(path):
    return "\n".join([d.page_content for d in PyPDFLoader(path).load()])
