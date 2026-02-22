import os
import re
import smtplib
import unicodedata
import streamlit as st
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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from duckduckgo_search import DDGS 

load_dotenv()

# Secure Key Loading
api_key = os.getenv("GOOGLE_API_KEY") or (st.secrets["GOOGLE_API_KEY"] if "GOOGLE_API_KEY" in st.secrets else None)

# Use Google Embeddings (Fastest for Cloud)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-004", google_api_key=api_key)

def get_llm(model_name, temp, user_key=None):
    """Universal AI Selector Engine"""
    if "gpt" in model_name.lower() or "o1" in model_name.lower():
        key = user_key if user_key else os.getenv("OPENAI_API_KEY")
        if not key: raise ValueError("OpenAI Key required.")
        return ChatOpenAI(model=model_name, temperature=temp, api_key=key)
    
    elif "claude" in model_name.lower():
        key = user_key if user_key else os.getenv("ANTHROPIC_API_KEY")
        if not key: raise ValueError("Anthropic Key required.")
        return ChatAnthropic(model=model_name, temperature=temp, api_key=key)
    
    else:
        key = user_key if user_key else api_key
        if not key: raise ValueError("Gemini Key required.")
        return ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=temp, 
            google_api_key=key,
            max_output_tokens=4000,
            safety_settings={"HARM_CATEGORY_HARASSMENT": "BLOCK_NONE", "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE"}
        )

def build_knowledge_base(user_key=None): # Add user_key here
    docs = []
    kb_path = "/tmp/knowledge_base"
    
    # 1. Initialize embeddings with the correct key
    current_key = user_key if user_key else api_key
    current_embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004", 
        google_api_key=current_key
    )
    
    if not os.path.exists(kb_path): return None
    for file in os.listdir(kb_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(kb_path, file))
            docs.extend(loader.load())
    
    if not docs: return None
    
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    
    # 2. Use the local embeddings object
    return FAISS.from_documents(documents=splits, embedding=current_embeddings)
    
def extract_emails(text):
    """Finds client email address from text - THIS WAS THE MISSING FUNCTION"""
    email_pattern = r"[\w\.-]+@[\w\.-]+\.\w+"
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else "client@example.com"

def get_client_name(text, model_name, user_key=None):
    # Using 2.0 Flash as the default searcher to avoid 1.5
    llm = get_llm("gemini-2.0-flash", 0, user_key)
    prompt = f"Identify the recipient company name from this RFP. Return ONLY the name. \n\n Text: {text[:2000]}"
    try:
        name = llm.invoke(prompt).content.strip()
        return name if len(name.split()) <= 5 else "Prospective Client"
    except Exception: return "Prospective Client"

def research_competitors(rfp_text):
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(f"Market trends for {rfp_text[:50]}", max_results=2)]
            return "\n".join([f"{r['title']}: {r['body']}" for r in results])
    except: return "Market data unavailable."

def generate_proposal(rfp_text, vectorstore, web_data, model_name, temp, client_name, user_key=None):
    llm = get_llm(model_name, temp, user_key)
    system_prompt = (f"Autonomous Sales AI. Draft B2B proposal for {client_name}. STRICT: NEVER use 'Your Organization'. ALWAYS use {client_name}. Format with Markdown. \n\nContext: {{context}} \nWeb Data: {{web_data}}")
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "Draft for:\n{input}")])
    chain = ({"context": vectorstore.as_retriever() | (lambda docs: "\n\n".join(d.page_content for d in docs)), "web_data": lambda x: web_data, "input": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    return chain.invoke(rfp_text)

def generate_email_body(proposal_text, model_name, client_name, user_key=None):
    llm = get_llm("gemini-2.0-flash", 0.5, user_key)
    prompt = f"Write a 3-sentence professional email to {client_name} about the attached proposal. \n\n Proposal: {proposal_text[:500]}"
    try: return str(llm.invoke(prompt).content)
    except: return f"Dear {client_name}, please find our proposal attached."

def send_real_email(recipient_email, subject, body, attachment_path, sender_email, app_password):
    try:
        clean_body = unicodedata.normalize('NFKD', str(body)).encode('ascii', 'ignore').decode('ascii')
        clean_subj = unicodedata.normalize('NFKD', str(subject)).encode('ascii', 'ignore').decode('ascii')
        msg = MIMEMultipart()
        msg['From'], msg['To'], msg['Subject'] = sender_email, recipient_email, Header(clean_subj, 'utf-8')
        msg.attach(MIMEText(clean_body, 'plain', 'utf-8'))
        with open(attachment_path, "rb") as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read()); encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(attachment_path)}"')
            msg.attach(part)
        server = smtplib.SMTP('smtp.gmail.com', 587); server.starttls()
        server.login(sender_email, app_password); server.sendmail(sender_email, recipient_email, msg.as_string()); server.quit()
        return True
    except Exception as e: return str(e)

def extract_rfp_text(path):
    return "\n".join([d.page_content for d in PyPDFLoader(path).load()])
