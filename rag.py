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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from duckduckgo_search import DDGS 

load_dotenv()

# Cloud Secret Management
api_key = os.getenv("GOOGLE_API_KEY") or (st.secrets["GOOGLE_API_KEY"] if "GOOGLE_API_KEY" in st.secrets else None)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

def get_llm(model_name, temp, user_key=None):
    if "gpt" in model_name.lower():
        key = user_key if user_key else os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(model=model_name, temperature=temp, api_key=key)
    elif "claude" in model_name.lower():
        key = user_key if user_key else os.getenv("ANTHROPIC_API_KEY")
        return ChatAnthropic(model=model_name, temperature=temp, api_key=key)
    else:
        key = user_key if user_key else api_key
        return ChatGoogleGenerativeAI(model=model_name, temperature=temp, google_api_key=key, max_output_tokens=4000)

def build_knowledge_base():
    docs = []
    kb_path = "/tmp/knowledge_base"
    if not os.path.exists(kb_path): return None
    for file in os.listdir(kb_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(kb_path, file))
            docs.extend(loader.load())
    if not docs: return None
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    return FAISS.from_documents(documents=splits, embedding=embeddings)

def get_client_name(text, model_name, user_key=None):
    llm = get_llm(model_name, 0, user_key)
    prompt = f"Find the recipient company name in this text. Return ONLY the name. Rules: No notes. Fallback to 'Prospective Client'. \n\n Text: {text[:2000]}"
    try:
        name = llm.invoke(prompt).content.strip()
        return name if len(name.split()) <= 5 else "Prospective Client"
    except: return "Prospective Client"

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
    llm = get_llm(model_name, 0.5, user_key)
    prompt = f"Write a short 3-sentence email to {client_name} about the attached proposal. No placeholders. \n\n Proposal: {proposal_text[:500]}"
    try: return str(llm.invoke(prompt).content)
    except: return "Please find our proposal attached."

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
