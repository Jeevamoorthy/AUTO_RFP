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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from duckduckgo_search import DDGS 

load_dotenv()

# Global Tool Initialization
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_llm(model_name, temp, user_key=None):
    """Universal AI Selector Logic"""
    
    # 1. OPENAI PROVIDER
    if "gpt" in model_name.lower() or "o1" in model_name.lower():
        key = user_key if user_key else st.secrets.get("OPENAI_API_KEY")
        if not key: raise ValueError("Please enter your OpenAI API Key in the sidebar.")
        return ChatOpenAI(model=model_name, temperature=temp, api_key=key)
    
    # 2. ANTHROPIC PROVIDER
    elif "claude" in model_name.lower():
        key = user_key if user_key else st.secrets.get("ANTHROPIC_API_KEY")
        if not key: raise ValueError("Please enter your Anthropic API Key in the sidebar.")
        return ChatAnthropic(model=model_name, temperature=temp, api_key=key)
    
    # 3. GOOGLE PROVIDER (Default / Your List)
    else:
        key = user_key if user_key else (st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        if not key: raise ValueError("Please enter your Gemini API Key in the sidebar.")
        return ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=temp, 
            google_api_key=key,
            max_output_tokens=4000,
            safety_settings={"HARM_CATEGORY_HARASSMENT": "BLOCK_NONE", "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE"}
        )

# ... (Keep your web_search_bypass, build_knowledge_base, extract_emails, extract_rfp_text functions)

def get_client_name(text, model_name, user_key=None):
    llm = get_llm(model_name, 0, user_key)
    prompt = f"Identify the recipient company name from this RFP. Return ONLY the name. No notes. \n\n Text: {text[:2000]}"
    try:
        name = llm.invoke(prompt).content.strip()
        return name if len(name.split()) <= 5 else "Prospective Client"
    except Exception: return "Prospective Client"

def research_competitors(rfp_text):
    try:
        with DDGS() as ddgs:
            query = f"Market trends for {rfp_text[:50]}"
            results = [r for r in ddgs.text(query, max_results=2)]
            return "\n".join([f"{r['title']}: {r['body']}" for r in results])
    except: return "Market data unavailable."

def generate_proposal(rfp_text, vectorstore, web_data, model_name, temp, client_name, user_key=None):
    llm = get_llm(model_name, temp, user_key)
    retriever = vectorstore.as_retriever()
    
    system_prompt = (
        "You are an Autonomous Sales AI. Draft a professional B2B proposal. "
        f"THE CLIENT IS: {client_name}. NEVER use 'Your Organization'. ALWAYS use {client_name}. "
        "Format with Markdown (# Titles, ** Bold)."
        "\n\nContext: {context} \nWeb Data: {web_data}"
    )
    
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "Draft for:\n{input}")])
    def format_docs(docs): return "\n\n".join(d.page_content for d in docs)
    chain = ({"context": retriever | format_docs, "web_data": lambda x: web_data, "input": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    return chain.invoke(rfp_text)

def generate_email_body(proposal_text, model_name, client_name, user_key=None):
    llm = get_llm(model_name, 0.5, user_key)
    prompt = f"Write a 3-sentence professional email to {client_name} about the attached proposal. \n\n Proposal: {proposal_text[:500]}"
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
        server.login(sender_mail, app_password); server.sendmail(sender_email, recipient_email, msg.as_string()); server.quit()
        return True
    except Exception as e: return str(e)
