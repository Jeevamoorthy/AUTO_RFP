import os
import re
import smtplib
import unicodedata
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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from duckduckgo_search import DDGS 

load_dotenv()

# Global Tool Initialization
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def web_search_bypass(query):
    """Bypasses the broken LangChain DuckDuckGo tool."""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            return "\n".join([f"{r['title']}: {r['body']}" for r in results])
    except Exception:
        return "Market data currently unavailable via live search."

def build_knowledge_base():
    docs = []
    kb_path = "data/knowledge_base"
    os.makedirs(kb_path, exist_ok=True)
    for file in os.listdir(kb_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(kb_path, file))
            docs.extend(loader.load())
    if not docs:
        return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return FAISS.from_documents(documents=splits, embedding=embeddings)

def extract_emails(text):
    email_pattern = r"[\w\.-]+@[\w\.-]+\.\w+"
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else "client@example.com"

def get_client_name(text, model_name):
    """Identifies client name using stable model to save quota."""
    # Using 1.5-flash-latest here because it has a huge free quota
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    prompt = (
        "Identify the recipient company name from this RFP. Return ONLY the name. "
        "Rules: No notes. If missing, return 'Prospective Client'. \n\n"
        f"Text Sample: {text[:2000]}"
    )
    try:
        name = llm.invoke(prompt).content.strip()
        if len(name.split()) > 5 or "explicitly" in name.lower():
            return "Prospective Client"
        return name
    except Exception:
        return "Prospective Client"

def research_competitors(rfp_text):
    query = f"Market pricing and trends for: {rfp_text[:100]}"
    return web_search_bypass(query)

def generate_proposal(rfp_text, vectorstore, web_data, model_name, temp, client_name):
    """Generates the main proposal using the UI-selected model."""
    # Using the exact model_name passed from the Streamlit UI
    llm = ChatGoogleGenerativeAI(
        model=model_name, 
        temperature=temp, 
        max_output_tokens=4000,
        safety_settings={
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }
    )
    retriever = vectorstore.as_retriever()
    
    system_prompt = (
        "You are an Autonomous Sales AI. Draft a professional B2B proposal. "
        f"THE CLIENT IS: {client_name}. "
        f"STRICT: NEVER use 'Your Organization'. ALWAYS use {client_name}. "
        "Format with Markdown (# Titles, ** Bold)."
        "\n\nContext: {context} \nWeb Data: {web_data}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Draft proposal for:\n{input}")
    ])
    
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "web_data": lambda x: web_data, "input": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain.invoke(rfp_text)

def generate_email_body(proposal_text, model_name, client_name):
    """Generates email text using stable model to save quota."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5)
    prompt = f"Write a 3-sentence professional email to {client_name} about the attached proposal. \n\n Proposal: {proposal_text[:500]}"
    try:
        return str(llm.invoke(prompt).content)
    except Exception:
        return f"Dear {client_name}, please find our proposal attached."

def send_real_email(recipient_email, subject, body, attachment_path, sender_email, app_password):
    try:
        # Nuclear option sanitization for ASCII errors
        clean_body = unicodedata.normalize('NFKD', str(body)).encode('ascii', 'ignore').decode('ascii')
        clean_subject = unicodedata.normalize('NFKD', str(subject)).encode('ascii', 'ignore').decode('ascii')

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = Header(clean_subject, 'utf-8')
        msg.attach(MIMEText(clean_body, 'plain', 'utf-8'))

        if os.path.exists(attachment_path):
            filename = os.path.basename(attachment_path)
            with open(attachment_path, "rb") as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename="{filename}"')
                msg.attach(part)
        else:
            return f"Error: Attachment not found at {attachment_path}"

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        return str(e)

def extract_rfp_text(path):
    loader = PyPDFLoader(path)
    return "\n".join([d.page_content for d in loader.load()])
