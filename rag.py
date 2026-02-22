import os
import re
import smtplib
import unicodedata
from email import encoders
from email.header import Header
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import unicodedata

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Global Tool Initialization
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
search_tool = DuckDuckGoSearchRun()


def build_knowledge_base():
    """Reads company documents and builds a local FAISS database."""
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
    """Finds client email address from text."""
    email_pattern = r"[\w\.-]+@[\w\.-]+\.\w+"
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else "client@example.com"


def get_client_name(text, model_name):
    """Identifies the recipient company name with safety filters."""
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    prompt = (
        "Identify the recipient company name from this RFP. Return ONLY the name. "
        "Rules: No notes, no explanations. If missing, return 'Prospective Client'. \n\n"
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
    """Agentic web search for market context."""
    try:
        return search_tool.run(f"Market pricing for: {rfp_text[:100]}")
    except Exception:
        return "Market data currently unavailable."


def generate_proposal(
    rfp_text,
    vectorstore,
    web_data,
    model_name,
    temp,
    client_name,
    provider="gemini"
):
    """
    Generates a professional B2B proposal using selected LLM provider.
    provider options:
    - gemini
    - openai
    - claude
    """

    # --- Dynamic LLM Selection ---
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=temp
        )

    elif provider == "claude":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model="claude-3-sonnet-20240229",
            temperature=temp
        )

    else:  # Default Gemini
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temp,
            max_output_tokens=6000
        )

    retriever = vectorstore.as_retriever()

    system_prompt = (
        "You are an Autonomous Sales AI. Draft a professional B2B proposal. "
        f"THE CLIENT IS: {client_name}. "
        f"STRICT: NEVER use 'Your Organization'. ALWAYS use {client_name}. "
        "Format with Markdown (# Titles, ** Bold)."
        "\n\nContext: {context}\nWeb Data: {web_data}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Draft proposal for:\n{input}")
    ])

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "web_data": lambda x: web_data,
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(rfp_text)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Draft proposal for:\n{input}")
    ])
    
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "web_data": lambda x: web_data, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(rfp_text)


# --- Update 1: Force the email body to be a string ---
def generate_email_body(proposal_text, model_name, client_name, provider="gemini"):
    """
    Generates a short professional email body using selected LLM.
    """

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.5
        )

    elif provider == "claude":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model="claude-3-sonnet-20240229",
            temperature=0.5
        )

    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.5
        )

    prompt = (
        f"Write a 3-sentence professional email to {client_name} "
        f"about the attached proposal.\n\n"
        f"Proposal Summary:\n{proposal_text[:800]}"
    )

    try:
        response = llm.invoke(prompt)
        content = response.content
        if isinstance(content, list):
            return " ".join(str(x) for x in content)
        return str(content)
    except Exception:
        return f"Hello {client_name}, please find our proposal attached for your review."
def sanitize_text(text):
    """The 'Nuclear Option': Strips every single non-ASCII character from the text."""
    if not text:
        return ""
    # Convert to string just in case it's a list/object
    text = str(text)
    # Decompose special characters (like ligatures) into separate letters
    normalized = unicodedata.normalize('NFKD', text)
    # Keep ONLY standard characters (ASCII 0-127) and ignore everything else
    return "".join(c for c in normalized if ord(c) < 128)

def send_real_email(recipient_email, subject, body, attachment_path, sender_email, app_password):
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = Header(str(subject), 'utf-8')

        msg.attach(MIMEText(str(body), 'plain', 'utf-8'))

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
        server.sendmail(
        sender_email,
        recipient_email,
        msg.as_bytes()
        )
        server.quit()

        return True

    except Exception as e:
        return f"Technical Dispatch Error: {str(e)}"

def extract_rfp_text(path):
    """Extracts text from PDF."""
    loader = PyPDFLoader(path)
    return "\n".join([d.page_content for d in loader.load()])
