def build_knowledge_base(user_key=None):
    """Resilient Knowledge Base Builder"""
    docs = []
    kb_path = "/tmp/knowledge_base"
    
    current_key = user_key if user_key else api_key
    if not current_key:
        return None # Don't even try without a key

    # --- THE MODEL HUNTER LOOP ---
    # We try different naming conventions to find what Google likes in your region
    embeddings_model = None
    for m_name in ["models/text-embedding-004", "text-embedding-004", "models/embedding-001"]:
        try:
            test_emb = GoogleGenerativeAIEmbeddings(model=m_name, google_api_key=current_key)
            test_emb.embed_query("test") # Test if this name works
            embeddings_model = test_emb
            break # If success, stop looking
        except:
            continue
            
    if not embeddings_model:
        raise ValueError("Google API could not find a valid embedding model. Check your API Key.")

    if not os.path.exists(kb_path): return None
    for file in os.listdir(kb_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(kb_path, file))
            docs.extend(loader.load())
    
    if not docs: return None
    
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    return FAISS.from_documents(documents=splits, embedding=embeddings_model)
