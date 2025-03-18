
from langchain.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Clé API et modèles
API_KEY = "AIzaSyDAoRDqZJR8EJ3bZe5zoA62mqK2q7a8oMA"
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-1.5-flash"


def preparer_base(dir_path=None, save_path="faiss_index"):
    """Charge les documents, génère les embeddings et sauvegarde la base FAISS."""

    # 1. Chargement des documents
    documents = []

    if dir_path:
        dLoader = DirectoryLoader(dir_path, glob="*.txt", show_progress=True) 
        documents += dLoader.load()

    if not documents:
        raise ValueError("Aucun document trouvé pour créer la base.")

    # 2. Découpage en chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # 3. Conversion en embeddings
    embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=API_KEY)
    vector_store = FAISS.from_documents(chunks, embedding_model)

    # 4. Sauvegarde de FAISS
    vector_store.save_local(save_path)
    print(f" Base FAISS sauvegardée dans {save_path}")

# utilisation
dir_path = "corpus_txt"

preparer_base(dir_path)
