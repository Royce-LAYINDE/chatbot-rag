from fastapi import FastAPI
from pydantic import BaseModel
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage

# Configuration du modèle et de l'indexation
API_KEY = "AIzaSyDAoRDqZJR8EJ3bZe5zoA62mqK2q7a8oMA"
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-1.5-flash"
FAISS_PATH = "faiss_index"

# Charger FAISS
embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=API_KEY)
vector_store = FAISS.load_local(FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Création de l’API avec FastAPI
app = FastAPI()




# Modèle pour la requête utilisateur
class UserMessage(BaseModel): 
    question: str

@app.get("/")
def bienvenue():
    return {
        "message": "Bonjour et bienvenue au Dakar Institute of Technology (DIT) ! "
                   "Je suis Nico, votre assistant virtuel. En quoi puis-je vous aider aujourd'hui ?"
    }

@app.post("/chat")
def chatbot_interaction(user_message: UserMessage):

    query = user_message.question

    # Récupérer les documents pertinents
    retrieved_docs = vector_store.similarity_search(query, k=3)
    if not retrieved_docs:
        return {"reponse": "Je suis désolé, je ne dispose pas d’informations pertinentes sur ce sujet. "
        "N'hésiter pas à nous contacter par mail info@dit.sn ."}


    # Sélectionner le meilleur document
    best_doc = retrieved_docs[0]  
    document_source = best_doc.metadata.get("source", "Document inconnu")


    # contexte de réponse
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""Tu es Nico, le chatbot officiel du Dakar Institute of Technology (DIT).
    Ta mission est d’aider les visiteurs du site à trouver des informations sur l’école de manière 
    claire et professionnelle.\n**Document analysé** : {document_source}\n
    **Règles** :Si l’information est disponible, donne-la directement sans mentionner comment tu l’as 
    obtenue. Ne redirige jamais vers un site web ou une autre source externe.Si une information semble 
    incomplète, donne ce que tu sais mais reste dans ton rôle d’assistant du DIT.Utilise un ton 
    professionnel et institutionnel, comme si tu faisais partie du personnel du DIT.Si tu ne trouves 
    pas la réponse, indique simplement que tu ne disposes pas de cette information actuellement
    (pas besoin de préciser que ce n'est pas dans les documents que tu as), et invite l'utilisateur à 
    nous contacter.Réponds de manière fluide et naturelle, comme un assistant humain du DIT.\n**
    Extrait du document** :{context}\n**Question utilisateur** :{query}\n**Réponse détaillée** :"""

    # Génération de la réponse avec Gemini
    chat_model = ChatGoogleGenerativeAI(model=LLM_MODEL, api_key=API_KEY)
    messages = [
        SystemMessage(content="Tu es Nico, le chatbot du DIT. Donne des réponses claires et utiles."),
        HumanMessage(content=prompt)
    ]
    
    response = chat_model(messages)
    return {"reponse": response.content}