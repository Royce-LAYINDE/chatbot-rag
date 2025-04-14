# Chatbot RAG 

Ce projet consiste à développer un assistant virtuel intelligent pour le site du Dakar Institute of Technology (DIT), en s'appuyant sur l'approche Retrieval-Augmented Generation (RAG). L'objectif est de répondre de manière pertinente aux questions des visiteurs en exploitant une base de connaissances interne construite à partir de documents officiels du DIT.

## Objectifs

- Offrir un accès rapide et automatisé aux informations du DIT (formations, admissions, contacts, etc.)
- Améliorer l’expérience utilisateur en ligne en réduisant la charge de travail du personnel administratif
- Renforcer l’image technologique du DIT à travers un service intelligent accessible 24/7

## Fonctionnement général

Le chatbot repose sur deux modules clés :

1. **Module Retriever**  
   - Recherche de documents pertinents à partir d’une base vectorielle (FAISS) alimentée par des embeddings générés via `GoogleGenerativeAIEmbeddings`
   - Utilisation de la méthode `similarity_search_with_score()` pour extraire les 3 passages les plus pertinents

2. **Module Générateur**  
   - Génération d’une réponse naturelle à partir des documents récupérés
   - Utilisation du modèle **Gemini 1.5 Flash**
   - Structuration du prompt pour un ton professionnel et informatif

## Données et sources

Les données indexées proviennent de :
- Extraction manuelle depuis le site officiel du DIT
- Web scraping ciblé
- Documents institutionnels (plaquettes, maquettes de formation, règlements)

Les documents sont prétraités, segmentés avec `RecursiveCharacterTextSplitter` (blocs de 500 caractères avec recouvrement), puis vectorisés.

## Stack technique

| Composant        | Technologie                         |
|------------------|--------------------------------------|
| Framework API     | FastAPI                              |
| Vector DB         | FAISS                                |
| Modèle de génération | Gemini 1.5 Flash (Google)            |
| Embeddings        | GoogleGenerativeAIEmbeddings         |
| Langage           | Python 3.10+                         |
| Déploiement       | Docker (image disponible sur Docker Hub) |

## Déploiement rapide avec Docker

1. Télécharger l'image :
   ```bash
   docker pull roy61/chatbotdit-rag
   ```
2. Lancer le conteneur :

  ```bash
  docker run -d -p 8000:8000 roy61/chatbotdit-rag
  ```
3. Accéder au chatbot :

- Interface de base : http://127.0.0.1:8000
- Interface Swagger (documentation interactive) : http://127.0.0.1:8000/docs
Exemple de test
- Via navigateur :
Accéder directement à l’interface via : http://127.0.0.1:8000/docs
- Via Postman :
Envoyer une requête POST vers :

  ```bash
  http://127.0.0.1:8000/chat
  ```bash
Avec un body JSON comme :

  ```bash
  {
    "question": "Quels masters propose le DIT ?"
  }
  ```
- Via cURL :
  ```bash
  curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels masters propose le DIT ?"}'
  ```
## Optimisations apportées
- Nettoyage des réponses (suppression des phrases génériques comme "Selon les documents que j’ai analysés…")
- Formatage des listes, titres et paragraphes pour améliorer la lisibilité
- Structuration du prompt pour guider le modèle à rester factuel et institutionnel

## Auteur
[Malick Royce LAYINDE](https://roylab.xyz/)
Projet réalisé dans le cadre d'une exploration technique autour de la génération augmentée par récupération (RAG) et des grands modèles de langage appliqués à un contexte réel.

**Image Docker** : roy61/chatbotdit-rag
