import chromadb
from sentence_transformers import SentenceTransformer
import os
import requests
import json

# Configuration
OLLAMA_MODEL = 'gpt-oss:20b'  # Modèles : phi3:mini, mistral:7b, llama3.2:3b
OLLAMA_BASE_URL = 'http://localhost:11434'
EMBEDDING_MODEL = 'BAAI/bge-m3'
COLLECTION_NAME = 'rgpd_bge'

# Forcer l'utilisation du CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def interroger_llm(prompt, model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL):
    """
    Interroge un LLM local via Ollama
    """
    response = requests.post(
        f'{base_url}/api/generate',
        json={
            'model': model,
            'prompt': prompt,
            'stream': False
        }
    )
    return response.json()['response']

def rag_complet(question, collection_name=COLLECTION_NAME, embedding_model=EMBEDDING_MODEL, 
                llm_model=OLLAMA_MODEL, n_results=3):
    """
    Pipeline RAG complet: recherche vectorielle + génération de réponse
    """
    print(f"Question: {question}")
    print(f"{'='*60}\n")
    
    # 1. Recherche vectorielle
    print("1. Recherche des chunks pertinents...")
    model = SentenceTransformer(embedding_model, device='cpu')
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name=collection_name)
    
    query_embedding = model.encode([question])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )
    
    print(f"   ✓ {n_results} chunks trouvés\n")
    
    # Afficher les chunks trouvés
    print(f"{'='*60}")
    print("Chunks pertinents:")
    print(f"{'='*60}\n")
    for i, (doc, meta, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        score = 1 - distance
        print(f"{i+1}. {meta['titre']} (similarité: {score:.4f})")
        print(f"   {doc[:200]}...\n")
    
    # 2. Construire le prompt avec contexte
    print(f"{'='*60}")
    print("2. Construction du prompt avec contexte...")
    print(f"{'='*60}\n")
    
    contexte = "\n\n".join([
        f"[{meta['titre']}]\n{doc}" 
        for doc, meta in zip(results['documents'][0], results['metadatas'][0])
    ])
    
    prompt = f"""Tu es un assistant juridique expert du RGPD. Réponds à la question en te basant UNIQUEMENT sur le contexte fourni ci-dessous.

CONTEXTE:
{contexte}

QUESTION: {question}

INSTRUCTIONS:
- Réponds en français de manière claire et précise
- CITE EXPLICITEMENT les numéros d'articles et de considérants utilisés dans ta réponse
- Utilise le format "selon l'Article X" ou "comme indiqué au Considérant Y"
- Structure ta réponse avec les références juridiques
- Ne mentionne AUCUNE information qui ne figure pas dans le contexte fourni

RÉPONSE:"""

    print(f"   Longueur du prompt: {len(prompt)} caractères")
    print(f"   Modèle LLM: {llm_model}\n")
    
    # 3. Génération de la réponse par le LLM
    print(f"{'='*60}")
    print("3. Génération de la réponse par le LLM...")
    print(f"{'='*60}\n")
    
    try:
        reponse = interroger_llm(prompt, model=llm_model)
        
        print("RÉPONSE DU LLM:")
        print(f"{'-'*60}")
        print(reponse)
        print(f"{'-'*60}\n")
        
        return reponse
    
    except Exception as e:
        print(f"⚠ Erreur lors de l'appel au LLM: {e}")
        print("\nAssurez-vous qu'Ollama est lancé avec:")
        print(f"  ollama run {llm_model}")
        return None

if __name__ == "__main__":
    print("="*60)
    print("RAG INTERACTIF - RGPD")
    print("="*60)
    print("\nPosez votre question sur le RGPD (ou 'quit' pour quitter)\n")
    
    while True:
        question = input("❓ Question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nAu revoir !")
            break
        
        if not question:
            print("⚠ Veuillez poser une question.\n")
            continue
        
        rag_complet(question)
        
        print("\n" + "="*60 + "\n")
