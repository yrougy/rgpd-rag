import chromadb
from sentence_transformers import SentenceTransformer
import os
import random

# Forcer l'utilisation du CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def rechercher(query, collection_name='rgpd_bge', model_name='BAAI/bge-m3', n_results=3):
    """
    Recherche dans le RGPD indexé
    """
    print(f"Question: {query}")
    print(f"{'='*60}\n")
    
    # 1. Charger le modèle d'embedding
    print("Chargement du modèle...")
    model = SentenceTransformer(model_name, device='cpu')
    
    # 2. Connecter à ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name=collection_name)
    
    # 3. Générer l'embedding de la question
    print("Génération de l'embedding de la question...")
    query_embedding = model.encode([query])[0]
    
    # 4. Rechercher dans ChromaDB
    print(f"Recherche des {n_results} chunks les plus pertinents...\n")
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )
    
    # 5. Afficher les résultats
    print(f"{'='*60}")
    print(f"Top {n_results} résultats:")
    print(f"{'='*60}\n")
    
    for i, (doc, meta, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        # Avec cosinus, distance = 1 - similarité, donc similarité = 1 - distance
        score = 1 - distance
        print(f"{i+1}. {meta['titre']}")
        print(f"   Score de similarité cosinus: {score:.4f}")
        print(f"   Type: {meta['type']} | Numéro: {meta['numero']}")
        print(f"   Extrait: {doc[:250]}...")
        print()
    
    # 6. Comparaison avec un article random
    print(f"{'='*60}")
    print("Comparaison avec un article aléatoire:")
    print(f"{'='*60}\n")
    
    # Récupérer tous les IDs
    all_items = collection.get(include=['metadatas'])
    random_id = random.choice(all_items['ids'])
    
    # Récupérer l'article random avec son embedding
    random_item = collection.get(
        ids=[random_id],
        include=['documents', 'metadatas', 'embeddings']
    )
    
    # Calculer la similarité avec la question (cosinus)
    import numpy as np
    from numpy.linalg import norm
    
    random_embedding = np.array(random_item['embeddings'][0])
    query_emb_np = np.array(query_embedding)
    
    # Similarité cosinus
    cosine_similarity = np.dot(query_emb_np, random_embedding) / (norm(query_emb_np) * norm(random_embedding))
    
    print(f"Article aléatoire: {random_item['metadatas'][0]['titre']}")
    print(f"Score de similarité cosinus: {cosine_similarity:.4f}")
    print(f"Extrait: {random_item['documents'][0][:250]}...")
    
    best_score = 1 - results['distances'][0][0]
    print(f"\n→ Différence avec le meilleur résultat: {best_score - cosine_similarity:.4f}")

if __name__ == "__main__":
    # Exemples de questions
    questions = [
        "Quels sont mes droits d'accès à mes données personnelles ?",
        "Comment supprimer mes données ?",
        "Qu'est-ce qu'un délégué à la protection des données ?"
    ]
    
    # Tester la première question
    rechercher(questions[0])
    
    # Pour tester les autres, décommentez :
    # rechercher(questions[1])
    # rechercher(questions[2])
