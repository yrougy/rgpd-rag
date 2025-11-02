import json
import chromadb
from sentence_transformers import SentenceTransformer

def indexer_rgpd(model_name='BAAI/bge-m3', collection_name='rgpd_bge'):
    """
    Indexe les chunks du RGPD dans ChromaDB avec BGE-M3
    """
    # 1. Charger les chunks
    print("1. Chargement des chunks...")
    with open('rgpd_chunks.json', 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"   ✓ {len(chunks)} chunks chargés")
    
    # 2. Charger le modèle d'embedding
    print(f"\n2. Chargement du modèle {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"   ✓ Modèle chargé")
    
    # 3. Initialiser ChromaDB
    print("\n3. Initialisation de ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Supprimer la collection si elle existe
    try:
        client.delete_collection(name=collection_name)
    except:
        pass
    
    # Créer la collection avec similarité cosinus
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # Utiliser la similarité cosinus
    )
    print(f"   ✓ Collection '{collection_name}' créée")
    
    # 4. Indexer les chunks
    print("\n4. Génération des embeddings et indexation...")
    documents = []
    metadatas = []
    ids = []
    
    for chunk in chunks:
        # Texte à embedder
        text = f"{chunk['titre']}\n\n{chunk['contenu']}"
        documents.append(text)
        
        # Métadonnées
        metadatas.append({
            'type': chunk['type'],
            'numero': chunk['numero'],
            'titre': chunk['titre']
        })
        
        ids.append(chunk['id'])
    
    # Générer les embeddings
    print("   Génération des embeddings...")
    embeddings = model.encode(documents, show_progress_bar=True)
    
    # Ajouter à ChromaDB
    print("   Ajout à ChromaDB...")
    collection.add(
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"\n✓ Indexation terminée!")
    print(f"✓ {len(documents)} chunks indexés dans '{collection_name}'")
    
    return collection

if __name__ == "__main__":
    collection = indexer_rgpd()
