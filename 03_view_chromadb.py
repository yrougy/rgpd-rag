import chromadb

def visualiser_chromadb(collection_name='rgpd_bge'):
    """Affiche le contenu de la collection ChromaDB"""
    
    # Connexion
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Lister les collections
    print("Collections disponibles:")
    collections = client.list_collections()
    for col in collections:
        print(f"  - {col.name}")
    
    if not collections:
        print("  Aucune collection trouvée!")
        return
    
    print(f"\n{'='*60}")
    print(f"Collection: {collection_name}")
    print(f"{'='*60}")
    
    # Charger la collection
    try:
        collection = client.get_collection(name=collection_name)
    except:
        print(f"Collection '{collection_name}' non trouvée!")
        return
    
    # Stats
    count = collection.count()
    print(f"\nNombre total de chunks: {count}")
    
    # Récupérer 2 exemples avec leurs embeddings
    results = collection.get(limit=2, include=['documents', 'metadatas', 'embeddings'])
    
    print(f"\n{'='*60}")
    print("Exemples de chunks:")
    print(f"{'='*60}")
    
    for i, (doc, meta, embedding) in enumerate(zip(results['documents'], results['metadatas'], results['embeddings'])):
        print(f"\n{i+1}. {meta['titre']}")
        print(f"   Type: {meta['type']} | Numéro: {meta['numero']}")
        print(f"   Contenu (150 premiers car.): {doc[:150]}...")
        print(f"\n   Vecteur d'embedding:")
        print(f"   - Dimension: {len(embedding)}")
        print(f"   - Premières valeurs: {embedding[:10]}")
        print(f"   - Dernières valeurs: {embedding[-10:]}")
        print(f"   - Min: {min(embedding):.4f}, Max: {max(embedding):.4f}")
    
    # Stats par type
    all_results = collection.get(include=['metadatas'])
    types = {}
    for meta in all_results['metadatas']:
        t = meta['type']
        types[t] = types.get(t, 0) + 1
    
    print(f"\n{'='*60}")
    print("Répartition par type:")
    for t, count in types.items():
        print(f"  {t}: {count}")

if __name__ == "__main__":
    visualiser_chromadb()
