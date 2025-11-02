import re
import json
from bs4 import BeautifulSoup

def decouper_rgpd(fichier_html='rgpd/rgpd.html'):
    """Découpe le RGPD en chunks par considérants et articles"""
    with open(fichier_html, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    
    chunks = []
    
    # Extraire tout le texte
    texte_complet = soup.get_text(separator=' ')
    texte_complet = re.sub(r'\s+', ' ', texte_complet)
    
    print("Extraction des considérants...")
    # Pattern pour considérants: (X) texte jusqu'au prochain (X+1) ou Article
    pattern_considerants = r'\((\d+)\)\s+(.*?)(?=\s*\(\d+\)|\s+CHAPITRE\s+I|$)'
    matches = list(re.finditer(pattern_considerants, texte_complet, re.DOTALL))
    
    for match in matches:
        numero = int(match.group(1))
        contenu = match.group(2).strip()
        
        if len(contenu) > 50:
            chunks.append({
                'type': 'considérant',
                'numero': numero,
                'titre': f"Considérant {numero}",
                'contenu': contenu,
                'id': f"considerant_{numero}"
            })
    
    print(f"✓ {len(chunks)} considérants extraits")
    
    # Extraction des articles
    # Les vrais articles sont dans "CHAPITRE I" et suivants
    print("\nExtraction des articles...")
    
    # Trouver où commence CHAPITRE I (après les considérants)
    match_chapitre = re.search(r'CHAPITRE\s+I', texte_complet)
    if match_chapitre:
        texte_articles = texte_complet[match_chapitre.start():]
        print(f"  Début des articles trouvé à position {match_chapitre.start()}")
    else:
        # Fallback: chercher après le dernier considérant
        dernier_considerant = max([c['numero'] for c in chunks if c['type'] == 'considérant'])
        pattern_dernier = rf'\({dernier_considerant}\).*?(?=Article\s+1\s+)'
        match_dernier = re.search(pattern_dernier, texte_complet, re.DOTALL)
        if match_dernier:
            texte_articles = texte_complet[match_dernier.end():]
        else:
            texte_articles = texte_complet
    
    # Extraire l'article 1 spécifiquement (souvent formaté différemment)
    # Essayer différentes variantes
    patterns_art1 = [
        r'Article\s+premier\s+(.*?)(?=\s+Article\s+\d+|$)',
        r'ARTICLE\s+PREMIER\s+(.*?)(?=\s+Article\s+\d+|$)',
        r'Article\s+1er\s+(.*?)(?=\s+Article\s+\d+|$)',
        r'Article\s+1\s+(.*?)(?=\s+Article\s+\d+|$)',
    ]
    
    match_art1 = None
    for pattern in patterns_art1:
        match_art1 = re.search(pattern, texte_articles, re.DOTALL | re.IGNORECASE)
        if match_art1:
            print(f"  ✓ Article 1 trouvé avec pattern: {pattern[:30]}...")
            break
    
    if not match_art1:
        # Debug: afficher ce qui vient après CHAPITRE I
        print(f"  ⚠ Article 1 non trouvé. Début du texte après CHAPITRE I:")
        print(f"    {texte_articles[:500]}")
    
    # Maintenant extraire les articles depuis cette position
    pattern_articles = r'Article\s+(\d+)\s+(.*?)(?=\s+Article\s+\d+|\s+CHAPITRE\s+[IVX]+|$)'
    matches_articles = list(re.finditer(pattern_articles, texte_articles, re.DOTALL | re.IGNORECASE))
    
    print(f"  Trouvé {len(matches_articles)} articles")
    
    articles_dict = {}  # Pour dédupliquer
    
    # Ajouter l'article 1 si trouvé
    if match_art1:
        contenu = match_art1.group(1).strip()
        if len(contenu) > 50:
            articles_dict[1] = {
                'type': 'article',
                'numero': 1,
                'titre': 'Article 1',
                'contenu': contenu,
                'id': 'article_1'
            }
            print(f"  ✓ Article 1 extrait spécifiquement")
    
    for match in matches_articles:
        numero = int(match.group(1))
        contenu = match.group(2).strip()
        
        if len(contenu) > 50:
            # Garder le plus long en cas de doublon
            if numero not in articles_dict or len(contenu) > len(articles_dict[numero]['contenu']):
                articles_dict[numero] = {
                    'type': 'article',
                    'numero': numero,
                    'titre': f"Article {numero}",
                    'contenu': contenu,
                    'id': f"article_{numero}"
                }
    
    chunks.extend(articles_dict.values())
    print(f"✓ {len(articles_dict)} articles uniques extraits")
    
    # Tri
    chunks.sort(key=lambda x: (0 if x['type'] == 'considérant' else 1, x['numero']))
    
    # Vérifications
    articles = [c for c in chunks if c['type'] == 'article']
    numeros = sorted([a['numero'] for a in articles])
    
    print(f"\n✓ Total: {len(chunks)} chunks créés")
    print(f"  - Considérants: {len([c for c in chunks if c['type'] == 'considérant'])}")
    print(f"  - Articles: {len(articles)} (numéros {min(numeros)} à {max(numeros)})")
    
    # Vérifier les articles manquants
    manquants = [i for i in range(1, 100) if i not in numeros]
    if manquants:
        print(f"\n⚠ Articles manquants: {manquants}")
    
    # Sauvegarde
    with open('rgpd_chunks.json', 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    return chunks

if __name__ == "__main__":
    chunks = decouper_rgpd()
