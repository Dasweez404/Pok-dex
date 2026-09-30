# Pokédex caméra

Appli web statique : pointez la caméra vers un Pokémon (peluche, carte, écran, dessin) → détection,
artwork officiel, nom français et **toutes les entrées Pokédex (toutes versions) fusionnées en un paragraphe**
(doublons supprimés, français d'abord, anglais en secours).

- Détection : CLIP (transformers.js) exécuté **dans le navigateur**, aucune clé d'API. Le modèle (~100 Mo) et les
  embeddings des 1025 noms sont mis en cache (IndexedDB) après le premier lancement.
- Données : [PokéAPI](https://pokeapi.co) (noms FR, entrées, types) et dépôt `PokeAPI/sprites` (artworks officiels).
- Lancer : `python3 -m http.server 8000` puis ouvrir http://localhost:8000 (la caméra exige HTTPS ou localhost).
  Sur téléphone, publiez le dossier (GitHub Pages, Netlify…) en HTTPS.
- La précision est bonne sur les Pokémon connus, moindre sur les cas ambigus : les boutons « Ce n'est pas lui ? »
  proposent les 3 meilleures alternatives, et la barre de recherche permet de chercher par nom/numéro.
