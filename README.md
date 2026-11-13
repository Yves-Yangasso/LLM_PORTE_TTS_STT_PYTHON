# LLM Porte — Recherche d'images, STT, TTS et notification email

Projet pédagogique en Python (notebooks Jupyter) qui combine :

- **Base vectorielle d'images** avec [ChromaDB](https://www.trychroma.com/) et des embeddings **OpenCLIP** : on indexe des photos de portes, puis on les retrouve par une requête texte (ex. « ouverte »).
- **Speech-to-Text (STT)** avec `SpeechRecognition` (API Google, `fr-FR`) : la requête de recherche peut être dictée au micro.
- **Notification par email** via SMTP (Gmail) : envoi d'un message « Une personne vient de sonner à la porte ».
- **Text-to-Speech (TTS)** avec `gTTS` : lecture à voix haute d'un texte saisi.

## Structure

```
.
├── tp_sendEmailPorte.ipynb   # TP principal : portes + STT + email + TTS
├── exo2_Session3.ipynb       # TD Session 3 : base vectorielle sur des photos de personnes
├── images/                   # Images de portes (.jpg) indexées dans ChromaDB
├── db_storePortes/           # Base ChromaDB persistante (collection "portes_collection")
├── dependencies/
│   └── requirements.txt      # Dépendances Python
└── .env                      # Paramètres SMTP (non versionnés en principe, voir plus bas)
```

## Prérequis

- Python 3.10+
- Jupyter (ou VS Code avec l'extension Jupyter)
- Un microphone pour la partie STT
- **PortAudio** pour `pyaudio` :
  - macOS : `brew install portaudio`
  - Linux : `sudo apt install portaudio19-dev`
- Une connexion Internet (téléchargement du modèle OpenCLIP, API Google STT, gTTS)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
python -m pip install -r dependencies/requirements.txt
```

Ensuite, dans VS Code / Jupyter, choisissez le noyau **`.venv`** (bouton « Select Kernel » en haut à droite du notebook).

> **`zsh: command not found: pip`** : sur macOS (Python Homebrew), la commande `pip` n'existe pas hors d'un environnement virtuel, seulement `pip3`. Utilisez `python3 -m pip ...` dans le terminal et `%pip install ...` dans les notebooks.

### La cellule `!pip install` des notebooks

La première cellule de chaque notebook est :

```python
!pip install pillow chromadb open_clip_torch tqdm ipykernel
```

Le `!` demande à Jupyter d'exécuter la ligne comme une commande shell plutôt que comme du Python. Elle installe les paquets nécessaires à la partie recherche d'images :

| Paquet | Rôle dans le projet |
|---|---|
| `pillow` | Ouvrir et afficher les images (`from PIL import Image`) |
| `chromadb` | Base vectorielle qui stocke les embeddings des images et répond aux requêtes |
| `open_clip_torch` | Modèle CLIP utilisé par `OpenCLIPEmbeddingFunction` pour transformer images **et** textes en vecteurs comparables (installe aussi PyTorch) |
| `tqdm` | Barres de progression, utilisées par OpenCLIP pendant le téléchargement du modèle |
| `ipykernel` | Noyau qui permet d'exécuter le notebook avec l'environnement Python courant |

Ces paquets sont déjà dans `requirements.txt` : si vous avez fait l'installation ci-dessus, cette cellule est inutile et peut être sautée. Si vous la gardez, préférez `%pip install ...`, qui installe toujours dans le noyau actif du notebook (`!pip` peut viser un autre Python). Elle n'installe pas les paquets STT, TTS et email (`SpeechRecognition`, `pyaudio`, `gtts`…), qui viennent uniquement de `requirements.txt`.

> `pywin32` n'existe que sous Windows : retirez-le du fichier `requirements.txt` si vous êtes sur macOS ou Linux.

## Configuration

Créez un fichier `.env` à la racine :

```env
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL=votre.adresse@gmail.com
PASSWORD=votre_mot_de_passe_application
```

Pour Gmail, utilisez un [mot de passe d'application](https://myaccount.google.com/apppasswords) (nécessite la validation en deux étapes), pas votre mot de passe habituel.

## Utilisation

Ouvrez `tp_sendEmailPorte.ipynb` et exécutez les cellules dans l'ordre :

1. **Chargement des images** depuis `images/`.
2. **Création de la collection** ChromaDB `portes_collection` (persistée dans `db_storePortes/`).
3. **Indexation** des images (`collection.add(ids=..., uris=...)`).
4. **Recherche par texte** :
   ```python
   collection.query(query_texts=["ouverte"], n_results=1, include=["uris", "distances"])
   ```
5. **Recherche vocale** : parlez pendant ~5 s, le texte reconnu sert de requête et l'image la plus proche s'affiche.
6. **Envoi d'un email** de notification via SMTP.
7. **TTS** : saisissez un texte, il est converti en MP3 et lu (tapez `quit` pour arrêter).

`exo2_Session3.ipynb` suit le même principe sur un dossier `images/personnes/*.jpeg` (non inclus dans le dépôt) avec la collection `personnes`.

## Remarques

- La lecture audio du TTS utilise `os.system('start ...')`, qui ne fonctionne que sous **Windows**. Sous macOS, remplacez par `afplay`, sous Linux par `xdg-open` ou `mpg123`.
- Relancer la cellule d'indexation sur une collection existante réutilise les mêmes identifiants : ChromaDB signalera des doublons.
- Le premier lancement télécharge les poids OpenCLIP, ce qui peut prendre quelques minutes.

## Technologies

ChromaDB · OpenCLIP · Pillow · SpeechRecognition · PyAudio · gTTS · smtplib · python-dotenv
