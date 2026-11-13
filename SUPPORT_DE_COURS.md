# Support de cours — Porte intelligente : recherche d'images, voix et notifications

> **TP associé :** `tp_sendEmailPorte.ipynb`
> **Durée indicative :** 3 h
> **Niveau :** bases de Python, première utilisation de Jupyter

---

## 1. Objectifs

À la fin de ce TP, vous saurez :

1. Expliquer ce qu'est un **embedding** et une **base de données vectorielle**.
2. Indexer des images dans **ChromaDB** avec le modèle **OpenCLIP**.
3. Retrouver une image à partir d'une **phrase** (recherche sémantique).
4. Transformer la **voix en texte** (Speech-to-Text, STT).
5. Envoyer un **email** automatiquement avec Python (SMTP).
6. Transformer du **texte en voix** (Text-to-Speech, TTS).

### Le scénario

On imagine une **porte connectée** :

```
 🎤 Voix  ──STT──▶  texte  ──▶  ChromaDB  ──▶  🖼️ image de porte la plus proche
                                                     │
                                                     ▼
                                    📧 email « quelqu'un sonne »  +  🔊 message vocal
```

---

## 2. Notions clés

### 2.1 Qu'est-ce qu'un embedding ?

Un **embedding** est une liste de nombres (un **vecteur**) qui représente le *sens* d'une donnée.

- Deux contenus proches en sens → deux vecteurs proches.
- Deux contenus différents → deux vecteurs éloignés.

```
"porte ouverte"   → [0.12, -0.40, 0.88, ...]   (512 nombres)
photo porte ouverte → [0.10, -0.38, 0.91, ...]   ← proche !
photo d'un chat    → [-0.70, 0.22, 0.05, ...]   ← loin
```

### 2.2 Pourquoi CLIP ?

**CLIP** (et sa version libre **OpenCLIP**) a été entraîné sur des millions de couples *image + légende*. Il place **les images et les textes dans le même espace de vecteurs**.

C'est ce qui permet de chercher une **image** avec une **phrase**, sans avoir jamais écrit de légende pour les images.

### 2.3 Qu'est-ce qu'une base vectorielle ?

Une base classique (SQL) cherche des valeurs **exactes** : `WHERE nom = 'porte'`.
Une base **vectorielle** cherche les vecteurs **les plus proches** d'un vecteur donné : « trouve-moi ce qui ressemble le plus à *porte ouverte* ».

**ChromaDB** est une base vectorielle simple, qui tourne en local et s'utilise directement depuis Python.

| Notion ChromaDB | Équivalent SQL (approximatif) |
|---|---|
| Client | Connexion à la base |
| Collection | Table |
| `id` | Clé primaire |
| `embedding` | Colonne « vecteur » |
| `uri` / `metadata` | Autres colonnes |
| `query()` | `SELECT ... ORDER BY similarité LIMIT n` |

### 2.4 Comment CLIP a appris : l'apprentissage contrastif

#### a) Le type d'apprentissage

CLIP (*Contrastive Language–Image Pre-training*) est entraîné par **apprentissage contrastif auto-supervisé** (on parle aussi de supervision *faible* par le langage naturel) :

| | Apprentissage supervisé classique | Apprentissage contrastif (CLIP) |
|---|---|---|
| Données | Images annotées à la main avec une classe (`chat`, `porte`…) | Couples **image + légende** récupérés sur le web, sans annotation manuelle |
| Ce qu'on apprend | Prédire une classe parmi une liste fixe | Rapprocher l'image de **sa** légende et l'éloigner des **autres** légendes |
| Sortie | Une probabilité par classe | Un **vecteur** (embedding) pour chaque image et chaque texte |
| Nouvelles catégories | Il faut ré-entraîner | Il suffit d'écrire un nouveau texte (**zero-shot**) |

Le modèle utilisé dans ce TP (valeurs par défaut de `OpenCLIPEmbeddingFunction`) :

- architecture **ViT-B-32** : un *Vision Transformer* pour les images + un *Transformer* pour le texte ;
- poids **`laion2b_s34b_b79k`** : entraîné sur **LAION-2B** (≈ 2 milliards de couples image/texte), **34 milliards** d'exemples vus, lots (*batchs*) de **79 000** couples ;
- sortie : un vecteur de **512 dimensions**.

> ⚠️ **Dans le TP, on n'entraîne rien.** On utilise le modèle déjà entraîné (*inférence*) : c'est du **transfert zero-shot**. Les formules ci-dessous expliquent comment ce modèle a été obtenu.

#### b) Les deux encodeurs

```
image  ──▶ [ Encodeur image  f ] ──▶ vecteur u (512) ──┐
                                                       ├──▶ similarité cosinus
texte  ──▶ [ Encodeur texte  g ] ──▶ vecteur v (512) ──┘
```

Chaque vecteur est ensuite **normalisé** (ramené à une longueur de 1) :

$$
\hat{u} = \frac{u}{\lVert u \rVert}, \qquad \hat{v} = \frac{v}{\lVert v \rVert}, \qquad \text{avec } \lVert u \rVert = \sqrt{\sum_{k=1}^{512} u_k^2}
$$

#### c) La similarité cosinus

La similarité cosinus mesure **l'angle** entre deux vecteurs, sans tenir compte de leur longueur :

$$
\cos(u, v) = \frac{u \cdot v}{\lVert u \rVert \, \lVert v \rVert} = \frac{\sum_{k=1}^{n} u_k v_k}{\sqrt{\sum_{k=1}^{n} u_k^2}\;\sqrt{\sum_{k=1}^{n} v_k^2}}
$$

| Valeur | Interprétation |
|---|---|
| $\cos = 1$ | même direction → même sens |
| $\cos = 0$ | vecteurs perpendiculaires → aucun rapport |
| $\cos = -1$ | directions opposées |

Si les vecteurs sont déjà normalisés ($\lVert \hat u \rVert = \lVert \hat v \rVert = 1$), le cosinus est simplement le **produit scalaire** : $\cos(\hat u, \hat v) = \hat u \cdot \hat v$.

**Exemple à la main** (en 3 dimensions) : $u = (1, 2, 2)$ et $v = (2, 1, 2)$

$$
u \cdot v = 1\times2 + 2\times1 + 2\times2 = 8, \qquad \lVert u \rVert = \lVert v \rVert = \sqrt{1+4+4} = 3
$$

$$
\cos(u, v) = \frac{8}{3 \times 3} \approx 0{,}889
$$

**Vérification en Python :**

```python
import numpy as np

u = np.array([1, 2, 2])
v = np.array([2, 1, 2])
cos = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
print(round(cos, 3))   # 0.889
```

#### d) La fonction de perte (loss) : InfoNCE symétrique

Pendant l'entraînement, on prend un lot de $N$ couples $(I_1, T_1), \dots, (I_N, T_N)$ : l'image $I_i$ va avec la légende $T_i$.

**1. Matrice de similarités.** On calcule le cosinus de **chaque image avec chaque texte**, multiplié par un facteur d'échelle $s = 1/\tau$ (τ = *température*, apprise par le modèle ; $s$ est plafonné à 100) :

$$
L_{ij} = s \cdot \cos(\hat u_i, \hat v_j) = \frac{\hat u_i \cdot \hat v_j}{\tau}
$$

```
                texte T1   texte T2   texte T3
image I1      [  L11 ✅     L12 ❌     L13 ❌  ]
image I2      [  L21 ❌     L22 ✅     L23 ❌  ]
image I3      [  L31 ❌     L32 ❌     L33 ✅  ]
```

Les **bonnes paires** sont sur la **diagonale** (exemples *positifs*) ; tout le reste sert d'exemples *négatifs*.

**2. Perte image → texte.** Pour chaque image, on veut que sa vraie légende gagne parmi les $N$ légendes. C'est une **entropie croisée** sur chaque ligne (softmax) :

$$
\mathcal{L}_{I \to T} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{L_{ii}}}{\sum_{j=1}^{N} e^{L_{ij}}}
$$

**3. Perte texte → image.** Même chose dans l'autre sens, sur chaque colonne :

$$
\mathcal{L}_{T \to I} = -\frac{1}{N} \sum_{j=1}^{N} \log \frac{e^{L_{jj}}}{\sum_{i=1}^{N} e^{L_{ij}}}
$$

**4. Perte totale** (la moyenne des deux) :

$$
\boxed{\;\mathcal{L}_{\text{CLIP}} = \frac{1}{2}\left(\mathcal{L}_{I \to T} + \mathcal{L}_{T \to I}\right)\;}
$$

Minimiser cette perte **augmente** le cosinus des bonnes paires (diagonale) et **diminue** celui des mauvaises paires. Plus le lot est grand, plus il y a de négatifs, et meilleur est l'apprentissage (d'où les lots de 79 000 couples).

**Rôle de la température τ :** plus τ est petit, plus le softmax est « tranchant » : le modèle est fortement pénalisé dès qu'une mauvaise paire se rapproche de la bonne.

**Exemple chiffré** (une seule ligne, $N = 3$, avec $s = 10$ pour garder des nombres lisibles) :

Cosinus de l'image $I_1$ avec les 3 textes : $(0{,}30;\ 0{,}10;\ 0{,}05)$ → logits $L_{1j} = (3;\ 1;\ 0{,}5)$

$$
p_{11} = \frac{e^{3}}{e^{3} + e^{1} + e^{0,5}} = \frac{20{,}09}{20{,}09 + 2{,}72 + 1{,}65} \approx 0{,}82
\qquad\Rightarrow\qquad -\log(0{,}82) \approx 0{,}20
$$

Si le modèle confondait les textes (cosinus tous égaux), on aurait $p_{11} = 1/3$ et une perte de $-\log(1/3) \approx 1{,}10$. **La perte baisse quand le modèle distingue bien la bonne légende.**

**En code (pseudo-PyTorch, d'après l'article CLIP) :**

```python
# u : (N, 512) vecteurs images,  v : (N, 512) vecteurs textes
u = u / u.norm(dim=1, keepdim=True)          # normalisation
v = v / v.norm(dim=1, keepdim=True)
logits = s * u @ v.T                          # matrice N x N des L_ij
cibles = torch.arange(N)                      # la bonne réponse est sur la diagonale
loss_i = F.cross_entropy(logits, cibles)      # image -> texte (lignes)
loss_t = F.cross_entropy(logits.T, cibles)    # texte -> image (colonnes)
loss = (loss_i + loss_t) / 2
```

#### e) Le lien avec la `distance` renvoyée par ChromaDB

La collection du TP utilise l'espace **`cosine`** (visible dans la configuration de la base). ChromaDB renvoie alors une **distance cosinus** :

$$
d(u, v) = 1 - \cos(u, v) \qquad (0 = \text{identique},\ 1 = \text{sans rapport},\ 2 = \text{opposé})
$$

Dans le TP, la recherche « ouverte » renvoie une distance de **0,79**, soit un cosinus de $1 - 0{,}79 = 0{,}21$. Ce n'est pas « faible » : entre une image et un texte, CLIP donne en pratique des cosinus autour de **0,2 à 0,35**. Ce qui compte, c'est le **classement** : l'image renvoyée est celle qui a le plus grand cosinus parmi toutes.

> 💡 Pour des vecteurs normalisés, la distance euclidienne et le cosinus sont liés : $\lVert \hat u - \hat v \rVert^2 = 2 - 2\cos(\hat u, \hat v)$. Elles donnent donc **le même classement**.

### 2.5 STT, TTS, SMTP

| Sigle | Signification | Bibliothèque utilisée |
|---|---|---|
| **STT** | Speech-to-Text : voix → texte | `SpeechRecognition` (+ `pyaudio` pour le micro) |
| **TTS** | Text-to-Speech : texte → voix | `gTTS` (Google Text-to-Speech) |
| **SMTP** | Protocole d'envoi d'emails | `smtplib` (inclus dans Python) |

---

## 3. Préparer l'environnement

### 3.1 Créer un environnement virtuel

Dans un terminal, à la racine du projet :

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
python -m pip install -r dependencies/requirements.txt
```

**macOS / Linux :** installez d'abord PortAudio (nécessaire pour le micro) :

```bash
brew install portaudio             # macOS
sudo apt install portaudio19-dev   # Linux
```

> `pywin32` ne s'installe que sous Windows. Sur macOS/Linux, supprimez la ligne du fichier `requirements.txt` si l'installation échoue.

### 3.2 Choisir le bon noyau dans le notebook

Dans VS Code ou Jupyter : **Select Kernel → `.venv`**.

Si vous installez un paquet depuis le notebook, utilisez **`%pip`** et non `!pip` :

```python
%pip install pillow chromadb open_clip_torch tqdm ipykernel
```

`%pip` installe dans le Python du noyau actif. `!pip` lance une commande shell qui peut viser un autre Python (ou ne pas exister du tout : `command not found: pip`).

---

## 4. Le TP pas à pas

> ⚠️ **Exécutez les cellules dans l'ordre.** Si vous redémarrez le noyau, toutes les variables sont effacées : il faut tout relancer depuis le début.

### Étape 1 — Charger et afficher les images

```python
from PIL import Image
import glob

listeImages = []
for file_name in glob.glob("images/*.jpg"):
    listeImages.append(Image.open(file_name))

display(listeImages[1])
```

- `glob.glob("images/*.jpg")` renvoie la liste des fichiers `.jpg` du dossier.
- `Image.open` (bibliothèque Pillow) ouvre une image.
- `display` l'affiche dans le notebook.

### Étape 2 — Créer la base et la collection

```python
import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

client = chromadb.PersistentClient(path="db_storePortes")

collection = client.get_or_create_collection(
    name="portes_collection",
    data_loader=ImageLoader(),                      # sait ouvrir une image depuis son chemin
    embedding_function=OpenCLIPEmbeddingFunction(), # transforme image/texte en vecteur
)
```

- `PersistentClient` enregistre la base **sur le disque** (dossier `db_storePortes/`). Elle survit à la fermeture du notebook.
- `get_or_create_collection` crée la collection si elle n'existe pas, sinon la réutilise.
- Au premier lancement, OpenCLIP **télécharge le modèle** (quelques centaines de Mo) : soyez patients.

> 💡 Dans le notebook d'origine, la collection est rangée dans une variable nommée `chromadb`, ce qui **écrase le module** du même nom. Préférez un nom comme `collection` : c'est plus clair et évite des erreurs.

### Étape 3 — Préparer les identifiants et les chemins

```python
import os

indexImage = []
CheminImages = []
dossierImages = "images/"

for index, image_name in enumerate(sorted(os.listdir(dossierImages))):
    if image_name.endswith(".jpg"):
        indexImage.append(str(index))
        CheminImages.append(os.path.join(dossierImages, image_name))

print(indexImage)
print(CheminImages)
```

- Chaque image reçoit un **identifiant unique** (`"0"`, `"1"`, …). ChromaDB exige des identifiants de type **texte**, d'où `str(index)`.
- `sorted(...)` garantit le même ordre à chaque exécution, donc les mêmes identifiants.

### Étape 4 — Indexer les images

```python
collection.add(ids=indexImage, uris=CheminImages)
collection.count()   # → 9
```

Pour chaque chemin :

1. `ImageLoader` ouvre l'image ;
2. OpenCLIP calcule son **vecteur de 512 nombres** ;
3. ChromaDB enregistre `id + chemin + vecteur`.

> **Relancer `add` avec les mêmes ids** produit des avertissements de doublons : les images sont déjà dans la base. Utilisez `collection.upsert(...)` pour ajouter **ou** mettre à jour sans erreur.

### Étape 5 — Rechercher une image par du texte

```python
resultat = collection.query(
    query_texts=["porte ouverte"],
    n_results=1,
    include=["uris", "distances"],
)
print(resultat)

for chemin in resultat["uris"][0]:
    display(Image.open(chemin))
```

- Le texte est transformé en vecteur par **le même modèle** que les images.
- ChromaDB renvoie les `n_results` images dont le vecteur est le plus proche.
- `distances` : **plus la valeur est petite, plus l'image correspond** au texte. Ici $d = 1 - \cos$ (voir § 2.4 e).

Exemple de résultat :

```python
{'ids': [['3']],
 'uris': [['images/81834LE2wZY.jpg']],
 'distances': [[0.79]]}
```

`[0]` sert à prendre les résultats de **la première requête** (on peut envoyer plusieurs textes à la fois).

### Étape 6 — Rechercher avec la voix (STT)

```python
import speech_recognition as sr

recognizer = sr.Recognizer()
microphone = sr.Microphone()

with microphone as source:
    recognizer.adjust_for_ambient_noise(source)   # calibre le bruit ambiant
    print("Parlez...")
    audio = recognizer.listen(source, timeout=5)

try:
    texte = recognizer.recognize_google(audio, language="fr-FR")
    print("Vous avez dit :", texte)

    resultat = collection.query(query_texts=[texte], n_results=1, include=["uris"])
    for chemin in resultat["uris"][0]:
        display(Image.open(chemin))

except sr.UnknownValueError:
    print("Je n'ai pas compris.")
except sr.RequestError as e:
    print("Erreur du service de reconnaissance :", e)
```

- `recognize_google` envoie l'audio à un service Google : **connexion Internet obligatoire**.
- Sur macOS, autorisez l'accès au micro pour VS Code / le terminal (*Réglages → Confidentialité → Microphone*).

### Étape 7 — Envoyer un email (SMTP)

**Ne jamais écrire un mot de passe dans le code.** On le range dans un fichier `.env`, qui ne doit **pas** être publié (ajoutez-le au `.gitignore`).

Fichier `.env` :

```env
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL=votre.adresse@gmail.com
PASSWORD=votre_mot_de_passe_application
```

Pour Gmail, créez un **mot de passe d'application** (validation en deux étapes obligatoire) : <https://myaccount.google.com/apppasswords>.

```python
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()   # lit le fichier .env

expediteur = os.getenv("EMAIL")
destinataire = "destinataire@example.com"

message = MIMEMultipart()
message["From"] = expediteur
message["To"] = destinataire
message["Subject"] = "Information porte"
message.attach(MIMEText("Bonjour ! Une personne vient de sonner à la porte.", "plain"))

try:
    serveur = smtplib.SMTP(os.getenv("SMTP_SERVER"), int(os.getenv("SMTP_PORT")))
    serveur.starttls()                                   # chiffre la connexion
    serveur.login(expediteur, os.getenv("PASSWORD"))     # authentification
    serveur.sendmail(expediteur, destinataire, message.as_string())
    serveur.quit()
    print("Email envoyé !")
except Exception as e:
    print("Erreur lors de l'envoi :", e)
```

| Étape | Rôle |
|---|---|
| `MIMEMultipart` | Construit un email (en-têtes + corps) |
| `starttls()` | Passe la connexion en chiffré (port 587) |
| `login()` | S'authentifie auprès du serveur |
| `sendmail()` | Envoie le message |

### Étape 8 — Faire parler l'ordinateur (TTS)

```python
from gtts import gTTS
import os, sys, tempfile

texte = "Quelqu'un sonne à la porte."
tts = gTTS(text=texte, lang="fr")

with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
    fichier = f.name
tts.save(fichier)

# Lecture selon le système
if sys.platform == "win32":
    os.system(f'start "" "{fichier}"')
elif sys.platform == "darwin":
    os.system(f'afplay "{fichier}"')
else:
    os.system(f'xdg-open "{fichier}"')
```

- `gTTS` génère un **MP3** (Internet nécessaire).
- La commande de lecture dépend du système : `start` (Windows), `afplay` (macOS), `xdg-open` (Linux).

---

## 5. Comment la base est stockée

Le dossier `db_storePortes/` contient :

```
db_storePortes/
├── chroma.sqlite3          ← collections, ids, chemins des images (base SQLite)
└── <uuid>/                 ← index vectoriel HNSW
    ├── data_level0.bin     ← les vecteurs (512 nombres par image)
    └── header.bin, length.bin, link_lists.bin
```

- Les **images ne sont pas copiées** dans la base : seuls leur **chemin** et leur **vecteur** sont stockés. Si vous déplacez le dossier `images/`, les chemins deviennent invalides.
- **HNSW** est une structure en graphe qui permet de trouver les vecteurs proches **sans comparer avec tous** : c'est ce qui rend la recherche rapide même avec des millions d'éléments.

Pour regarder le contenu depuis le notebook :

```python
data = collection.get(include=["uris", "embeddings"])
for id_, uri, vec in zip(data["ids"], data["uris"], data["embeddings"]):
    print(id_, uri, "→", len(vec), "valeurs :", vec[:5].round(3), "...")
```

Pour ouvrir la partie SQLite : extension VS Code **SQLite Viewer**, ou dans un terminal :

```bash
sqlite3 db_storePortes/chroma.sqlite3
SELECT * FROM embedding_metadata;
```

---

## 6. Erreurs fréquentes

| Message | Cause | Solution |
|---|---|---|
| `zsh: command not found: pip` | `pip` absent du système (macOS) | `python3 -m pip ...` dans le terminal, `%pip ...` dans le notebook |
| `ModuleNotFoundError: No module named 'chromadb'` | Paquet non installé **dans le noyau actif** | Vérifier le noyau (`.venv`), puis `%pip install chromadb` et redémarrer le noyau |
| `NameError: name 'indexImage' is not defined` | Cellule exécutée avant celle qui crée la variable, ou noyau redémarré | Relancer les cellules **dans l'ordre** (ou *Run All*) |
| Avertissement *existing embedding ID* | `add` relancé avec des ids déjà présents | Utiliser `upsert` ou ne pas relancer `add` |
| `OSError` / erreur PortAudio | `pyaudio` sans PortAudio | `brew install portaudio` puis réinstaller `pyaudio` |
| `SMTPAuthenticationError` | Mauvais mot de passe ou mot de passe Gmail normal | Utiliser un **mot de passe d'application** |
| `UnknownValueError` (STT) | Voix inaudible ou trop de bruit | Parler plus près du micro, relancer |

---

## 7. Exercices

### Exercice 1 — Plusieurs résultats
Modifiez la recherche pour afficher les **3 images** les plus proches de « porte en bois », avec leur distance sous chaque image.

### Exercice 2 — Comparer des requêtes
Testez « porte ouverte », « porte fermée », « porte rouge », « open door ». Notez l'image renvoyée et la distance.
**Question :** le modèle comprend-il mieux le français ou l'anglais ? Pourquoi, selon vous ?

### Exercice 3 — Ajouter des images
Ajoutez 3 nouvelles photos de portes dans `images/`, puis indexez-les **sans recréer** les 9 premières (indice : `upsert`, ou des ids qui ne sont pas encore utilisés).

### Exercice 4 — Métadonnées
Ajoutez à chaque image une métadonnée `{"etat": "ouverte"}` ou `{"etat": "fermee"}` (paramètre `metadatas=` de `add`/`upsert`).
Puis faites une recherche limitée aux portes fermées avec `where={"etat": "fermee"}`.

### Exercice 5 — Le scénario complet
Écrivez une fonction `sonnette()` qui :
1. écoute la voix du visiteur ;
2. affiche l'image de porte la plus proche de ce qu'il a dit ;
3. envoie un email contenant le texte reconnu ;
4. répond à voix haute : « Merci, votre message a été transmis. »

### Exercice 6 — Calculer la similarité soi-même
1. À la main, calculez $\cos(u, v)$ pour $u = (3, 4)$ et $v = (4, 3)$. *(Réponse attendue : 0,96.)*
2. Récupérez les vecteurs de deux images avec `collection.get(ids=["0", "3"], include=["embeddings"])`, calculez leur cosinus avec NumPy, puis la distance $1 - \cos$.
3. Calculez le vecteur du texte « porte ouverte » avec `OpenCLIPEmbeddingFunction()(["porte ouverte"])`, puis son cosinus avec chacune des 9 images. Retrouvez-vous le même classement que `collection.query` ?

### Exercice 7 — La loss à la main
Pour une image, les cosinus avec 3 légendes sont $(0{,}25;\ 0{,}20;\ 0{,}05)$ et la bonne légende est la première. Avec $s = 10$ :
1. Calculez les logits, la probabilité softmax de la bonne légende et la perte $-\log p$.
2. Recommencez avec $s = 100$. Que remarquez-vous ? Quel est l'effet de la température ?

### Exercice 8 (bonus) — Sécurité
Le notebook d'origine contient un mot de passe écrit en clair.
1. Expliquez en 3 lignes pourquoi c'est dangereux, surtout avec Git.
2. Que faut-il faire si un mot de passe a déjà été publié sur GitHub ?

---

## 8. Questions de révision

1. Quelle est la différence entre une recherche SQL classique et une recherche vectorielle ?
2. Pourquoi peut-on chercher une **image** avec un **texte** grâce à CLIP ?
3. Quel type d'apprentissage utilise CLIP ? En quoi diffère-t-il d'une classification supervisée ?
4. Écrivez la formule de la similarité cosinus. Pourquoi normalise-t-on les vecteurs ?
5. Dans la loss de CLIP, où se trouvent les exemples positifs et négatifs dans la matrice des similarités ? Quel est le rôle de la température τ ?
6. ChromaDB renvoie une distance de 0,79 : quel est le cosinus correspondant ?
7. Que contient un embedding produit par OpenCLIP ici (type, taille) ?
8. Que se passe-t-il si on renomme le dossier `images/` après l'indexation ?
9. Quelle est la différence entre `add` et `upsert` ?
10. Pourquoi utilise-t-on `starttls()` avant `login()` ?
11. Quelles étapes du TP ont besoin d'une connexion Internet ?

---

## 9. Pour aller plus loin

- Documentation ChromaDB : <https://docs.trychroma.com/>
- OpenCLIP : <https://github.com/mlfoundations/open_clip>
- SpeechRecognition : <https://github.com/Uberi/speech_recognition>
- gTTS : <https://gtts.readthedocs.io/>
- Pistes : reconnaissance vocale hors ligne (Whisper), TTS hors ligne (`pyttsx3`), envoi de la photo en pièce jointe de l'email.
