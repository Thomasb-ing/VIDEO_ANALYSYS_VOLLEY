# Essais sur Chevry

## VballNet (poids ONNX disponibles)

Les deux poids de `data/poids/vball-net-model.tgz` sont extraits dans `test-SOA/models/`.
`test_vball.py` adapte le pre/post-traitement RGB du depot `vball-net/src/inference_onnx.py`.
Il utilise le dernier extrait `outputs/chevry_*/chevry_raw.mp4` (chemin affiche au lancement),
ou un extrait explicite via `--video`. Aucune extraction supplementaire n'est necessaire.

Depuis la racine, avec la venv existante :

```powershell
.\test-SOA\.venv\Scripts\python.exe test-SOA/test_vball.py --model fast --duration 10 --threshold 0.5
.\test-SOA\.venv\Scripts\python.exe test-SOA/test_vball.py --model standard --duration 120 --threshold 0.5
```

Dependances presentes dans la venv lors de la creation du script. Pour une nouvelle venv :
`python -m pip install -r test-SOA/requirements-vball.txt`.
Le script fonctionne sur CPU et ne charge pas de modele Keras. L'ecriture CSV utilise
la bibliotheque standard, donc aucune dependance pandas n'est requise.

Chaque essai cree `outputs/vball_<modele>_<date>/` avec `annotated.mp4` (sans audio),
`detections.csv` et `summary.json`. Aucun ecrasement des essais precedents.
Le CSV contient les temps dans l'extrait et dans le match (si `extraction.json` est present),
les coordonnees en pixels de l'image originale et le maximum de la heatmap. Coordonnees
vides lorsqu'aucun ballon n'est detecte. Les deux premieres images sont marquees
`warmup=1` car l'historique est complete par repetition de la premiere image.
Le taux de detection ne mesure pas la precision ; le maximum de heatmap n'est pas une
confiance calibree. Pas de lissage ni d'interpolation. Une duree depassant l'extrait est refusee.

## Volleyball-Metrics (installation initiale)

### Comparaison des filtres couleur

`test_filtre_ballon_couleur_forme.py` selectionne 40 images avec detection VballNet
(hors initialisation), reparties dans le dernier essai standard de 120 secondes.
Il relit la video brute et produit une galerie HTML, les images originales,
les versions jaune/bleu sur fond noir, les masques binaires et quatre planches de zooms.

```powershell
.\test-SOA\.venv\Scripts\python.exe test-SOA/test_filtre_ballon_couleur_forme.py
```

Options : `--run DOSSIER`, `--count 40`, `--s-min 100`, `--v-min 80`, `--crop-size 96`.
Plages HSV OpenCV : jaune H=20..35, bleu H=95..130. Aucun critere de forme ni
nettoyage morphologique : il s'agit d'un diagnostic couleur. Les detections VballNet
peuvent etre fausses ; les croix des zooms indiquent seulement ses propositions.
Resultats : `outputs/filtres_couleur_<date>/index.html` et `selection.csv`.

Le script `test_chevry.py` utilise les fonctions du depot voisin sans le modifier.
Il recherche le MP4 contenant `Chevry` dans `../data`, extrait par defaut
01:00 a 03:00 sans ajouter d'annotations, puis execute calibration, segmentation
des echanges, suivi des joueurs, ballon, actions, statistiques et rendu.
Ce sont 120 secondes continues, temps morts inclus, pas 120 secondes de jeu effectif.
Le choix de cette plage n'a pas encore ete verifie visuellement.

## 1. Venv (PowerShell, depuis la racine du projet)

Python 3.12 est present sur cette machine. Une venv isolee a ete creee dans
`test-SOA/.venv`. Pour la recreer si elle n'existe pas :

```powershell
python -m venv test-SOA/.venv
```

L'activation est facultative : toutes les commandes utilisent explicitement son Python.

## 2. Installation

Pour extraire uniquement le MP4 :

```powershell
.\test-SOA\.venv\Scripts\python.exe -m pip install --upgrade pip
.\test-SOA\.venv\Scripts\python.exe -m pip install imageio-ffmpeg opencv-python
.\test-SOA\.venv\Scripts\python.exe test-SOA/test_chevry.py --mode extract --start 60
```

`imageio-ffmpeg` fournit FFmpeg : pas besoin de l'installer dans le PATH.
L'encodage H.264 permet une coupe precise sans se limiter aux images cles.
La source reste intacte ; chaque essai a son propre dossier dans `outputs`.

Pour l'analyse, choisir d'abord une installation PyTorch. Option CPU, plus lente :

```powershell
.\test-SOA\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\test-SOA\.venv\Scripts\python.exe -m pip install -r test-SOA/requirements.txt
.\test-SOA\.venv\Scripts\python.exe -m pip check
.\test-SOA\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Pour NVIDIA, remplacer la commande PyTorch CPU par celle du selecteur officiel
https://pytorch.org/get-started/locally/ (Windows / Pip / Python / CUDA compatible
avec le pilote). Employer le Python de la venv a la place de `pip3`.
Le reste du fichier utilise ONNX Runtime CPU pour la re-identification.

Le requirements du Backend d'origine fige notamment `torch==2.13.0+cu132` et
`torchvision==0.28.0+cu132` : ne pas le recopier aveuglement sur une autre machine.
Notre fichier limite les dependances a cet essai direct ; Node, React et FastAPI
ne sont pas necessaires. Les versions Ultralytics/Transformers sont celles du
depot, dont le code utilise des APIs internes : installation et compatibilite
restent a valider. Si pip ne trouve pas ces versions, ne pas supposer qu'une
ancienne version sera compatible sans verifier ces APIs.

## 3. Poids indispensables (absents du clone actuel)

Deposer les deux detecteurs specialises du projet dans :

```text
Volleyball-Metrics/Backend/MachineLearning/models/ballDetection_yolo26x_best.pt
Volleyball-Metrics/Backend/MachineLearning/models/ballDetection_yolo11x_best.pt
```

Le README ne fournit pas de lien direct verifie pour ces deux poids : les obtenir
aupres du projet https://github.com/kairsato/Volleyball-Metrics ou les entrainer
avec ses scripts/datasets. Renommer un YOLO generique ne remplace pas ces modeles.

Pour les echanges, recuperer le checkpoint VideoMAE **3 etats** indique par :
https://github.com/masouduut94/volleyball_analytics/wiki/Datasets-and-Weights

Le placer dans :

```text
Volleyball-Metrics/Backend/Analysis/GameStatusDetection/Models/VolleyballAnalytics/3-states/checkpoint/
    config.json
    preprocessor_config.json
    model.safetensors
```

Conserver aussi les autres fichiers fournis avec ce checkpoint. Le code local
lit explicitement `model.safetensors`. Il privilegie `Models/Custom/checkpoint`
si ce dossier existe : ne pas creer un dossier Custom vide.

`Analysis/ActionDetection/action_detector.pt` est facultatif : sans lui,
Volleyball-Metrics utilise des heuristiques et les types d'actions doivent etre
consideres comme provisoires. Le modele generique joueurs `yolo26x.pt` et
le modele ReID `yolo26m-reid.onnx` sont normalement telecharges par Ultralytics
au premier usage ; cela requiert un acces reseau et la version compatible.

## 4. Verification et lancement

```powershell
.\test-SOA\.venv\Scripts\python.exe test-SOA/test_chevry.py --mode check
.\test-SOA\.venv\Scripts\python.exe test-SOA/test_chevry.py --mode run --start 60 --duration 120
```

Le mode check verifie la presence des bibliotheques et des checkpoints locaux,
pas leur qualite ni toute leur compatibilite. Le mode run importe les modules
avant la calibration et s'arrete si une sortie attendue manque.

Une fenetre OpenCV permet de placer les reperes du terrain et du filet sur la
premiere image de l'extrait. Suivre les libelles, puis Entree pour confirmer ;
Echap annule. Choisir un debut avec terrain visible et camera stable : ce script
ne recalcule pas la calibration lors d'un changement de plan.

Sorties dans `test-SOA/outputs/chevry_<date>/` :

- `chevry_raw.mp4` : extrait non annote.
- `extraction.json` : provenance, debut et duree mesuree.
- `analysis/analysis.mp4` : rendu annote par Volleyball-Metrics.
- `analysis/` : JSON terrain, echanges, joueurs, ballon, actions, statistiques et dashboard HTML.

Les identifiants dessines sont des identifiants de suivi, pas des numeros de
maillot verifies. Cet essai n'inclut pas les corrections de score et de noms
proposees par l'interface web.

## Etat de verification

La venv est creee. L'installation des dependances depuis cet environnement a
echoue sur une restriction reseau (`WinError 10013`). L'extraction et l'inference
n'ont donc pas encore ete executees ; les checkpoints specialises sont egalement
absents. Les commandes ci-dessus sont a executer dans un terminal avec acces PyPI.
