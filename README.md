# PROJET D'ANALYSE VIDEO
 L'idée est de transformer une vidéo de volley, en une structure data permettant l'analyse de points de cette vidéo (posiiton des joueurs sur le terrain, trajectoire de la balle, qui touche la balle, bloc, service, faute ...)


# Etapes clés
## Détection du terrain
La première étape est alors de detecter dans la vidéo le terrain, puis le filet.
Ensuite on pourra alors reconstruire depuis la vidéo, une homographie permettant la bijection entre un pixel de la vidéo et un point sur un terrain (euclidien).

## Détection des joueurs
Détecter les joueurs (pieds + numero de maillot)
Ainsi qu'un  tracking sur les différentes frames

## Détection du ballon + trajectoire de ballon


## Construction des data de points
Maintenant on peut , via changement de direction de balle, deviner quel joueur a touché le ballon, et donc construire une data organisé comme {point=0; score=[0,0], chaine={18->12->5->...}
Où chaine a la forme la plus souhaitable pour décrire la physionomie du point (qui touche la balle, l'ordre, les coordonnées lors de la touche ...)



----------------------------------------------------------------------------
------------------------------------------------------------------------
------------------------------------------
**Je te conseille une approche hybride : tester Volleyball-Metrics comme première référence, réutiliser les briques qui fonctionnent sur tes vidéos, et développer toi-même la construction et la validation des données de points.** Tout réécrire serait coûteux ; reprendre l’application entière sans évaluation risquerait de produire des statistiques trompeuses.

J’ai lu ton README, celui de Volleyball-Metrics déjà présent dans le dossier, examiné une partie du code et recherché d’autres projets. Je n’ai pas exécuté les modèles : les performances citées ci-dessous sont celles annoncées par leurs auteurs.

**Les projets les plus pertinents**

| Projet | Intérêt pour ton objectif | Mon conseil |
|---|---|---|
| [Volleyball-Metrics](<https://github.com/kairsato/Volleyball-Metrics>) | Chaîne déjà assemblée : terrain, joueurs, ballon, échanges, actions, interface et statistiques. | Premier candidat à essayer sur tes vidéos. |
| [volleyball\_analytics](<https://github.com/masouduut94/volleyball_analytics>) | Plusieurs modèles spécialisés ; notamment VideoMAE pour distinguer service, jeu et arrêt. | Source intéressante pour découper les échanges. |
| [VolleyVision](<https://github.com/shukkkur/VolleyVision>) | Détection du ballon, joueurs, actions et terrain ; jeux de données associés. L’auteur indique que la production complète de statistiques n’a pas abouti. | Réservoir de données et d’idées plutôt que solution complète. |
| [VNL-STES — CVPR Workshops 2025](<https://hoangqnguyen.github.io/stes/>) | Détection du moment **et du lieu** des événements de volley. Données : 1 028 échanges et 6 137 événements annotés, issus de huit matchs. | La piste de recherche la plus directement alignée sur ta chaîne de touches. |
| [OpenVolley / datavolley](<https://github.com/openvolley/datavolley>) | Lecture de fichiers de scouting pour l’analyse statistique. | Référence pour organiser les données métier ; ne fait pas l’extraction vidéo. |

VNL-STES mérite particulièrement ton attention : il cherche directement à localiser les événements, sans exiger au préalable une trajectoire parfaite du ballon sur chaque image. Ses catégories et ses annotations devront cependant être adaptées à ta définition des actions. Ce n’est pas un système complet d’identification des joueurs. [Présentation et code](<https://hoangqnguyen.github.io/stes/>).

**Ce que je retiens de Volleyball-Metrics**

Son architecture correspond très bien à ton README. Il prévoit aussi une calibration et une identification assistées par l’utilisateur, utiles pour démarrer.

En revanche, son README rapporte :

- un rappel du ballon de **0,758** sur son propre jeu de validation ;
- aucune évaluation formelle du suivi des identités ;
- aucune évaluation formelle de la classification des actions ;
- un modèle de segmentation des échanges à récupérer séparément.

Ce sont des limites directement pertinentes pour tes statistiques : une touche manquée ou attribuée au mauvais joueur change la chaîne du point. [README du projet](<https://github.com/kairsato/Volleyball-Metrics>).

Dans le code local, la recherche de contacts exploite effectivement les changements de direction et de vitesse du ballon. Elle exclut les positions interpolées, ce qui est une bonne précaution. Mais **une heuristique plausible n’est pas encore une mesure validée**.

Je l’utiliserais donc pour produire un premier résultat et mesurer ses erreurs, avant de décider quelles parties conserver.

**Deux ajustements importants à ton approche**

Le premier concerne la géométrie. Ton README parle d’« homothétie » : il faut une **homographie**, qui relie le plan du sol à son image. Elle convient aux positions des pieds lorsqu’ils sont au sol, mais appliquer cette transformation au ballon en l’air ne donne pas sa véritable projection verticale sur le terrain. Même connaître la caméra ne lève pas, à lui seul, l’ambiguïté de profondeur d’un point observé dans une seule image. [Explication OpenCV](<https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html>).

Pour une première version, je conserverais donc les **positions du ballon en pixels**, les **positions des joueurs au sol en mètres**, et reporterais les vitesses et trajectoires 3D.

Le deuxième concerne les contacts. Le changement de direction du ballon est un indice, mais il faudra le combiner avec la proximité des joueurs, leur posture et le contexte temporel. Une attaque et un bloc peuvent se succéder très rapidement ; une occultation peut masquer précisément le contact recherché. VNL-STES mentionne lui-même ces difficultés. [Limites du modèle](<https://hoangqnguyen.github.io/stes/>).

**La démarche que je suivrais**

1. **Définir un premier périmètre simple.**  
   Commencer par des échanges en plan large, sur une ou deux vidéos représentatives. Si les vidéos YouTube comportent zooms, changements de caméra et ralentis, prévoir explicitement leur traitement : les ralentis ne doivent pas être comptés comme de nouveaux échanges et la calibration doit suivre les mouvements de caméra.
2. **Créer un petit corpus de référence avant d’entraîner quoi que ce soit.**  
   Annoter, par exemple, 30 à 50 échanges : début, fin, équipe gagnante, instant des touches, joueur et type d’action lorsque visibles. Ajouter quelques séquences annotées image par image pour évaluer le ballon et les identités. Ce corpus sert à diagnostiquer, pas encore à démontrer une généralisation.

   [CVAT](<https://www.cvat.ai/academy/track-mode>) peut faciliter l’annotation des objets et des trajectoires grâce aux pistes et à l’interpolation.
3. **Faire tourner Volleyball-Metrics sur exactement ces échanges.**  
   Mesurer les touches oubliées ou inventées, les erreurs de joueur et le temps nécessaire pour corriger les résultats. C’est ce test qui doit décider de l’adoption du projet.
4. **Construire une chaîne minimale dont chaque module est remplaçable.**  
   Pour commencer : calibration manuelle du terrain, détection des joueurs, un seul tracker, détection du ballon, propositions de contacts, puis correction humaine. Une base YOLO avec ByteTrack ou BoT-SORT est déjà disponible dans Ultralytics. Je ne commencerais pas par reproduire un ensemble de cinq trackers. [Documentation du suivi](<https://docs.ultralytics.com/modes/track/>).
5. **Améliorer uniquement le principal point faible observé.**  
   Si le ballon manque trop souvent : adapter le détecteur ou tester une approche temporelle de la famille TrackNet. Si les contacts sont mal localisés malgré une bonne détection : expérimenter STES. Si les identités changent : travailler l’association des pistes, puis l’identification par numéro. TrackNetV4 constitue une piste de suivi temporel, pas une garantie de performance immédiate sur le volley. [Article](<https://arxiv.org/abs/2409.14543>).

Je garderais initialement une **identité locale par échange**, avec une association manuelle au joueur réel. Lire automatiquement tous les numéros et maintenir les identités sur un match entier est un chantier supplémentaire.

**Les données à construire dès le départ**

Ta chaîne `18 → 12 → 5` est une bonne vue synthétique, mais je stockerais une ligne par événement :

| Table | Informations principales |
|---|---|
| `rallies` | Match, set, début, fin, score avant/après, équipe au service, gagnant. |
| `events` | Échange, ordre, instant, équipe, joueur éventuel, action, coordonnées, confiance, validation humaine. |
| `tracks` | Instant, identifiant de piste, position image, position au sol si disponible, visibilité. |

Il faut distinguer **identifiant de piste**, **identité du joueur** et **numéro de maillot**. Conserver aussi les valeurs inconnues et distinguer observation, interpolation et correction humaine.

Cela permet de commencer par les durées des échanges, les enchaînements d’actions et les résultats par équipe, puis d’ajouter les analyses individuelles lorsque l’identification devient suffisamment fiable. La cause précise d’une faute peut rester inconnue même lorsque le gagnant du point est connu.

Pour évaluer la chaîne, je regarderais surtout :

- précision et rappel des contacts, avec une tolérance temporelle annoncée ;
- attribution correcte du joueur et du type d’action ;
- proportion d’échanges dont la chaîne entière est correcte ;
- temps de correction par échange.

Les matchs servant au test doivent être séparés de ceux utilisés pour l’entraînement. Pour les statistiques, il faudra aussi vérifier si les erreurs se concentrent sur certains types d’actions : manquer davantage les blocs que les services crée un biais.

**Ma décision pour ton projet :** consacrer d’abord quelques jours à un essai comparatif sur un petit corpus annoté. Conserver Volleyball-Metrics s’il réduit nettement le travail de correction ; sinon, en reprendre les modules utiles dans une chaîne Python plus simple. Ton investissement principal devrait aller au **schéma des événements, à la validation et à la correction des annotations** : c’est ce qui rendra les analyses statistiques exploitables.