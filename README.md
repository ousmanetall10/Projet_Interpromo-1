# Projet Interpromo
Ce projet a été réalisé dans le cadre d’un travail collaboratif au sein du groupe g4pip.

Contributeur principal :
 • Ousmane Tall (membre de g4pip)
 
Ce projet exploite des données issues de **Open Data University** pour analyser et visualiser divers aspects liés aux ressources et activités universitaires. 
L'objectif est de fournir des insights exploitables à partir des données publiques disponibles.


# Tableau de Bord du Changement Climatique et des Catastrophes Naturelles

## Description
Ce tableau de bord interactif visualise la relation entre le changement climatique et les catastrophes naturelles en France. Il offre une visualisation complète des données à travers plusieurs vues :

- **Cartes** : Visualisation interactive des paramètres météorologiques et des catastrophes naturelles par département
- **Climat** : Analyse des tendances climatiques incluant température, précipitations et humidité
- **Catastrophes** : Analyse détaillée des catastrophes naturelles, leur distribution et évolution
- **Études de cas** : 
  - Analyse du changement climatique global avec focus sur les régions montagneuses
  - Analyse des fortes températures en Corse pendant l'été

## Fonctionnalités
- Cartes interactives avec données départementales
- Analyse temporelle des paramètres climatiques
- Distribution saisonnière des catastrophes naturelles
- Analyse comparative entre différentes périodes
- Multiples types de visualisation (cartes, graphiques, diagrammes circulaires)
- Vues et filtres personnalisables

## Prérequis
- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Plotly
- GeoPandas
- Scikit-learn

## Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/g4pip/Projet_Interpromo.git
cd tableau-bord-climat
```
3. Installer les packages requis :
```bash
pip install plotly.express
...
```

## Fichiers de Données Requis
Assurez-vous d'avoir les fichiers CSV suivants dans votre répertoire de projet :
- data_ville_annuel.csv
- data_ville_mensuel.csv
- catastrophes_naturelles_annuelles_dep_incendies.csv
- catastrophes_naturelles_mensuelles_dep_incendies.csv
- lignes_points_proches10.csv
- lignes_points_proches4.csv
- extracted_national_data.csv
- etude_de_cas_corse_ete.csv

## Lancement de l'Application

1. Naviguer vers le répertoire du projet :
```bash
cd Projet_Interpromo
```

2. Lancer l'application Streamlit :
```bash
python -m streamlit run .\dashboard.py
```


## Utilisation
Le tableau de bord est divisé en quatre onglets principaux :

1. **Onglet Cartes** :
   - Sélection de l'année via le curseur
   - Choix des paramètres météo ou types de catastrophes
   - Visualisation de la distribution géographique sur cartes interactives

2. **Onglet Climat** :
   - Sélection des départements et périodes
   - Visualisation des tendances climatiques
   - Comparaison entre différentes périodes

3. **Onglet Catastrophes** :
   - Filtrage par département et type de catastrophe
   - Analyse de la distribution saisonnière
   - Visualisation des tendances historiques

4. **Onglet Études de Cas** :
   - Analyse du changement climatique en montagne
   - Étude des températures en Corse

## Structure des Données
L'application attend la structure suivante dans les fichiers CSV :

### Données Annuelles des Villes (data_ville_annuel.csv) :
- Annee : Année
- Departement : Nom du département
- T_MENS : Température mensuelle
- PRENEI_MENS : Précipitations neigeuses
- PRELIQ_MENS : Précipitations liquides
- Et autres paramètres météo...

### Données des Catastrophes Naturelles (catastrophes_naturelles_annuelles_dep_incendies.csv) :
- Année : Année
- Département : Code département
- Type catastrophe : Type de catastrophe
- Nombre événements : Nombre d'événements

(Structure similaire pour les fichiers de données mensuelles)

## Contribution
Les contributions sont les bienvenues ! N'hésitez pas à soumettre une Pull Request.

## Contributeurs
[BOURRET Chloé,
DAUFES Louna,
DOUMA Nassira,
TALL Ousmane,
GONCUKLIYAN Maxime,
LARAVINE Noah,
BOUGHANEM Samy,
GHAZI Benjamin
]
