import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# =================================
# =     CONFIGURATION GLOBALE     =
# =================================
st.set_page_config(layout="wide")


# =================================
# =        FONCTIONS DE LOAD      =
# =================================

def load_data_main():
    return pd.read_csv("data_ville_annuel.csv")  # À adapter selon votre chemin

def load_data_disasters():
    return pd.read_csv("catastrophes_naturelles_annuelles_dep_incendies.csv")  # À adapter

def load_geometries():
    url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson"
    return gpd.read_file(url)

def load_data_monthly_disasters():
    return pd.read_csv("catastrophes_naturelles_mensuelles_dep_incendies.csv")  # À adapter

def load_data_monthly():
    return pd.read_csv("data_ville_mensuel.csv")  # À adapter

# =============================
# =    CHARGEMENT DES DONNÉES =
# =============================
main_data = load_data_main()
disasters_data = load_data_disasters()
geometries = load_geometries()
monthly_disasters_data = load_data_monthly_disasters()
monthly_data = load_data_monthly()

# Ajustement spécifique
main_data["T_MENS"] = main_data["T_MENS"].apply(lambda x: max(x, 9))  
disasters_data["Département"] = disasters_data["Département"].apply(lambda x: str(x).zfill(2))

# ==========================
# =    CALCULS GLOBAUX     =
# ==========================
param_cols = [
    "T_MENS", "PRENEI_MENS", "PRELIQ_MENS", "PRETOTM_MENS", "EVAP_MENS",
    "ETP_MENS", "PE_MENS", "SWI_MENS", "DRAINC_MENS", "RUNC_MENS",
    "ECOULEMENT_MENS"
]

# Min/max globaux pour la carte météo (onglet 1)
global_min_max_main = {
    param: (
        main_data[param].min() - 0.1 * abs(main_data[param].min()),
        main_data[param].max() - 0.1 * abs(main_data[param].max())
    ) for param in param_cols
}

# Min/max globaux pour les catastrophes (logarithmes)
disasters_data["Nombre_log"] = np.log1p(disasters_data["Nombre événements"])
global_min_disasters = disasters_data["Nombre_log"].min()
global_max_disasters = disasters_data["Nombre_log"].max()

# Min/Max globaux par type de catastrophe
global_min_max_disasters_by_type = (
    disasters_data.groupby("Type catastrophe")["Nombre_log"]
    .agg(["min", "max"])
    .reset_index()
)
global_min_max_disasters_by_type.rename(columns={"min": "global_min", "max": "global_max"}, inplace=True)


st.title("Réchauffement climatique et catastrophes naturelles")
st.markdown("""
Ce tableau de bord porte sur l’évolution des données météorologiques en France ainsi que les catastrophes naturelles. Il contient premièrement une page de cartes permettant de visualiser divers indicateurs météo et catastrophes naturelles. Sur la seconde page, vous trouverez les visuels portant sur les données météorologiques. Sur la page suivante se trouvent des visuels représentant les catastrophes naturelles. Enfin, sur la dernière page, vous trouverez deux études de cas : la première sur les changements climatiques en montage et la seconde sur les fortes températures en Corse en été.
""")

# =======================
# =  CRÉATION DES ONGLETS
# =======================
onglets = st.tabs(["Cartes", "Climat", "Catastrophes", "Étude de cas"])

# ================================================
# =                 ONGLET 1 : CARTES            =
# ================================================
with onglets[0]:
    # Curseur pour sélectionner une année
    min_year = int(min(main_data["Annee"].min(), disasters_data["Année"].min()))
    max_year = int(max(main_data["Annee"].max(), disasters_data["Année"].max()))
    selected_year = st.slider(
        "Sélectionnez une année", 
        min_value=min_year, 
        max_value=max_year, 
        value=min_year, 
        step=1
    )

    # Filtrer les données selon l'année sélectionnée
    filtered_main_data = main_data[main_data["Annee"] == selected_year]
    filtered_disasters_data = disasters_data[disasters_data["Année"] == selected_year]

    if filtered_main_data.empty:
        st.warning(f"Aucune donnée météorologique disponible pour l'année {selected_year}.")

    if filtered_disasters_data.empty:
        st.warning(f"Aucune donnée de catastrophes disponible pour l'année {selected_year}.")

    # Carte météo (col1) et Carte catastrophes (col2)
    col1, col2 = st.columns([1, 1])

    # =====================
    # Carte 1 : Météo
    # =====================
    with col1:
        param_main = st.selectbox(
            "Sélectionnez un paramètre météo", 
            options=param_cols,
            index=0
        )
        
        min_val_main, max_val_main = global_min_max_main[param_main]
        
        fig_main = px.scatter_mapbox(
            filtered_main_data,
            lat="latitude",
            lon="longitude",
            color=param_main,
            size=param_main,
            size_max=5,
            color_continuous_scale="jet",
            range_color=[min_val_main, max_val_main],
            opacity=0.7,
            mapbox_style="carto-positron",
            zoom=4,
            center={"lat": 46.603354, "lon": 2.469758},
            hover_name="Departement",
            hover_data={"latitude": True, "longitude": True, param_main: True},
        )

        fig_main.update_layout(
            height=500,
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )

        st.plotly_chart(fig_main, use_container_width=True)

    # =====================
    # Carte 2 : Catastrophes
    # =====================
    with col2:
        disaster_type = st.selectbox(
            "Sélectionnez un type de catastrophe", 
            options=disasters_data["Type catastrophe"].unique(), 
            index=0
        )
    
        filtered_disasters_by_type = filtered_disasters_data[filtered_disasters_data["Type catastrophe"] == disaster_type]
    
        # Récupération des min/max globaux pour ce type
        scale_row = global_min_max_disasters_by_type[
            global_min_max_disasters_by_type["Type catastrophe"] == disaster_type
        ]
        global_min_disaster_type = scale_row["global_min"].values[0] if not scale_row.empty else 0
        global_max_disaster_type = scale_row["global_max"].values[0] if not scale_row.empty else 1

        aggregated_disasters = (
            filtered_disasters_by_type
            .groupby("Département")["Nombre événements"]
            .sum()
            .reset_index()
        )
        aggregated_disasters["Nombre_log"] = np.log1p(aggregated_disasters["Nombre événements"])
        aggregated_disasters.rename(columns={"Département": "code"}, inplace=True)

        geo_merged = geometries.merge(aggregated_disasters, on="code", how="left")
        geo_merged["Nombre_log"] = geo_merged["Nombre_log"].fillna(0)
        geo_merged["Nombre événements"] = geo_merged["Nombre événements"].fillna(0)

        # Si pas de données
        if filtered_disasters_by_type.empty:
            # Carte grise
            geo_merged["Nombre_log"] = 0
            fig_disasters = px.choropleth_mapbox(
                geo_merged,
                geojson=geo_merged.geometry,
                locations=geo_merged.index,
                color="Nombre_log",
                color_continuous_scale=["#D3D3D3", "#D3D3D3"],
                range_color=(0, 1),
                mapbox_style="carto-positron",
                zoom=4,
                center={"lat": 46.603354, "lon": 1.888334},
                labels={"Nombre_log": "Nbre catastrophes (log)"},
                hover_name="nom",
                hover_data={"Nombre_log": False},
            )
            # Annotation
            fig_disasters.add_annotation(
                x=0.5, y=0.5, xref="paper", yref="paper",
                text="<b>Aucune donnée disponible pour cette année</b>",
                showarrow=False,
                font=dict(size=18, color="black"),
                align="center",
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="black",
                borderwidth=1,
            )
        else:
            # Carte avec données
            fig_disasters = px.choropleth_mapbox(
                geo_merged,
                geojson=geo_merged.geometry,
                locations=geo_merged.index,
                color="Nombre_log",
                color_continuous_scale="Reds",
                range_color=(global_min_disaster_type, global_max_disaster_type),
                mapbox_style="carto-positron",
                zoom=4,
                center={"lat": 46.603354, "lon": 1.888334},
                labels={"Nombre_log": "Nbre catastrophes (log)"},
                hover_name="nom",
                hover_data={"Nombre_log": True, "Nombre événements": True},
            )

        fig_disasters.update_layout(
            height=500, 
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )

        st.plotly_chart(fig_disasters, use_container_width=True)

# =======================
# =      ONGLET 2       =
# =       CLIMAT        =
# =======================
    with onglets[1]:
        # Sélection des départements à afficher (option multiple avec "Toute la France" par défaut)
        departements_selectionnes = st.multiselect(
            "Sélectionnez des départements (ou 'Toute la France' pour les moyennes globales)",
            options=["Toute la France"] + list(main_data["Departement"].unique()),
            default=["Toute la France"]
        )
    
        # Choisir entre "Toute l'année" ou "Mois spécifiques"
        periode_selectionnee = st.radio(
            "Sélectionnez la période",
            options=["Toute l'année", "Mois spécifiques"],
            index=0
        )
    
        # Sélection de mois spécifiques si "Mois spécifiques" est choisi
        if periode_selectionnee == "Mois spécifiques":
            mois_selectionnes = st.multiselect(
                "Sélectionnez des mois",
                options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            )
            # On désactive l'option "Courbes individuelles" en mode "Mois spécifiques"
            mode_affichage = "Courbe moyenne"  # Forcer à courbe moyenne
        else:
            # Si "Toute l'année", on garde les deux options disponibles
            mode_affichage = st.radio(
                "Mode d'affichage des courbes",
                options=["Courbes moyennes", "Courbes individuelles"],
                index=0
            )
    
        # Sélection de la plage de dates avec un curseur double
        min_year = main_data["Annee"].min()
        max_year = main_data["Annee"].max()
        start_year, end_year = st.slider(
            "Sélectionnez la plage de dates",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            step=1,
        )
    
        # Filtrage des données en fonction de la plage d'années
        main_data_filtered = main_data[(main_data["Annee"] >= start_year) & (main_data["Annee"] <= end_year)]
        
        # Paramètres à afficher
        parametres_graphique = ["T_MENS", "PRELIQ_MENS", "PRENEI_MENS", "SWI_MENS"]
    
        # Organisation des graphiques en deux colonnes
        col_graphs = st.columns(2)  # Crée 2 colonnes
    
        for i, parametre in enumerate(parametres_graphique):
            with col_graphs[i % 2]:  # Alterne entre la colonne 0 et 1
                # Choix des données à utiliser (annuelles ou mensuelles)
                if periode_selectionnee == "Toute l'année" or mois_selectionnes is None:
                    # Moyennes annuelles (en utilisant main_data)
                    if mode_affichage == "Courbes moyennes" or "Toute la France" in departements_selectionnes:
                        if "Toute la France" in departements_selectionnes:
                            moyenne = main_data_filtered.groupby("Annee")[parametre].mean().reset_index()
                            moyenne["Type"] = "Moyenne France"
                        else:
                            data_filtrée = main_data_filtered[main_data_filtered["Departement"].isin(departements_selectionnes)]
                            moyenne = data_filtrée.groupby("Annee")[parametre].mean().reset_index()
                            moyenne["Type"] = "Dept choisis"
    
                        data_to_plot = moyenne
                    else:
                        # Courbes individuelles pour les départements sélectionnés
                        data_filtrée = main_data_filtered[main_data_filtered["Departement"].isin(departements_selectionnes)]
                        moyenne_departements = data_filtrée.groupby(["Annee", "Departement"])[parametre].mean().reset_index()
                        moyenne_departements["Type"] = moyenne_departements["Departement"]
                        data_to_plot = moyenne_departements
                else:
                    # Moyennes mensuelles (en utilisant monthly_data)
                    monthly_data_filtered = monthly_data[
                        (monthly_data["Annee"] >= start_year) & (monthly_data["Annee"] <= end_year) & 
                        (monthly_data["Mois"].isin(mois_selectionnes))
                    ]
                    if "Toute la France" in departements_selectionnes:
                        moyenne = monthly_data_filtered.groupby(["Annee"])[parametre].mean().reset_index()
                        moyenne["Type"] = "Moyenne France"
                    else:
                        data_filtrée = monthly_data_filtered[monthly_data_filtered["Departement"].isin(departements_selectionnes)]
                        moyenne = data_filtrée.groupby(["Annee"])[parametre].mean().reset_index()
                        moyenne["Type"] = "Dept choisis"
    
                    data_to_plot = moyenne
                if parametre == "T_MENS":
                    title_temp = f"Tendance de températures ({start_year}-{end_year})"
                    legende="Température (°C)"
                elif parametre == "PRELIQ_MENS":
                    title_temp = f"Tendance de précipitations liquides ({start_year}-{end_year})"
                    legende="Précipitations (mm)"
                elif parametre == "PRENEI_MENS":
                    title_temp = f"Tendance de précipitations solides ({start_year}-{end_year})"
                    legende="Précipitations (mm)"
                elif parametre == "SWI_MENS":
                    title_temp = f"Tendance de l'indice de l'humidité ({start_year}-{end_year})"
                    legende="Indice d'humidité"
                
                # Création du graphique avec nuage de points, régression et moyenne mobile
                fig_graph = px.scatter(
                    data_to_plot,
                    x="Annee",
                    y=parametre,
                    color="Type",
                    trendline="lowess",  # Ajoute une courbe de régression (type LOWESS)
                    title=title_temp,
                    labels={"Annee": "Année",
                    parametre: legende
                    }
                )
    
                # Ajout d'une courbe moyenne mobile
                data_to_plot["Moyenne Mobile"] = data_to_plot.groupby("Type")[parametre].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
                for typ in data_to_plot["Type"].unique():
                    subset = data_to_plot[data_to_plot["Type"] == typ]
                    fig_graph.add_scatter(
                        x=subset["Annee"],
                        y=subset["Moyenne Mobile"],
                        mode="lines",
                        name=f"Moyenne Mobile ({typ})",
                        line=dict(dash="dash"),
                    )
    
                # Configuration du graphique
                fig_graph.update_layout(
                    height=400,  # Ajuste la hauteur pour une meilleure compacité
                    margin={"r": 0, "t": 50, "l": 0, "b": 0},
                )
    
                # Affichage du graphique
                st.plotly_chart(fig_graph, use_container_width=True)


# =======================
# =      ONGLET 3       =
# =   CATASTROPHES      =
# =======================
with onglets[2]:
    st.header("Visualisation des catastrophes naturelles")

    # Curseur pour sélectionner la plage d'années
    min_year_disasters = disasters_data["Année"].min()
    max_year_disasters = disasters_data["Année"].max()
    start_year, end_year = st.slider(
        "Sélectionnez la plage d'années",
        min_value=min_year_disasters,
        max_value=max_year_disasters,
        value=(min_year_disasters, max_year_disasters),
        step=1,
    )

    # Disposition côte à côte des listes déroulantes
    col1, col2 = st.columns(2)

    with col1:
        # Liste déroulante pour sélectionner les départements
        departements_disasters = ["Toute la France"] + list(disasters_data["Nom département"].unique())
        selected_departments = st.multiselect(
            "Sélectionnez des départements",
            options=departements_disasters,
            default=["Toute la France"]
        )

    with col2:
        # Liste déroulante pour sélectionner les types de catastrophes
        disaster_types = ["Tous les types"] + list(disasters_data["Type catastrophe"].unique())
        selected_types = st.multiselect(
            "Sélectionnez des types de catastrophes",
            options=disaster_types,
            default=["Tous les types"]
        )

    # Filtrage des données
    filtered_disasters = disasters_data[(
        disasters_data["Année"] >= start_year) &
        (disasters_data["Année"] <= end_year)
    ]
    filtered_monthly_disasters = monthly_disasters_data[(
        monthly_disasters_data["Année"] >= start_year) &
        (monthly_disasters_data["Année"] <= end_year)
    ]

    if "Toute la France" not in selected_departments:
        filtered_disasters = filtered_disasters[filtered_disasters["Nom département"].isin(selected_departments)]
        filtered_monthly_disasters = filtered_monthly_disasters[filtered_monthly_disasters["Nom département"].isin(selected_departments)]

    if "Tous les types" not in selected_types:
        filtered_disasters = filtered_disasters[filtered_disasters["Type catastrophe"].isin(selected_types)]
        filtered_monthly_disasters = filtered_monthly_disasters[filtered_monthly_disasters["Type catastrophe"].isin(selected_types)]

    # Créer une colonne "Saison" dans les données mensuelles
    saison_map = {
        12: "Hiver", 1: "Hiver", 2: "Hiver",  # Déc, Jan, Fév
        3: "Printemps", 4: "Printemps", 5: "Printemps",  # Mar, Avr, Mai
        6: "Été", 7: "Été", 8: "Été",  # Juin, Juil, Août
        9: "Automne", 10: "Automne", 11: "Automne",  # Sep, Oct, Nov
    }
    filtered_monthly_disasters["Saison"] = filtered_monthly_disasters["Mois"].map(saison_map)

    # Mapper les numéros de mois vers leurs noms
    month_names = {
        1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril", 5: "Mai", 6: "Juin",
        7: "Juillet", 8: "Août", 9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
    }
    filtered_monthly_disasters["Nom du mois"] = filtered_monthly_disasters["Mois"].map(month_names)
    filtered_monthly_disasters["Nom du mois"] = pd.Categorical(
        filtered_monthly_disasters["Nom du mois"],
        categories=list(month_names.values()),
        ordered=True
    )

    # Code couleur fixe pour les types de catastrophes
    color_map = {
        "Inondation": "#4000FF",  # Bleu foncé
        "Sécheresse": "#FF9500",  # Jaune
        "Incendie": "#FF0000",    # Rouge
        "Neige": "#ADD8E6",       # Bleu clair
    }

    # Ajout des boutons radio juste au-dessus du graphique pour choisir entre Mois et Saison
    view_option = st.radio(
        "Affichage par :",
        ["Mois", "Saison"],
        index=0,  # Sélectionner par défaut "Mois"
        horizontal=True  # Les boutons seront alignés horizontalement
    )

    # Organiser les graphiques
    col1, col2 = st.columns(2)

    with col1:
        if view_option == "Mois":
            # Agrégation par mois
            monthly_agg = filtered_monthly_disasters.groupby(["Nom du mois", "Type catastrophe"])["Nombre événements"].sum().reset_index()
            fig_monthly = px.bar(
                monthly_agg,
                x="Nom du mois",
                y="Nombre événements",
                color="Type catastrophe",
                title="Répartition des catastrophes par mois",
                labels={"Nombre événements": "Nombre de catastrophes", "Nom du mois": "Mois"},
                color_discrete_map=color_map,
            )
        else:
            # Agrégation par saison
            saison_agg = filtered_monthly_disasters.groupby(["Saison", "Type catastrophe"])["Nombre événements"].sum().reset_index()
            fig_monthly = px.bar(
                saison_agg,
                x="Saison",
                y="Nombre événements",
                color="Type catastrophe",
                title="Répartition des catastrophes par saison",
                labels={"Nombre événements": "Nombre de catastrophes", "Saison": "Saison"},
                color_discrete_map=color_map,
            )
        st.plotly_chart(fig_monthly, use_container_width=True)

    with col2:
        # Graphique 2 : Répartition en pourcentage des types de catastrophes
        disaster_pie = filtered_disasters.groupby("Type catastrophe")["Nombre événements"].sum().reset_index()
        fig_pie = px.pie(
            disaster_pie,
            names="Type catastrophe",
            values="Nombre événements",
            title="Répartition des types de catastrophes",
            color="Type catastrophe",
            color_discrete_map=color_map,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Graphique 3 : Évolution des catastrophes dans le temps par type de catastrophe
    yearly_trend = filtered_disasters.groupby(["Année", "Type catastrophe"])["Nombre événements"].sum().reset_index()
    fig_trend = px.bar(
        yearly_trend,
        x="Année",
        y="Nombre événements",
        color="Type catastrophe",
        title="Évolution du nombre de catastrophes annuelles par type",
        labels={"Nombre événements": "Nombre de catastrophes", "Année": "Année"},
        color_discrete_map=color_map,
    )
    fig_trend.update_xaxes(tickmode="linear")
    st.plotly_chart(fig_trend, use_container_width=True)


# ================================================
# =      ONGLET 4 : ÉTUDE DE CAS (2 SOUS-TABS)   =
# ================================================
with onglets[3]:
    st.header("Étude de cas")
    sous_onglets = st.tabs(["Changement climatique global", "Fortes températures en Corse"])

    # ==========================================================
    # SOUS-ONGLET 1 : Changement climatique global (existant)
    # ==========================================================
    with sous_onglets[0]:
        file_path = "lignes_points_proches10.csv"  # Remplacez par le chemin de votre fichier
        data = pd.read_csv(file_path)

        # Vérification des colonnes nécessaires
        if 'Annee' not in data.columns or 'PRENEI_MENS' not in data.columns or 'T_MENS' not in data.columns or 'PRELIQ_MENS' not in data.columns:
            st.error("Le fichier doit contenir les colonnes 'Annee', 'PRENEI_MENS', 'T_MENS' et 'PRELIQ_MENS'.")
        else:
            # Sélecteur pour choisir le paramètre, avec PRENEI_MENS comme valeur par défaut
            parametre = st.selectbox("Sélectionnez un paramètre", ["T_MENS", "PRENEI_MENS", "PRELIQ_MENS"], index=1)

            # Calculer l'évolution par année pour le paramètre sélectionné
            evolution_annee = data.groupby('Annee')[parametre].mean().reset_index()

            # Préparation des données pour la régression
            X = evolution_annee['Annee'].values.reshape(-1, 1)
            y = evolution_annee[parametre].values

            if len(X) > 0:
                # Transformation polynomiale des données (degré 3)
                poly = PolynomialFeatures(degree=3)
                X_poly = poly.fit_transform(X)

                # Création et entraînement du modèle de régression polynomiale
                model = LinearRegression()
                model.fit(X_poly, y)

                # Prédictions de la régression
                y_poly_pred = model.predict(X_poly)

                # Premier graphique
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=evolution_annee['Annee'],
                    y=evolution_annee[parametre],
                    mode='lines+markers',
                    name="",
                    line=dict(color='blue'),
                    marker=dict(size=6)
                ))
                fig1.add_trace(go.Scatter(
                    x=evolution_annee['Annee'],
                    y=y_poly_pred,
                    mode='lines',
                    name="",
                    line=dict(color='red', width=2)
                ))
                fig1.update_layout(
                    title=f"Évolution de {parametre} par année (régression polynomiale)",
                    xaxis_title="Année",
                    yaxis_title=f"{parametre} moyen",
                    template="plotly_white",
                    showlegend=False
                )
            else:
                st.warning("Pas de données disponibles pour la régression polynomiale.")
                fig1 = go.Figure()

            # Filtrer sur les mois Août à Juillet
            selected_months = list(range(8, 13)) + list(range(1, 8))
            filtered_data = data[data['Mois'].isin(selected_months)]

            # Ajouter une colonne 'Saison' : Août = 8 (reste 8 à 12), Janvier=1 => on décalle
            filtered_data['Saison'] = filtered_data['Mois'].apply(lambda x: x + 12 if x <= 7 else x)

            # Filtrer sur 1960-1970 et 2010-2020
            data_1960_1970 = filtered_data[(filtered_data['Annee'] >= 1960) & (filtered_data['Annee'] <= 1970)]
            data_2010_2020 = filtered_data[(filtered_data['Annee'] >= 2010) & (filtered_data['Annee'] <= 2020)]

            mean_1960_1970 = data_1960_1970.groupby('Saison')[parametre].mean().reset_index()
            mean_2010_2020 = data_2010_2020.groupby('Saison')[parametre].mean().reset_index()

            mean_1960_1970['Période'] = "1960-1970"
            mean_2010_2020['Période'] = "2010-2020"

            mean_combined = pd.concat([mean_1960_1970, mean_2010_2020])

            # Deuxième graphique
            fig2 = px.line(
                mean_combined,
                x="Saison",
                y=parametre,
                color="Période",
                labels={"Saison": "Mois (Août à Juillet)", parametre: f"{parametre} moyen"},
                title=f"Évolution de {parametre} moyen (1960-1970 vs 2010-2020)"
            )
            fig2.update_xaxes(
                tickmode='array',
                tickvals=list(range(8, 20)),
                ticktext=["Août", "Sept", "Oct", "Nov", "Déc", "Jan", "Fév", "Mars", "Avr", "Mai", "Juin", "Juil"]
            )

        # Troisième graphique : Évolution de l'humidité moyenne
        file_path_points = "lignes_points_proches4.csv"
        file_path_national = "extracted_national_data.csv"

        local_data = pd.read_csv(file_path_points)
        local_data = local_data[local_data['Mois'].isin([3, 4])]

        local_humidity = local_data.groupby('Annee')['SWI_MENS'].mean().reset_index()
        local_humidity.rename(columns={'SWI_MENS': 'Humidité_Sarcenas'}, inplace=True)

        # Charger les données nationales
        national_humidity = pd.read_csv(file_path_national)

        combined_data = pd.merge(local_humidity, national_humidity, on='Annee', how='outer').sort_values(by='Annee')
        combined_data['Moyenne_Mobile_Sarcenas'] = combined_data['Humidité_Sarcenas'].rolling(window=5, min_periods=1).mean()
        combined_data['Moyenne_Mobile_France'] = combined_data['Humidité_France'].rolling(window=5, min_periods=1).mean()

        X_sarcenas = combined_data['Annee'].values.reshape(-1, 1)
        y_sarcenas = combined_data['Moyenne_Mobile_Sarcenas'].values
        model_sarcenas = LinearRegression()
        model_sarcenas.fit(X_sarcenas, y_sarcenas)
        y_sarcenas_pred = model_sarcenas.predict(X_sarcenas)

        X_france = combined_data['Annee'].values.reshape(-1, 1)
        y_france = combined_data['Moyenne_Mobile_France'].values
        model_france = LinearRegression()
        model_france.fit(X_france, y_france)
        y_france_pred = model_france.predict(X_france)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=combined_data['Annee'],
            y=combined_data['Moyenne_Mobile_Sarcenas'],
            mode='lines',
            name='Moyenne Mobile Sarcenas',
            line=dict(color='blue')
        ))
        fig3.add_trace(go.Scatter(
            x=combined_data['Annee'],
            y=y_sarcenas_pred,
            mode='lines',
            name="",
            line=dict(color='blue', width=2)
        ))
        fig3.add_trace(go.Scatter(
            x=combined_data['Annee'],
            y=combined_data['Moyenne_Mobile_France'],
            mode='lines',
            name='Moyenne Mobile France',
            line=dict(color='red')
        ))
        fig3.add_trace(go.Scatter(
            x=combined_data['Annee'],
            y=y_france_pred,
            mode='lines',
            name="",
            line=dict(color='red', width=2)
        ))
        fig3.update_layout(
            title="Évolution de l'humidité moyenne (mars et avril)",
            xaxis_title="Année",
            yaxis_title="Humidité moyenne (SWI)",
            template="plotly_white"
        )
        # Enlever la légende supplémentaire
        for trace in fig3.data:
            if trace.name == "":
                trace.showlegend = False

        # Quatrième graphique : précipitations liquides/solides
        file_path_points_precip = "lignes_points_proches10.csv"
        local_data_precip = pd.read_csv(file_path_points_precip)

        # Vérif colonnes
        required_columns = ['Annee', 'Mois', 'PRELIQ_MENS', 'PRENEI_MENS']
        if not all(col in local_data_precip.columns for col in required_columns):
            st.error(f"Le fichier {file_path_points_precip} doit contenir {required_columns}.")
        else:
            # Déc, Jan, Fév, Mars
            selected_months = [12, 1, 2, 3]
            local_data_precip = local_data_precip[local_data_precip['Mois'].isin(selected_months)]
            local_data_precip = local_data_precip[
                (local_data_precip['Annee'] >= 1960) & 
                (local_data_precip['Annee'] <= 2023)
            ]

            # Moyennes
            liquid_precipitation = (
                local_data_precip
                .groupby('Annee')['PRELIQ_MENS']
                .mean()
                .reset_index()
            )
            liquid_precipitation.rename(columns={'PRELIQ_MENS': 'Précipitations_Liquides'}, inplace=True)

            solid_precipitation = (
                local_data_precip
                .groupby('Annee')['PRENEI_MENS']
                .mean()
                .reset_index()
            )
            solid_precipitation.rename(columns={'PRENEI_MENS': 'Précipitations_Solides'}, inplace=True)

            combined_precipitation = pd.merge(liquid_precipitation, solid_precipitation, on='Annee')

            # Moyennes mobiles (3 ans)
            combined_precipitation['Moyenne_Mobile_Liquides'] = (
                combined_precipitation['Précipitations_Liquides'].rolling(window=3, min_periods=1).mean()
            )
            combined_precipitation['Moyenne_Mobile_Solides'] = (
                combined_precipitation['Précipitations_Solides'].rolling(window=3, min_periods=1).mean()
            )

            # Régressions
            X = combined_precipitation['Annee'].values.reshape(-1, 1)
            y_liquid = combined_precipitation['Moyenne_Mobile_Liquides'].fillna(0).values
            model_liquid = LinearRegression()
            model_liquid.fit(X, y_liquid)
            combined_precipitation['Regression_Liquides'] = model_liquid.predict(X)

            y_solid = combined_precipitation['Moyenne_Mobile_Solides'].fillna(0).values
            model_solid = LinearRegression()
            model_solid.fit(X, y_solid)
            combined_precipitation['Regression_Solides'] = model_solid.predict(X)

            # Tracé
            fig4 = px.line(
                combined_precipitation,
                x="Annee",
                y=[
                    "Moyenne_Mobile_Liquides",
                    "Moyenne_Mobile_Solides",
                    "Regression_Liquides",
                    "Regression_Solides",
                ],
                labels={
                    "Annee": "Année",
                    "value": "Précipitations moyennes (mm)",
                    "variable": "Type de précipitations"
                },
                title="Évolution des précipitations liquides et solides (Décembre à Mars)"
            )
            fig4.update_layout(
                template="plotly_white",
                xaxis_title="Année",
                yaxis_title="Précipitations moyennes (mm)",
                title_font_size=16
            )

            # Supprimer les noms de trace "Regression_..."
            for trace in fig4.data:
                if "Regression" in trace.name:
                    trace.name = ""

        # Disposition des 4 graphiques
        col_a, col_b = st.columns([1, 1], gap="small")
        with col_a:
            st.plotly_chart(fig1, use_container_width=True, key="fig1")
            st.plotly_chart(fig3, use_container_width=True, key="fig3")
        with col_b:
            st.plotly_chart(fig2, use_container_width=True, key="fig2")
            st.plotly_chart(fig4, use_container_width=True, key="fig4")
            # Le quatrième graphique est omis ici afin qu'il n'apparaisse pas dans ce sous-onglet.

    # ==========================================================
    # SOUS-ONGLET 2 : Fortes températures en Corse (Nouveau)
    # ==========================================================
    with sous_onglets[1]:
        df_corse_ete = pd.read_csv("etude_de_cas_corse_ete.csv")
        variable_y = st.selectbox("Sélectionnez une variable",
        options=["T_MENS", "PRELIQ_MENS", "SWI_MENS"],
        index=0  # T_MENS par défaut
        )

        # Variables des catastrophes naturelles
        nombre_secheresse = df_corse_ete.groupby('Annee')['Sécheresse'].sum() 
        nombre_incendie = df_corse_ete.groupby('Annee')['Incendie'].sum()

        # Variable de la température
        moy_an_temp = df_corse_ete.groupby('Annee')['T_MENS'].mean()

        # Création de la figure
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=nombre_incendie.index, 
                y=nombre_incendie.values,
                name="Nombre d'incendies",
                marker=dict(color='red'),
                yaxis='y2',
                zorder=2
            )
        )

        # Ajouter les barres pour les sécheresses avec un décalage
        fig.add_trace(
            go.Bar(
                x=nombre_secheresse.index,
                y=nombre_secheresse.values,
                name="Nombre de sécheresses",
                marker=dict(color='orange'),
                yaxis='y3'
            )
        )

        # Ajouter la trace pour les températures 
        fig.add_trace(
            go.Scatter(
                x=moy_an_temp.index,
                y=moy_an_temp.values,
                mode='lines+markers',
                name='Températures',
                marker=dict(color='blue'),
                line=dict(color='blue'),
                yaxis='y1',
                zorder=3
            )
        )

        fig.update_layout(
            title= variable_y + " moyennes et nombre de sécheresses et incendies sur l'été",
            xaxis=dict(title="Année"),
            yaxis=dict(
                title= variable_y,
                titlefont=dict(color='blue'),
                tickfont=dict(color='blue'),
                side='left'
            ),
            yaxis2=dict(
                title="Nombre d'incendies",
                titlefont=dict(color='red'),
                tickfont=dict(color='red'),
                overlaying='y',
                side='right',
                position=0.99
            ),
            yaxis3=dict(
                title="Nombre de sécheresses",
                titlefont=dict(color='orange'),
                tickfont=dict(color='orange'),
                anchor='free',
                overlaying='y',
                side='right',
                position=1
            ),
            barmode='overlay',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        # Désactivation de la grille des axes pour éviter les traits horizontaux
        fig.update_layout(
            yaxis=dict(showgrid=False),
            yaxis2=dict(showgrid=False),
            yaxis3=dict(showgrid=False)
        )
        st.plotly_chart(fig, use_container_width=True)
