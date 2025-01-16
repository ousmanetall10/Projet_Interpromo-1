import pandas as pd
import streamlit as st
import plotly.express as px
import geopandas as gpd
import numpy as np

# Charger les données pour le premier dashboard
def load_data_main():
    return pd.read_csv("data_ville_annuel.csv")
    
# Charger les données pour le second dashboard
def load_data_disasters():
    return pd.read_csv("catastrophes_naturelles_annuelles_dep_incendies.csv")

# Charger les géométries des départements français
def load_geometries():
    url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson"
    return gpd.read_file(url)

# Charger les données pour le troisième dashboard
def load_data_monthly_disasters():
    return pd.read_csv("catastrophes_naturelles_mensuelles_dep_incendies.csv")

# Charger les données mensuelles
def load_data_monthly():
    return pd.read_csv("data_ville_mensuel.csv")


# Charger les données
main_data = load_data_main()
disasters_data = load_data_disasters()
geometries = load_geometries()
monthly_disasters_data = load_data_monthly_disasters()
monthly_disasters_data['Département'] = monthly_disasters_data['Département'].astype(str).str.zfill(2)
monthly_data = load_data_monthly()

# Préparation des données
main_data["T_MENS"] = main_data["T_MENS"].apply(lambda x: max(x, 9))  # Ajustement spécifique

disasters_data["Département"] = disasters_data["Département"].apply(lambda x: str(x).zfill(2))

# Calcul des min et max globaux pour la première carte
param_cols = ["T_MENS", "PRENEI_MENS", "PRELIQ_MENS", "PRETOTM_MENS", "EVAP_MENS", "ETP_MENS", "PE_MENS", "SWI_MENS", "DRAINC_MENS", "RUNC_MENS", "ECOULEMENT_MENS"]
global_min_max_main = {
    param: (
        main_data[param].min() - 0.1 * abs(main_data[param].min()),
        main_data[param].max() - 0.1 * abs(main_data[param].max())
    ) for param in param_cols
}

# Calcul des min et max globaux pour la deuxième carte (logarithmiques inclus)
disasters_data["Nombre_log"] = np.log1p(disasters_data["Nombre événements"])
global_min_disasters = disasters_data["Nombre_log"].min()
global_max_disasters = disasters_data["Nombre_log"].max()
# Calcul des min et max globaux pour chaque type de catastrophe
global_min_max_disasters_by_type = disasters_data.groupby("Type catastrophe")["Nombre_log"].agg(["min", "max"]).reset_index()
global_min_max_disasters_by_type.rename(columns={"min": "global_min", "max": "global_max"}, inplace=True)

# Streamlit App
st.set_page_config(layout="wide")

# Onglets pour les visualisations
onglets = st.tabs(["Cartes", "Climat", "Catastrophes"])

with onglets[0]:
    # Curseur pour sélectionner une année (uniquement dans l'onglet Cartes)
    min_year = int(min(main_data["Annee"].min(), disasters_data["Année"].min()))
    max_year = int(max(main_data["Annee"].max(), disasters_data["Année"].max()))
    selected_year = st.slider("Sélectionnez une année", min_value=min_year, max_value=max_year, value=min_year, step=1)

    # Filtrage des données pour l'année sélectionnée
    filtered_main_data = main_data[main_data["Annee"] == selected_year]
    filtered_disasters_data = disasters_data[disasters_data["Année"] == selected_year]

    # Vérifier les données disponibles et gérer les cas séparément
    if filtered_main_data.empty:
        st.warning(f"Aucune donnée météorologique disponible pour l'année {selected_year}.")

    if filtered_disasters_data.empty:
        st.warning(f"Aucune donnée de catastrophes disponible pour l'année {selected_year}.")

    # Carte 1 : Carte météorologique
    col1, col2 = st.columns([1, 1])

    with col1:
        param_main = st.selectbox("Sélectionnez un paramètre météo", 
                                  options=param_cols,
                                  index=0)
        
        min_val_main, max_val_main = global_min_max_main[param_main]
        
        fig_main = px.scatter_mapbox(
            filtered_main_data,
            lat="latitude",
            lon="longitude",
            color=param_main,
            size=param_main,
            size_max=5,  # Contrôle de la taille maximale des points
            color_continuous_scale="jet",
            range_color=[min_val_main, max_val_main],
            opacity=0.7,
            mapbox_style="carto-positron",
            zoom=4,
            center={"lat": 46.603354, "lon": 2.469758},
            hover_name="Departement", # Optionnel : pour afficher le nom de la département au survol
            hover_data={"latitude": True, "longitude": True, param_main: True},
        )

        fig_main.update_layout(
            height=500,
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )

        st.plotly_chart(fig_main, use_container_width=True)

    # Carte 2 : Carte des catastrophes
    with col2:
        disaster_type = st.selectbox(
            "Sélectionnez un type de catastrophe", 
            options=disasters_data["Type catastrophe"].unique(), 
            index=0
        )
    
        filtered_disasters_by_type = filtered_disasters_data[filtered_disasters_data["Type catastrophe"] == disaster_type]
    
        # Récupération des min et max globaux pour ce type de catastrophe
        scale_row = global_min_max_disasters_by_type[global_min_max_disasters_by_type["Type catastrophe"] == disaster_type]
        global_min_disaster_type = scale_row["global_min"].values[0] if not scale_row.empty else 0
        global_max_disaster_type = scale_row["global_max"].values[0] if not scale_row.empty else 1

        aggregated_disasters = filtered_disasters_by_type.groupby("Département")["Nombre événements"].sum().reset_index()
        aggregated_disasters["Nombre_log"] = np.log1p(aggregated_disasters["Nombre événements"])
        aggregated_disasters.rename(columns={"Département": "code"}, inplace=True)

        geo_merged = geometries.merge(aggregated_disasters, left_on="code", right_on="code", how="left")
        geo_merged["Nombre_log"] = geo_merged["Nombre_log"].fillna(0)
        geo_merged["Nombre événements"] = geo_merged["Nombre événements"].fillna(0)



        # Ajout d'un calque gris si les données sont absentes
        if filtered_disasters_by_type.empty:
            # Remplir toutes les régions en gris clair
            geo_merged["Nombre_log"] = 0
            fig_disasters = px.choropleth_mapbox(
                geo_merged,
                geojson=geo_merged.geometry,
                locations=geo_merged.index,
                color="Nombre_log",
                color_continuous_scale=["#D3D3D3", "#D3D3D3"],  # Gris clair
                range_color=(0, 1),  # Échelle arbitraire pour garder les teintes uniformes
                mapbox_style="carto-positron",
                zoom=4,
                center={"lat": 46.603354, "lon": 1.888334},
                labels={"Nombre_log": "Nbre catastrophes (log)"},
                hover_name="nom", # Optionnel : pour afficher le nom de la département au survol
                hover_data={"Nombre_log": False},  # Pas d'informations sur le survol
            )
            # Ajouter un texte centré au milieu de la carte
            fig_disasters.add_annotation(
                x=0.5, y=0.5, xref="paper", yref="paper",  # Positionnement au centre de la carte
                text="<b>Aucune donnée disponible pour cette année</b>",
                showarrow=False,
                font=dict(size=18, color="black"),
                align="center",
                bgcolor="rgba(255, 255, 255, 0.7)",  # Fond légèrement transparent pour le texte
                bordercolor="black",
                borderwidth=1,
            )
        else:
            # Code existant pour tracer la carte avec des données
            fig_disasters = px.choropleth_mapbox(
                geo_merged,
                geojson=geo_merged.geometry,
                locations=geo_merged.index,
                color="Nombre_log",
                color_continuous_scale="Reds",
                range_color=(global_min_disaster_type, global_max_disaster_type),  # Échelle adaptée au type
                mapbox_style="carto-positron",
                zoom=4,
                center={"lat": 46.603354, "lon": 1.888334},
                labels={"Nombre_log": "Nbre catastrophes (log)"},
                hover_name="nom", # Optionnel : pour afficher le nom de la département au survol
                hover_data={"Nombre_log": True, "Nombre événements": True},
            )

        fig_disasters.update_layout(height=500, margin={"r": 0, "t": 0, "l": 0, "b": 0})

        # Affichage de la carte dans la colonne 2
        st.plotly_chart(fig_disasters, use_container_width=True)
    # Onglet Climat
    with onglets[1]:
        # Sélection des départements à afficher (option multiple avec "Toute la France" par défaut)
        departements_selectionnes = st.multiselect(
            "Sélectionnez des départements (ou 'Toute la France' pour les moyennes globales)",
            options=["Toute la France"] + list(main_data["Departement"].unique()),
            default=["Toute la France"]
        )
    
        # Mode d'affichage des courbes : individuelle ou moyenne
        mode_affichage = st.radio(
            "Mode d'affichage des courbes",
            options=["Courbes individuelles", "Courbe moyenne"],
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
    
        # Sélection des mois via des cases à cocher
        mois_selectionnes = st.multiselect(
            "Sélectionnez les mois à afficher (tous cochés pour moyenne annuelle)",
            options=["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"],
            default=["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]  # Par défaut tous les mois sont sélectionnés
        )
    
        for i, parametre in enumerate(parametres_graphique):
            with col_graphs[i % 2]:  # Alterne entre la colonne 0 et 1
                if mode_affichage == "Courbe moyenne" or "Toute la France" in departements_selectionnes:
                    # Moyenne nationale ou moyenne des départements sélectionnés
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
    
                # Si l'utilisateur choisit de se concentrer sur les mois, filtrer les données mensuelles
                if mois_selectionnes:
                    # Mapper les mois sélectionnés
                    mois_map = {
                        "Janvier": 1, "Février": 2, "Mars": 3, "Avril": 4, "Mai": 5, "Juin": 6,
                        "Juillet": 7, "Août": 8, "Septembre": 9, "Octobre": 10, "Novembre": 11, "Décembre": 12
                    }
                    mois_num_selectionnes = [mois_map[m] for m in mois_selectionnes]
                
                    # Filtrer les données mensuelles par les mois sélectionnés
                    monthly_data_filtered = monthly_data[(monthly_data["Annee"] >= start_year) & (monthly_data["Annee"] <= end_year)]
                    monthly_data_filtered = monthly_data_filtered[monthly_data_filtered["Mois"].isin(mois_num_selectionnes)]
                
                    # Calcul des moyennes annuelles pour les départements et paramètres
                    if "Toute la France" in departements_selectionnes:
                        # Calcul de la moyenne des mois sélectionnés pour chaque année
                        moyenne_mensuelle = monthly_data_filtered.groupby("Annee")[parametre].mean().reset_index()
                        moyenne_mensuelle["Type"] = "Moyenne France"
                    else:
                        data_filtrée_mensuelle = monthly_data_filtered[monthly_data_filtered["Departement"].isin(departements_selectionnes)]
                        # Calcul de la moyenne des mois sélectionnés pour chaque année
                        moyenne_mensuelle = data_filtrée_mensuelle.groupby("Annee")[parametre].mean().reset_index()
                        moyenne_mensuelle["Type"] = "Dept choisis"
                
                    data_to_plot = moyenne_mensuelle

    
                # Création du graphique avec nuage de points, régression et moyenne mobile
                fig_graph = px.scatter(
                    data_to_plot,
                    x="Annee",
                    y=parametre,
                    color="Type",
                    trendline="lowess",  # Ajoute une courbe de régression (type LOWESS)
                    title=f"Tendance de {parametre} ({start_year}-{end_year})",
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
    filtered_disasters = disasters_data[
        (disasters_data["Année"] >= start_year) &
        (disasters_data["Année"] <= end_year)
    ]
    filtered_monthly_disasters = monthly_disasters_data[
        (monthly_disasters_data["Année"] >= start_year) &
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
        "Inondation": "#00008B",  # Bleu foncé
        "Sécheresse": "#FFD700",  # Jaune
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

    # Graphique 3 : Évolution des catastrophes dans le temps
    yearly_trend = filtered_disasters.groupby(["Année"])["Nombre événements"].sum().reset_index()
    fig_trend = px.bar(
        yearly_trend,
        x="Année",
        y="Nombre événements",
        title="Évolution du nombre de catastrophes annuelles",
        labels={"Nombre événements": "Nombre de catastrophes", "Année": "Année"},
    )
    fig_trend.update_xaxes(tickmode="linear")
    st.plotly_chart(fig_trend, use_container_width=True)