"""
SLOTTY - Application Web Interactive Yield Management
Avec graphiques et calculs en temps réel
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io

# ============================================================
# PROTECTION PAR MOT DE PASSE
# ============================================================

def check_password():
    """
    Gère l'authentification par mot de passe.
    
    Returns:
        bool: True si authentifié, False sinon
    """
    # Essayer de récupérer le mot de passe depuis les secrets Streamlit Cloud
    # Si pas de secrets (local), utiliser un mot de passe par défaut
    try:
        correct_password = st.secrets.get("password", "slotty2024")
    except:
        # En local, mot de passe par défaut
        correct_password = "slotty2024"
    
    # Vérifier si déjà authentifié dans la session
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # Si pas encore authentifié, afficher l'écran de connexion
    if not st.session_state.authenticated:
        # Afficher l'écran de connexion
        st.markdown("""
        <style>
        .login-container {
            max-width: 400px;
            margin: 100px auto;
            padding: 40px;
            background-color: #f0f2f6;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.title("🔐 Slotty - Accès protégé")
        st.markdown("### Veuillez vous connecter")
        
        # Champ de mot de passe
        password = st.text_input(
            "Mot de passe :",
            type="password",
            placeholder="Entrez votre mot de passe",
            key="password_input"
        )
        
        # Bouton de connexion
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            login_button = st.button("🔓 Se connecter", use_container_width=True)
        
        # Vérifier le mot de passe
        if login_button:
            if password == correct_password:
                st.session_state.authenticated = True
                st.success("✅ Connexion réussie !")
                st.rerun()
            else:
                st.error("❌ Mot de passe incorrect")
        
        # Info sur le mot de passe par défaut (à retirer en production)
        with st.expander("ℹ️ Informations de connexion"):
            st.info("""
            **Mot de passe par défaut (local) :** `slotty2024`
            
            Pour changer le mot de passe sur Streamlit Cloud :
            1. Va dans Settings de ton app
            2. Secrets → Ajoute `password = "ton_nouveau_mdp"`
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Arrêter l'exécution ici si pas authentifié
        st.stop()
    
    return True

# Vérifier l'authentification avant de charger l'app
check_password()

# ============================================================
# CONSTANTES - Extraction des valeurs magiques (FIX PRIORITÉ 2)
# ============================================================
TAUX_ULTRA_PROMO = 15  # En dessous de ce taux, réduction maximale appliquée
JOUEURS_PAR_TERRAIN = 4  # Nombre de joueurs par créneau (padel = 4 joueurs)
SEMAINES_PAR_MOIS = 4.33  # Nombre moyen de semaines par mois (52/12)
JOURS_ORDRE_FR = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
JOURS_ORDRE_EN = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Mapping anglais -> français
JOURS_EN_TO_FR = {
    'Monday': 'Lundi',
    'Tuesday': 'Mardi',
    'Wednesday': 'Mercredi',
    'Thursday': 'Jeudi',
    'Friday': 'Vendredi',
    'Saturday': 'Samedi',
    'Sunday': 'Dimanche'
}

# ============================================================
# FONCTIONS UTILITAIRES (FIX PRIORITÉ 2 - Réduction duplication)
# ============================================================

def detecter_et_convertir_jours(df, colonne_jour='jour_semaine'):
    """
    Détecte la langue des jours (FR/EN) et convertit en français si nécessaire.
    
    Args:
        df (DataFrame): DataFrame contenant la colonne des jours
        colonne_jour (str): Nom de la colonne contenant les jours
    
    Returns:
        DataFrame: DataFrame avec les jours en français
        str: Langue détectée ('FR' ou 'EN')
    """
    # Prendre le premier jour pour détecter la langue
    premier_jour = df[colonne_jour].iloc[0]
    
    # Détecter la langue
    if premier_jour in JOURS_ORDRE_EN:
        langue = 'EN'
        # Convertir en français
        df = df.copy()
        df[colonne_jour] = df[colonne_jour].map(JOURS_EN_TO_FR)
    elif premier_jour in JOURS_ORDRE_FR:
        langue = 'FR'
        # Déjà en français, ne rien faire
    else:
        # Essayer de normaliser la casse
        premier_jour_title = premier_jour.capitalize()
        if premier_jour_title in JOURS_ORDRE_EN:
            langue = 'EN'
            df = df.copy()
            df[colonne_jour] = df[colonne_jour].str.capitalize().map(JOURS_EN_TO_FR)
        elif premier_jour_title in JOURS_ORDRE_FR:
            langue = 'FR'
            df = df.copy()
            df[colonne_jour] = df[colonne_jour].str.capitalize()
        else:
            langue = 'UNKNOWN'
    
    return df, langue


def create_heatmap(pivot_data, title, colorscale, hover_text=None, text_suffix="%", 
                   zmin=0, zmax=100, colorbar_title="Valeur", show_text=True):
    """
    Crée un heatmap Plotly standardisé pour éviter la duplication de code.
    
    Args:
        pivot_data (DataFrame): DataFrame pivot avec les données
        title (str): Titre du graphique
        colorscale (str): Échelle de couleurs Plotly ('RdYlGn_r', 'RdYlGn', etc.)
        hover_text (list, optional): Texte au survol. Auto-généré si None.
        text_suffix (str): Suffixe pour le texte affiché (%, €, pts, etc.)
        zmin, zmax (float): Limites min/max de l'échelle de couleurs
        colorbar_title (str): Titre de la barre de couleurs
        show_text (bool): Afficher le texte dans les cases
    
    Returns:
        go.Figure: Figure Plotly configurée
    """
    # Gérer les NaN pour l'affichage (remplacer par 0)
    pivot_display = pivot_data.fillna(0)
    
    # Créer le texte d'affichage
    if show_text:
        text_display = []
        for i in range(len(pivot_data.index)):
            row_text = []
            for j in range(len(pivot_data.columns)):
                val = pivot_data.iloc[i, j]
                if pd.notna(val) and val > 0:
                    row_text.append(f"{val:.0f}")
                else:
                    row_text.append("")
            text_display.append(row_text)
    else:
        text_display = None
    
    # Créer hover text si non fourni
    if hover_text is None:
        hover_text = []
        for i in range(len(pivot_data.index)):
            row_hover = []
            for j in range(len(pivot_data.columns)):
                val = pivot_data.iloc[i, j]
                if pd.notna(val) and val > 0:
                    row_hover.append(
                        f"{pivot_data.index[i]}<br>"
                        f"{pivot_data.columns[j]}<br>"
                        f"Valeur: {val:.1f}{text_suffix}"
                    )
                else:
                    row_hover.append("Pas de données")
            hover_text.append(row_hover)
    
    # Créer la figure
    fig = go.Figure(data=go.Heatmap(
        z=pivot_display.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale=colorscale,
        text=text_display,
        texttemplate=f'%{{text}}{text_suffix}' if show_text else None,
        textfont={"size": 10, "color": "white"},
        hovertext=hover_text,
        hoverinfo='text',
        colorbar=dict(title=colorbar_title),
        zmin=zmin,
        zmax=zmax,
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        height=500,
        xaxis_title="Heure",
        yaxis_title="Jour",
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


def verifier_merge_donnees(df_merged, nom_colonne_test='taux'):
    """
    Vérifie si un merge pandas a réussi.
    
    Args:
        df_merged (DataFrame): DataFrame résultat du merge
        nom_colonne_test (str): Colonne à tester pour vérifier le merge
    
    Returns:
        bool: True si le merge a réussi, False sinon
    """
    if nom_colonne_test not in df_merged.columns:
        return False
    return not df_merged[nom_colonne_test].isna().all()


# Configuration de la page
st.set_page_config(
    page_title="Slotty - Yield Management",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Titre
st.title("🎾 Slotty - Yield Management Interactif")
st.markdown("---")

# Sidebar - Paramètres
st.sidebar.header("⚙️ Paramètres")

# Upload fichier
uploaded_file = st.sidebar.file_uploader("📁 Charger fichier CSV", type=['csv'])

if uploaded_file is not None:
    # Charger les données
    df = pd.read_csv(uploaded_file)
    
    # Ajouter colonnes nécessaires
    df['date'] = pd.to_datetime(df['date'])
    df['jour_semaine'] = df['date'].dt.day_name()
    df['jour_semaine_num'] = df['date'].dt.dayofweek
    df['semaine'] = df['date'].dt.isocalendar().week
    
    # CONVERSION AUTOMATIQUE ANGLAIS -> FRANÇAIS (FIX bug heatmaps vides)
    df, langue_detectee = detecter_et_convertir_jours(df, 'jour_semaine')
    
    st.sidebar.success(f"✅ Fichier chargé : {len(df)} lignes")
    if langue_detectee == 'EN':
        st.sidebar.info("🌐 Jours convertis de l'anglais vers le français")
    elif langue_detectee == 'FR':
        st.sidebar.info("🇫🇷 Jours détectés en français")
    
    # Paramètres de pricing
    st.sidebar.markdown("### 🎯 Paramètres de Pricing")
    
    seuil_remplissage = st.sidebar.slider(
        "Seuil de remplissage pour promo (%)",
        min_value=0,
        max_value=100,
        value=40,
        step=5,
        help="Créneaux avec < X% de remplissage auront des prix promo"
    )
    
    prix_plancher = st.sidebar.slider(
        "Prix plancher (€/joueur)",
        min_value=3.0,
        max_value=12.0,
        value=6.0,
        step=0.5,
        help="Prix minimum en promotion"
    )
    
    st.sidebar.markdown("---")
    
    # Info sur les données
    st.sidebar.markdown("### 📊 Informations")
    st.sidebar.metric("Terrains", df['terrain'].nunique())
    st.sidebar.metric("Période", f"{df['date'].min().date()} au {df['date'].max().date()}")
    st.sidebar.metric("Taux de remplissage global", f"{(df['statut']=='réservé').sum() / len(df) * 100:.1f}%")
    
    # Fonction de calcul du prix dynamique (FIX PRIORITÉ 2: Docstring ajoutée)
    def get_prix_dynamique(taux, prix_base):
        """
        Calcule le prix dynamique selon le taux de remplissage.
        
        Logique de pricing :
        - Si taux >= seuil : prix normal (pas de promo)
        - Si taux <= 15% : prix plancher (promo maximale)
        - Entre les deux : interpolation linéaire
        
        Args:
            taux (float): Taux de remplissage en % (0-100)
            prix_base (float): Prix normal sans promotion en €
        
        Returns:
            float: Prix dynamique calculé en €
        
        Examples:
            >>> get_prix_dynamique(10, 12)  # Très vide
            6.0  # Prix plancher
            >>> get_prix_dynamique(80, 12)  # Bien rempli
            12.0  # Prix normal
            >>> get_prix_dynamique(27.5, 12)  # Moyennement rempli
            9.0  # Prix intermédiaire
        """
        if taux >= seuil_remplissage:
            return prix_base
        if taux <= TAUX_ULTRA_PROMO:
            return prix_plancher
        # Protection contre division par zéro (ajoutée FIX Priorité 1)
        if seuil_remplissage <= TAUX_ULTRA_PROMO:
            return prix_plancher
        # Interpolation linéaire entre prix_plancher et prix_base
        ratio = (taux - TAUX_ULTRA_PROMO) / (seuil_remplissage - TAUX_ULTRA_PROMO)
        return prix_plancher + (prix_base - prix_plancher) * ratio
    
    # ============================================================
    # CALCUL DU TAUX DE REMPLISSAGE PAR CRÉNEAU
    # ============================================================
    # Agrégation par jour de la semaine et heure de début
    # pour obtenir un taux moyen sur toute la période
    remplissage = df.groupby(['jour_semaine', 'heure_debut']).agg({
        'statut': lambda x: (x == 'réservé').sum() / len(x) * 100,  # % de créneaux réservés
        'prix_par_joueur': 'first'  # Prix de base (supposé constant par créneau)
    }).reset_index()
    remplissage.columns = ['jour', 'heure', 'taux', 'prix_base']
    
    # ============================================================
    # CALCUL DES PRIX DYNAMIQUES
    # ============================================================
    # Applique la fonction de pricing à chaque combinaison jour/heure
    remplissage['prix_dynamique'] = remplissage.apply(
        lambda row: get_prix_dynamique(row['taux'], row['prix_base']), 
        axis=1
    )
    # Calcule le % de réduction par rapport au prix normal
    remplissage['reduction'] = ((remplissage['prix_base'] - remplissage['prix_dynamique']) / 
                                remplissage['prix_base'] * 100)
    
    # Créneaux éligibles aux promos
    creneaux_promo = remplissage[remplissage['taux'] < seuil_remplissage].copy()
    
    # VÉRIFICATION CRITIQUE : S'assurer qu'on a des données
    if len(remplissage) == 0:
        st.error("❌ Impossible de calculer les taux de remplissage")
        st.info("Vérifiez que votre CSV contient les colonnes requises : date, heure_debut, terrain, prix_par_joueur, statut")
        st.stop()
    
    # ============================================================
    # ONGLETS
    # ============================================================
    
    tab0, tab1, tab2, tab3, tab_prix, tab4 = st.tabs([
        "🏟️ Terrains", 
        "📊 Vue d'ensemble", 
        "💰 Revenues", 
        "📅 Impact Yield", 
        "💵 Grille Prix",
        "📈 Détails"
    ])
    
    # ============================================================
    # TAB 0 : VUE D'ENSEMBLE TERRAINS
    # ============================================================
    with tab0:
        st.header("🏟️ Vue d'ensemble des terrains")
        st.markdown("### Situation actuelle (AVANT application des promos)")
        
        # Filtre Semaine / Weekend
        col_filtre1, col_filtre2 = st.columns([1, 3])
        with col_filtre1:
            filtre_jour = st.selectbox(
                "Période",
                ["Tous", "Semaine (Lun-Ven)", "Weekend (Sam-Dim)"],
                index=0
            )
        
        # Filtrer les données selon le choix
        if filtre_jour == "Semaine (Lun-Ven)":
            df_filtre = df[df['jour_semaine_num'] < 5]
            periode_label = "en semaine"
        elif filtre_jour == "Weekend (Sam-Dim)":
            df_filtre = df[df['jour_semaine_num'] >= 5]
            periode_label = "le weekend"
        else:
            df_filtre = df
            periode_label = "toute la semaine"
        
        # VÉRIFICATION : Le filtre a-t-il des données ?
        if len(df_filtre) == 0:
            st.warning(f"⚠️ Aucune donnée {periode_label} dans votre fichier. Affichage de toutes les données.")
            df_filtre = df
            periode_label = "toute la semaine (aucune donnée pour le filtre sélectionné)"
        
        with col_filtre2:
            st.info(f"📊 Analyse {periode_label} | {len(df_filtre):,} créneaux")
        
        st.markdown("---")
        
        # Analyser par terrain (avec filtre)
        analyse_terrains = df_filtre.groupby('terrain').agg({
            'statut': [
                ('total_creneaux', 'count'),
                ('creneaux_reserves', lambda x: (x == 'réservé').sum()),
                ('creneaux_vides', lambda x: (x == 'libre').sum())
            ]
        })
        
        # Aplatir les colonnes
        analyse_terrains.columns = ['total_creneaux', 'creneaux_reserves', 'creneaux_vides']
        analyse_terrains['taux_remplissage'] = (analyse_terrains['creneaux_reserves'] / 
                                                 analyse_terrains['total_creneaux'] * 100)
        analyse_terrains = analyse_terrains.reset_index()
        
        # Calculer CA actuel par terrain
        ca_actuel_terrains = df_filtre[df_filtre['statut']=='réservé'].groupby('terrain').agg({
            'prix_par_joueur': lambda x: (x * JOUEURS_PAR_TERRAIN).sum()  # 4 joueurs par créneau
        }).reset_index()
        ca_actuel_terrains.columns = ['terrain', 'ca_actuel']
        
        analyse_terrains = analyse_terrains.merge(ca_actuel_terrains, on='terrain', how='left')
        analyse_terrains['ca_actuel'] = analyse_terrains['ca_actuel'].fillna(0)
        
        # Métriques globales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Nombre de terrains",
                len(analyse_terrains)
            )
        
        with col2:
            taux_moyen = analyse_terrains['taux_remplissage'].mean()
            st.metric(
                "Taux moyen de remplissage",
                f"{taux_moyen:.1f}%"
            )
        
        with col3:
            total_creneaux_vides = analyse_terrains['creneaux_vides'].sum()
            st.metric(
                "Total créneaux vides",
                f"{total_creneaux_vides:,}"
            )
        
        with col4:
            ca_total_actuel = analyse_terrains['ca_actuel'].sum()
            st.metric(
                "CA actuel total",
                f"{ca_total_actuel:,.0f}€"
            )
        
        st.markdown("---")
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Taux de remplissage par terrain")
            
            fig_terrain_taux = go.Figure()
            
            colors = ['#F44336' if t < 40 else '#FF9800' if t < 60 else '#4CAF50' 
                     for t in analyse_terrains['taux_remplissage']]
            
            fig_terrain_taux.add_trace(go.Bar(
                x=analyse_terrains['terrain'].astype(str),
                y=analyse_terrains['taux_remplissage'],
                marker_color=colors,
                text=analyse_terrains['taux_remplissage'].round(1).astype(str) + '%',
                textposition='outside',
                name='Taux de remplissage'
            ))
            
            fig_terrain_taux.add_hline(
                y=seuil_remplissage,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Seuil promo: {seuil_remplissage}%"
            )
            
            fig_terrain_taux.update_layout(
                xaxis_title="Terrain",
                yaxis_title="Taux de remplissage (%)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_terrain_taux, use_container_width=True, key="terrain_taux")
        
        with col2:
            st.subheader("💰 CA actuel par terrain")
            
            fig_terrain_ca = go.Figure()
            
            fig_terrain_ca.add_trace(go.Bar(
                x=analyse_terrains['terrain'].astype(str),
                y=analyse_terrains['ca_actuel'],
                marker_color='#2196F3',
                text=analyse_terrains['ca_actuel'].apply(lambda x: f"{x:,.0f}€"),
                textposition='outside',
                name='CA actuel'
            ))
            
            fig_terrain_ca.update_layout(
                xaxis_title="Terrain",
                yaxis_title="CA actuel (€)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_terrain_ca, use_container_width=True, key="terrain_ca")
        
        # Tableau détaillé
        st.markdown("---")
        st.subheader("📋 Détail par terrain")
        
        # Formater le dataframe pour affichage
        df_terrain_display = analyse_terrains.copy()
        df_terrain_display['taux_remplissage'] = df_terrain_display['taux_remplissage'].round(1).astype(str) + '%'
        df_terrain_display['ca_actuel'] = df_terrain_display['ca_actuel'].apply(lambda x: f"{x:,.0f}€")
        
        df_terrain_display.columns = [
            'Terrain',
            'Total créneaux',
            'Créneaux réservés',
            'Créneaux vides',
            'Taux remplissage',
            'CA actuel'
        ]
        
        st.dataframe(df_terrain_display, use_container_width=True, hide_index=True)
        
        # Heatmap remplissage par terrain et heure
        st.markdown("---")
        st.subheader(f"🗺️ Heatmap : Remplissage par terrain et heure {periode_label}")
        
        # Calculer taux de remplissage par terrain et heure (avec filtre)
        remplissage_terrain_heure = df_filtre.groupby(['terrain', 'heure_debut']).agg({
            'statut': lambda x: (x == 'réservé').sum() / len(x) * 100
        }).reset_index()
        remplissage_terrain_heure.columns = ['terrain', 'heure', 'taux']
        
        # Créer pivot
        pivot_terrain = remplissage_terrain_heure.pivot(
            index='terrain',
            columns='heure',
            values='taux'
        )
        
        # Renommer les index pour affichage
        pivot_terrain.index = ['Terrain ' + str(i) for i in pivot_terrain.index]
        
        # Utiliser la fonction create_heatmap (REFACTORING - Priorité 2)
        fig_heatmap_terrain = create_heatmap(
            pivot_data=pivot_terrain,
            title=f"Taux de remplissage par terrain et heure {periode_label}",
            colorscale='RdYlGn_r',
            text_suffix='%',
            zmin=0,
            zmax=100,
            colorbar_title="Taux (%)"
        )
        
        fig_heatmap_terrain.update_layout(
            height=400,
            yaxis_title="Terrain"
        )
        
        st.plotly_chart(fig_heatmap_terrain, use_container_width=True, key="heatmap_terrain")
        
        # Insights
        st.info(f"""
        **💡 Insights :**
        - Terrain le mieux rempli : Terrain {analyse_terrains.loc[analyse_terrains['taux_remplissage'].idxmax(), 'terrain']} 
          ({analyse_terrains['taux_remplissage'].max():.1f}%)
        - Terrain le moins rempli : Terrain {analyse_terrains.loc[analyse_terrains['taux_remplissage'].idxmin(), 'terrain']} 
          ({analyse_terrains['taux_remplissage'].min():.1f}%)
        - Écart de remplissage : {analyse_terrains['taux_remplissage'].max() - analyse_terrains['taux_remplissage'].min():.1f} points
        - **Opportunité** : {total_creneaux_vides} créneaux vides à optimiser avec le yield management
        """)
    
    # ============================================================
    # TAB 1 : VUE D'ENSEMBLE
    # ============================================================
    with tab1:
        st.header("📊 Vue d'ensemble")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Créneaux vides éligibles",
                f"{len(creneaux_promo)}",
                help="Combinaisons jour/heure avec < X% de remplissage"
            )
        
        # Calculer le nombre RÉEL de créneaux vides
        df_avec_taux_vue = df.merge(
            remplissage[['jour', 'heure', 'taux']], 
            left_on=['jour_semaine', 'heure_debut'],
            right_on=['jour', 'heure'],
            how='left'
        )
        nb_creneaux_vides_reels = len(df_avec_taux_vue[
            (df_avec_taux_vue['statut'] == 'libre') & 
            (df_avec_taux_vue['taux'] < seuil_remplissage)
        ])
        
        with col2:
            st.metric(
                "Créneaux RÉELS vides éligibles",
                f"{nb_creneaux_vides_reels:,}",
                help=f"Créneaux individuels réels dans les données (terrains × dates × heures)"
            )
        
        with col3:
            prix_moyen_promo = creneaux_promo['prix_dynamique'].mean()
            st.metric(
                "Prix moyen promo",
                f"{prix_moyen_promo:.1f}€",
                delta=f"-{((12-prix_moyen_promo)/12*100):.0f}%",
                delta_color="inverse"
            )
        
        with col4:
            nb_creneaux_vides = len(df[(df['statut']=='libre') & 
                                       (df['heure_debut'] < '17:00')])
            st.metric(
                "Créneaux vides totaux",
                f"{nb_creneaux_vides:,}",
                help="Tous les créneaux vides (même ceux non éligibles)"
            )
        
        st.markdown("---")
        
        # Graphique : Aperçu des prix selon le taux
        st.subheader("🎯 Aperçu des prix dynamiques")
        
        # Générer courbe théorique
        taux_range = np.arange(0, 100, 1)
        prix_theorique = [get_prix_dynamique(t, 12.0) for t in taux_range]
        
        fig_apercu = go.Figure()
        
        # Courbe théorique
        fig_apercu.add_trace(go.Scatter(
            x=taux_range,
            y=prix_theorique,
            mode='lines',
            name='Prix dynamique',
            line=dict(color='#2196F3', width=3),
            fill='tozeroy',
            fillcolor='rgba(33, 150, 243, 0.1)'
        ))
        
        # Ligne prix de base
        fig_apercu.add_trace(go.Scatter(
            x=[0, 100],
            y=[12, 12],
            mode='lines',
            name='Prix de base',
            line=dict(color='#4CAF50', width=2, dash='dash')
        ))
        
        # Ligne prix plancher
        fig_apercu.add_trace(go.Scatter(
            x=[0, 100],
            y=[prix_plancher, prix_plancher],
            mode='lines',
            name='Prix plancher',
            line=dict(color='#F44336', width=2, dash='dash')
        ))
        
        # Ligne seuil
        fig_apercu.add_vline(
            x=seuil_remplissage,
            line_dash="dot",
            line_color="orange",
            annotation_text=f"Seuil promo: {seuil_remplissage}%",
            annotation_position="top"
        )
        
        fig_apercu.update_layout(
            title="Prix par joueur selon le taux de remplissage",
            xaxis_title="Taux de remplissage (%)",
            yaxis_title="Prix par joueur (€)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_apercu, use_container_width=True, key="apercu_prix")
        
        # Distribution des créneaux
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Distribution des taux de remplissage")
            
            fig_dist = px.histogram(
                remplissage,
                x='taux',
                nbins=20,
                title="Nombre de créneaux par tranche de remplissage",
                labels={'taux': 'Taux de remplissage (%)', 'count': 'Nombre de créneaux'},
                color_discrete_sequence=['#2196F3']
            )
            
            fig_dist.add_vline(
                x=seuil_remplissage,
                line_dash="dash",
                line_color="red",
                annotation_text="Seuil promo"
            )
            
            st.plotly_chart(fig_dist, use_container_width=True, key="dist_taux")
        
        with col2:
            st.subheader("💰 Distribution des prix dynamiques")
            
            fig_prix = px.histogram(
                creneaux_promo,
                x='prix_dynamique',
                nbins=15,
                title="Nombre de créneaux par niveau de prix promo",
                labels={'prix_dynamique': 'Prix dynamique (€)', 'count': 'Nombre de créneaux'},
                color_discrete_sequence=['#FF9800']
            )
            
            st.plotly_chart(fig_prix, use_container_width=True, key="dist_prix")
    
    # ============================================================
    # TAB 2 : REVENUES
    # ============================================================
    with tab2:
        st.header("💰 Potentiel de Revenues Additionnels")
        
        # CALCUL CORRECT : Compter les créneaux RÉELS vides dans les données
        # Pas juste les combinaisons jour/heure
        
        # Identifier les créneaux vides éligibles (< seuil) dans les données réelles
        df_avec_taux = df.merge(
            remplissage[['jour', 'heure', 'taux', 'prix_dynamique']], 
            left_on=['jour_semaine', 'heure_debut'],
            right_on=['jour', 'heure'],
            how='left'
        )
        
        # VÉRIFICATION CRITIQUE : Le merge a-t-il fonctionné ?
        if df_avec_taux['taux'].isna().all():
            st.error("❌ Erreur lors de la fusion des données - Les noms de jours ou heures ne correspondent pas")
            st.stop()
        
        # Créneaux vides ET éligibles promo
        creneaux_vides_eligibles = df_avec_taux[
            (df_avec_taux['statut'] == 'libre') & 
            (df_avec_taux['taux'] < seuil_remplissage)
        ]
        
        nb_creneaux_vides_eligibles = len(creneaux_vides_eligibles)
        
        if nb_creneaux_vides_eligibles > 0:
            prix_moyen_promo = creneaux_vides_eligibles['prix_dynamique'].mean()
        else:
            prix_moyen_promo = prix_plancher
        
        # Calculer les scénarios sur les créneaux RÉELS
        scenarios = {
            'Conservateur': 0.20,
            'Modéré': 0.35,
            'Optimiste': 0.50
        }
        
        results = []
        nb_semaines = df['semaine'].nunique()
        
        for nom, taux_vente in scenarios.items():
            nb_vendus = int(nb_creneaux_vides_eligibles * taux_vente)
            ca_total = nb_vendus * prix_moyen_promo * JOUEURS_PAR_TERRAIN  # 4 joueurs
            ca_mensuel = ca_total / (nb_semaines / SEMAINES_PAR_MOIS)  # Convertir en mensuel
            
            results.append({
                'Scénario': nom,
                'Taux vente': f"{int(taux_vente*100)}%",
                'Créneaux vendus': nb_vendus,
                'CA total': ca_total,
                'CA mensuel': ca_mensuel
            })
        
        df_results = pd.DataFrame(results)
        
        # Métriques en haut
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Créneaux vides éligibles",
                f"{nb_creneaux_vides_eligibles:,}",
                help="Créneaux réels actuellement vides avec < X% remplissage"
            )
        
        with col2:
            st.metric(
                "Scénario Conservateur",
                f"{df_results.iloc[0]['CA mensuel']:,.0f}€/mois",
                help="20% des créneaux vides vendus"
            )
        
        with col3:
            st.metric(
                "Scénario Modéré",
                f"{df_results.iloc[1]['CA mensuel']:,.0f}€/mois",
                help="35% des créneaux vides vendus"
            )
        
        with col4:
            st.metric(
                "Scénario Optimiste",
                f"{df_results.iloc[2]['CA mensuel']:,.0f}€/mois",
                help="50% des créneaux vides vendus"
            )
        
        st.markdown("---")
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 CA Mensuel par Scénario")
            
            fig_ca = go.Figure()
            
            colors = ['#4CAF50', '#2196F3', '#FF9800']
            
            fig_ca.add_trace(go.Bar(
                x=df_results['Scénario'],
                y=df_results['CA mensuel'],
                marker_color=colors,
                text=df_results['CA mensuel'].apply(lambda x: f"{x:,.0f}€"),
                textposition='outside'
            ))
            
            fig_ca.update_layout(
                title="CA Mensuel Additionnel",
                yaxis_title="CA mensuel (€)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_ca, use_container_width=True, key="revenues_ca")
        
        with col2:
            st.subheader("📈 Créneaux Vendus par Scénario")
            
            fig_creneaux = go.Figure()
            
            fig_creneaux.add_trace(go.Bar(
                x=df_results['Scénario'],
                y=df_results['Créneaux vendus'],
                marker_color=colors,
                text=df_results['Créneaux vendus'],
                textposition='outside'
            ))
            
            fig_creneaux.update_layout(
                title="Volume de Créneaux Vendus",
                yaxis_title="Nombre de créneaux",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_creneaux, use_container_width=True, key="revenues_creneaux")
        
        # Tableau détaillé
        st.subheader("📋 Tableau Récapitulatif")
        
        # Formater le dataframe pour affichage
        df_display = df_results.copy()
        df_display['CA total'] = df_display['CA total'].apply(lambda x: f"{x:,.0f}€")
        df_display['CA mensuel'] = df_display['CA mensuel'].apply(lambda x: f"{x:,.0f}€")
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # Explication
        st.info(f"""
        **💡 Méthodologie de calcul :**
        
        **Base de calcul :**
        - {nb_creneaux_vides_eligibles:,} créneaux **réels** actuellement vides (statut='libre')
        - Avec un taux de remplissage < {seuil_remplissage}%
        - Sur la période : {nb_semaines:.0f} semaines de données
        
        **Formule :**
        - CA = Créneaux vendus × Prix moyen promo × 4 joueurs
        - Prix moyen promo : {prix_moyen_promo:.1f}€/joueur = {prix_moyen_promo*4:.0f}€/créneau
        
        **Important :**
        - Ces créneaux génèrent actuellement **0€** → CA 100% additionnel
        - Pas de cannibalisation : les créneaux bien remplis restent au prix normal
        
        **Note :**
        - "Créneaux jour/heure" = Combinaisons uniques (ex: "Lundi 9h") = {len(creneaux_promo)}
        - "Créneaux réels" = Instances réelles sur tous les terrains et toutes les dates = {nb_creneaux_vides_eligibles:,}
        - C'est sur les créneaux réels qu'on calcule le CA !
        """)
    
    # ============================================================
    # TAB 3 : CALENDRIER IMPACT YIELD
    # ============================================================
    with tab3:
        st.header("📅 Impact du Yield Management - Vue Calendrier")
        st.markdown("### Remplissage actuel vs. Remplissage avec promos")
        
        # Sélecteur de scénario
        col1, col2 = st.columns([1, 3])
        with col1:
            scenario_choisi = st.selectbox(
                "Scénario de vente",
                ["Conservateur (20%)", "Modéré (35%)", "Optimiste (50%)"],
                index=1
            )
        
        taux_vente_map = {
            "Conservateur (20%)": 0.20,
            "Modéré (35%)": 0.35,
            "Optimiste (50%)": 0.50
        }
        taux_vente = taux_vente_map[scenario_choisi]
        
        with col2:
            st.info(f"📊 Avec le scénario **{scenario_choisi}**, on vend {int(taux_vente*100)}% des créneaux vides éligibles aux promos")
        
        st.markdown("---")
        
        # Vérifier qu'on a des données
        if len(remplissage) == 0:
            st.error("Aucune donnée de remplissage disponible")
            st.stop()
        
        # Préparer les données pour le calendrier
        jours_order = JOURS_ORDRE_FR
        
        # Créer le DataFrame pour le calendrier
        calendrier_data = []
        
        for _, row in remplissage.iterrows():
            jour = row['jour']
            heure = row['heure']
            taux_actuel = row['taux']
            prix_dynamique = row['prix_dynamique']
            
            # Calculer taux additionnel si éligible
            if taux_actuel < seuil_remplissage:
                taux_additionnel = (100 - taux_actuel) * taux_vente
                taux_final = min(taux_actuel + taux_additionnel, 100)
            else:
                taux_additionnel = 0
                taux_final = taux_actuel
            
            calendrier_data.append({
                'jour': jour,
                'heure': heure,
                'taux_actuel': taux_actuel,
                'taux_additionnel': taux_additionnel,
                'taux_final': taux_final,
                'prix': prix_dynamique
            })
        
        df_cal = pd.DataFrame(calendrier_data)
        
        # ============================================================
        # DÉTECTION AUTOMATIQUE DE LA LANGUE DES JOURS (FIX heatmaps vides)
        # ============================================================
        jours_francais = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        jours_anglais = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Mapper anglais -> français
        jour_mapping_en_to_fr = {
            'Monday': 'Lundi', 'Tuesday': 'Mardi', 'Wednesday': 'Mercredi',
            'Thursday': 'Jeudi', 'Friday': 'Vendredi', 'Saturday': 'Samedi', 'Sunday': 'Dimanche'
        }
        
        # Détecter la langue utilisée dans les données
        jours_dans_data = df_cal['jour'].unique()
        if len(jours_dans_data) > 0:
            premier_jour = jours_dans_data[0]
            
            # Si en anglais, convertir en français
            if premier_jour in jours_anglais:
                st.info(f"📝 Jours détectés en anglais, conversion en français...")
                df_cal['jour'] = df_cal['jour'].map(jour_mapping_en_to_fr)
            # Si en français mais minuscules, normaliser
            elif premier_jour.lower() in [j.lower() for j in jours_francais]:
                df_cal['jour'] = df_cal['jour'].str.capitalize()
        
        jours_order = jours_francais
        
        # Debug info amélioré
        
        if len(df_cal) == 0:
            st.error("Aucune donnée disponible pour créer les calendriers")
            st.stop()
        
        # Créer deux heatmaps côte à côte
        col_heat1, col_heat2 = st.columns(2)
        
        with col_heat1:
            st.subheader("🔴 Remplissage ACTUEL")
            
            # Pivot pour le taux actuel
            pivot_actuel = df_cal.pivot(index='jour', columns='heure', values='taux_actuel')
            
            # Debug pivot avant reindex
            
            # Reindex avec les jours français
            pivot_actuel = pivot_actuel.reindex(jours_order)
            
            # Vérifier si on a des données valides
            if pivot_actuel.notna().sum().sum() == 0:
                st.error("❌ Aucune donnée après le reindex - Les jours dans vos données ne correspondent pas")
            else:
                # Remplacer NaN par 0 pour l'affichage seulement
                pivot_actuel_display = pivot_actuel.fillna(0)
                
                # Créer texte hover et texte d'affichage
                hover_actuel = []
                text_actuel = []
                for i in range(len(pivot_actuel.index)):
                    row_hover = []
                    row_text = []
                    for j in range(len(pivot_actuel.columns)):
                        val = pivot_actuel.iloc[i, j]
                        if pd.notna(val) and val > 0:
                            row_hover.append(f"{pivot_actuel.index[i]}<br>{pivot_actuel.columns[j]}<br>Actuel: {val:.0f}%")
                            row_text.append(f"{val:.0f}")
                        else:
                            row_hover.append("Pas de données")
                            row_text.append("")
                    hover_actuel.append(row_hover)
                    text_actuel.append(row_text)
                
                fig_actuel = go.Figure(data=go.Heatmap(
                    z=pivot_actuel_display.values,
                    x=pivot_actuel.columns,
                    y=pivot_actuel.index,
                    colorscale='RdYlGn_r',  # Inversé : Rouge = plein, Vert = vide
                    text=text_actuel,
                    texttemplate='%{text}%',
                    textfont={"size": 10, "color": "white"},
                    hovertext=hover_actuel,
                    hoverinfo='text',
                    colorbar=dict(title="Taux (%)", x=1.02),
                    zmin=0,
                    zmax=100,
                    showscale=True
                ))
                
                fig_actuel.update_layout(
                    height=500,
                    xaxis_title="Heure",
                    yaxis_title="Jour",
                    margin=dict(l=50, r=50, t=30, b=50)
                )
                
                st.plotly_chart(fig_actuel, use_container_width=True, key="heatmap_actuel")
        
        with col_heat2:
            st.subheader("🟢 Remplissage AVEC PROMOS")
            
            # Pivot pour le taux final
            pivot_final = df_cal.pivot(index='jour', columns='heure', values='taux_final')
            pivot_final = pivot_final.reindex(jours_order)
            
            pivot_add = df_cal.pivot(index='jour', columns='heure', values='taux_additionnel')
            pivot_add = pivot_add.reindex(jours_order)
            
            # Remplacer NaN par 0 pour l'affichage
            pivot_final_display = pivot_final.fillna(0)
            
            # Créer texte hover et texte d'affichage
            hover_final = []
            text_final = []
            for i in range(len(pivot_final.index)):
                row_hover = []
                row_text = []
                for j in range(len(pivot_final.columns)):
                    val_final = pivot_final.iloc[i, j]
                    val_add = pivot_add.iloc[i, j]
                    if pd.notna(val_final) and val_final > 0:
                        gain_text = f" (+{val_add:.0f}%)" if pd.notna(val_add) and val_add > 0 else ""
                        row_hover.append(f"{pivot_final.index[i]}<br>{pivot_final.columns[j]}<br>Final: {val_final:.0f}%<br>Gain: +{val_add:.0f}%")
                        row_text.append(f"{val_final:.0f}")
                    else:
                        row_hover.append("Pas de données")
                        row_text.append("")
                hover_final.append(row_hover)
                text_final.append(row_text)
            
            fig_final = go.Figure(data=go.Heatmap(
                z=pivot_final_display.values,
                x=pivot_final.columns,
                y=pivot_final.index,
                colorscale='RdYlGn_r',  # Inversé : Rouge = plein, Vert = vide
                text=text_final,
                texttemplate='%{text}%',
                textfont={"size": 10, "color": "white"},
                hovertext=hover_final,
                hoverinfo='text',
                colorbar=dict(title="Taux (%)", x=1.02),
                zmin=0,
                zmax=100,
                showscale=True
            ))
            
            fig_final.update_layout(
                height=500,
                xaxis_title="Heure",
                yaxis_title="Jour",
                margin=dict(l=50, r=50, t=30, b=50)
            )
            
            st.plotly_chart(fig_final, use_container_width=True, key="heatmap_final")
        
        # Heatmap du GAIN (différence)
        st.markdown("---")
        st.subheader("📈 GAIN de Remplissage (Différence)")
        
        pivot_gain = pivot_final - pivot_actuel
        pivot_gain_display = pivot_gain.fillna(0)
        
        # Créer texte hover pour gain
        hover_gain = []
        text_gain = []
        for i in range(len(pivot_gain.index)):
            row_hover = []
            row_text = []
            for j in range(len(pivot_gain.columns)):
                val_gain = pivot_gain.iloc[i, j]
                val_actuel = pivot_actuel.iloc[i, j]
                val_final = pivot_final.iloc[i, j]
                if pd.notna(val_gain) and pd.notna(val_actuel) and val_gain > 0:
                    row_hover.append(f"{pivot_gain.index[i]}<br>{pivot_gain.columns[j]}<br>{val_actuel:.0f}% → {val_final:.0f}%<br>Gain: +{val_gain:.0f}%")
                    row_text.append(f"{val_gain:.0f}")
                else:
                    row_hover.append("Pas de promo")
                    row_text.append("")
            hover_gain.append(row_hover)
            text_gain.append(row_text)
        
        fig_gain = go.Figure(data=go.Heatmap(
            z=pivot_gain_display.values,
            x=pivot_gain.columns,
            y=pivot_gain.index,
            colorscale='RdYlGn',  # Normal : Rouge = pas de gain, Vert = gros gain
            text=text_gain,
            texttemplate='+%{text}%',
            textfont={"size": 11, "color": "black"},
            hovertext=hover_gain,
            hoverinfo='text',
            colorbar=dict(title="Gain (%)"),
            zmin=0,
            zmax=50,
            showscale=True
        ))
        
        fig_gain.update_layout(
            title=f"Points de remplissage gagnés par créneau - Scénario {scenario_choisi}",
            height=500,
            xaxis_title="Heure de début",
            yaxis_title="Jour de la semaine"
        )
        
        st.plotly_chart(fig_gain, use_container_width=True, key="heatmap_gain")
        
        # Statistiques d'impact
        st.markdown("---")
        st.subheader("📊 Statistiques d'impact")
        
        col1, col2, col3, col4 = st.columns(4)
        
        creneaux_impactes = df_cal[df_cal['taux_additionnel'] > 0]
        
        if len(creneaux_impactes) > 0:
            with col1:
                st.metric(
                    "Créneaux impactés",
                    len(creneaux_impactes),
                    help="Nombre de créneaux jour/heure avec promos"
                )
            
            with col2:
                gain_moyen = creneaux_impactes['taux_additionnel'].mean()
                st.metric(
                    "Gain moyen",
                    f"+{gain_moyen:.1f} pts",
                    help="Points de remplissage gagnés en moyenne"
                )
            
            with col3:
                taux_actuel_moyen = creneaux_impactes['taux_actuel'].mean()
                st.metric(
                    "Taux actuel moyen",
                    f"{taux_actuel_moyen:.1f}%",
                    help="Sur les créneaux avec promos"
                )
            
            with col4:
                taux_final_moyen = creneaux_impactes['taux_final'].mean()
                st.metric(
                    "Taux final moyen",
                    f"{taux_final_moyen:.1f}%",
                    delta=f"+{taux_final_moyen - taux_actuel_moyen:.1f} pts"
                )
            
            # Top 10 gains
            st.markdown("---")
            st.subheader("🏆 Top 10 créneaux avec plus gros gain")
            
            top_gains = creneaux_impactes.nlargest(10, 'taux_additionnel')[
                ['jour', 'heure', 'taux_actuel', 'taux_additionnel', 'taux_final', 'prix']
            ].copy()
            
            top_gains.columns = ['Jour', 'Heure', 'Actuel (%)', 'Gain (pts)', 'Final (%)', 'Prix promo (€)']
            top_gains['Actuel (%)'] = top_gains['Actuel (%)'].round(1)
            top_gains['Gain (pts)'] = '+' + top_gains['Gain (pts)'].round(1).astype(str)
            top_gains['Final (%)'] = top_gains['Final (%)'].round(1)
            top_gains['Prix promo (€)'] = top_gains['Prix promo (€)'].round(1)
            
            st.dataframe(top_gains, use_container_width=True, hide_index=True)
        else:
            st.warning("Aucun créneau éligible aux promos avec les paramètres actuels")
        
        # Légende explicative
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("🟢 **Vert** : Créneaux vides (opportunité)")
        with col2:
            st.markdown("🟡 **Jaune** : Remplissage moyen")
        with col3:
            st.markdown("🔴 **Rouge** : Créneaux pleins")
    
    # ============================================================
    # TAB PRIX : GRILLE DE PRIX
    # ============================================================
    with tab_prix:
        st.header("💵 Grille des Prix Dynamiques")
        st.markdown("### Prix par jour et heure selon le taux de remplissage")
        
        # ============================================================
        # CONVERSION DES JOURS (même logique que Impact Yield)
        # ============================================================
        # Créer une copie pour ne pas modifier l'original
        remplissage_prix = remplissage.copy()
        
        jour_mapping_en_to_fr = {
            'Monday': 'Lundi', 'Tuesday': 'Mardi', 'Wednesday': 'Mercredi',
            'Thursday': 'Jeudi', 'Friday': 'Vendredi', 'Saturday': 'Samedi', 'Sunday': 'Dimanche'
        }
        
        # Détecter et convertir si nécessaire
        jours_francais = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        jours_anglais = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        if len(remplissage_prix) > 0:
            premier_jour = remplissage_prix['jour'].iloc[0]
            
            if premier_jour in jours_anglais:
                st.info(f"📝 Jours détectés en anglais, conversion en français...")
                remplissage_prix['jour'] = remplissage_prix['jour'].map(jour_mapping_en_to_fr)
            elif premier_jour.lower() in [j.lower() for j in jours_francais]:
                remplissage_prix['jour'] = remplissage_prix['jour'].str.capitalize()
        
        # Préparer les données pour le heatmap
        jours_order = JOURS_ORDRE_FR
        
        # Vérifier que nous avons des données
        if len(remplissage_prix) > 0:
            try:
                pivot_prix = remplissage_prix.pivot(
                    index='jour',
                    columns='heure',
                    values='prix_dynamique'
                )
                
                pivot_taux = remplissage_prix.pivot(
                    index='jour',
                    columns='heure',
                    values='taux'
                )
                
                # Réordonner les jours
                pivot_prix = pivot_prix.reindex(jours_order)
                pivot_taux = pivot_taux.reindex(jours_order)
                
                # Ne PAS remplacer les NaN par 0, les laisser tels quels
                pivot_prix_display = pivot_prix.copy()
                
                # Créer texte avec prix et taux
                hover_text = []
                text_display = []
                for i in range(len(pivot_prix.index)):
                    hover_row = []
                    text_row = []
                    for j in range(len(pivot_prix.columns)):
                        prix = pivot_prix.iloc[i, j]
                        taux = pivot_taux.iloc[i, j]
                        if pd.notna(prix) and pd.notna(taux):
                            hover_row.append(f"{pivot_prix.index[i]}<br>{pivot_prix.columns[j]}<br>Prix: {prix:.1f}€<br>Taux: {taux:.0f}%")
                            text_row.append(f"{prix:.1f}")
                        else:
                            hover_row.append("Pas de données")
                            text_row.append("")
                    hover_text.append(hover_row)
                    text_display.append(text_row)
                
                # Créer le heatmap
                fig_cal = go.Figure(data=go.Heatmap(
                    z=pivot_prix_display.values,
                    x=pivot_prix_display.columns,
                    y=pivot_prix_display.index,
                    colorscale='RdYlGn_r',
                    text=text_display,
                    hovertext=hover_text,
                    hoverinfo='text',
                    texttemplate='%{text}€',
                    textfont={"size": 11, "color": "white"},
                    colorbar=dict(title="Prix (€)"),
                    hoverongaps=False,
                    zmin=prix_plancher,
                    zmax=15
                ))
                
                fig_cal.update_layout(
                    title=f"Prix dynamiques par jour et heure (Créneaux 1h30)<br><sub>Seuil promo: {seuil_remplissage}% | Prix plancher: {prix_plancher}€</sub>",
                    xaxis_title="Heure de début",
                    yaxis_title="Jour de la semaine",
                    height=600,
                    xaxis={'side': 'bottom'}
                )
                
                st.plotly_chart(fig_cal, use_container_width=True, key="grille_prix")
                
                # Légende
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"🔴 **{prix_plancher:.0f}-{prix_plancher+1:.0f}€** : Ultra promo (≤15% remplissage)")
                with col2:
                    st.markdown(f"🟡 **{prix_plancher+3:.0f}-11€** : Promo modérée (15-{seuil_remplissage}%)")
                with col3:
                    st.markdown(f"🟢 **12€+** : Prix normal (≥{seuil_remplissage}%)")
                
                # Statistiques de la grille
                st.markdown("---")
                st.subheader("📊 Statistiques de la grille")
                
                col1, col2, col3, col4 = st.columns(4)
                
                nb_creneaux_ultra = (pivot_prix_display <= prix_plancher + 1).sum().sum()
                nb_creneaux_promo = ((pivot_prix_display > prix_plancher + 1) & 
                                     (pivot_prix_display < 12)).sum().sum()
                nb_creneaux_normal = (pivot_prix_display >= 12).sum().sum()
                total_creneaux = nb_creneaux_ultra + nb_creneaux_promo + nb_creneaux_normal
                
                with col1:
                    st.metric(
                        "Créneaux ultra-promo",
                        nb_creneaux_ultra,
                        help="Prix ≤ 7€"
                    )
                
                with col2:
                    st.metric(
                        "Créneaux promo modérée",
                        nb_creneaux_promo,
                        help="Prix entre 7€ et 12€"
                    )
                
                with col3:
                    st.metric(
                        "Créneaux prix normal",
                        nb_creneaux_normal,
                        help="Prix ≥ 12€"
                    )
                
                with col4:
                    pct_promo = (nb_creneaux_ultra + nb_creneaux_promo) / total_creneaux * 100
                    st.metric(
                        "% créneaux en promo",
                        f"{pct_promo:.0f}%"
                    )
                
            # FIX PRIORITÉ 2: Gestion d'exception spécifique au lieu de Exception générale
            except (KeyError, ValueError, AttributeError) as e:
                st.error(f"❌ Erreur lors de la création du calendrier de prix: {e}")
                st.info("💡 Vérifiez que vos données contiennent les colonnes requises")
        else:
            st.warning("⚠️ Aucune donnée disponible pour le calendrier")
    
    # ============================================================
    # TAB 4 : DÉTAILS
    # ============================================================
    with tab4:
        st.header("📈 Analyse Détaillée")
        
        # Top créneaux avec plus forte réduction
        st.subheader("🔥 Top 10 créneaux avec plus forte réduction")
        
        top_reductions = creneaux_promo.nlargest(10, 'reduction')[
            ['jour', 'heure', 'taux', 'prix_base', 'prix_dynamique', 'reduction']
        ].copy()
        
        top_reductions['taux'] = top_reductions['taux'].round(1).astype(str) + '%'
        top_reductions['prix_base'] = top_reductions['prix_base'].astype(str) + '€'
        top_reductions['prix_dynamique'] = top_reductions['prix_dynamique'].round(1).astype(str) + '€'
        top_reductions['reduction'] = top_reductions['reduction'].round(0).astype(int).astype(str) + '%'
        
        st.dataframe(top_reductions, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Taux de remplissage par heure
        st.subheader("📊 Taux de remplissage par heure")
        
        remplissage_heure = remplissage.groupby('heure')['taux'].mean().reset_index()
        
        fig_heure = go.Figure()
        
        fig_heure.add_trace(go.Bar(
            x=remplissage_heure['heure'],
            y=remplissage_heure['taux'],
            marker_color=['#F44336' if t < seuil_remplissage else '#4CAF50' 
                         for t in remplissage_heure['taux']],
            text=remplissage_heure['taux'].round(1).astype(str) + '%',
            textposition='outside'
        ))
        
        fig_heure.add_hline(
            y=seuil_remplissage,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Seuil promo: {seuil_remplissage}%"
        )
        
        fig_heure.update_layout(
            title="Taux de remplissage moyen par heure",
            xaxis_title="Heure de début",
            yaxis_title="Taux de remplissage (%)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_heure, use_container_width=True, key="details_heure")
        
        # Exporter les données
        st.markdown("---")
        st.subheader("💾 Export des données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export prix dynamiques
            csv_prix = creneaux_promo.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger grille de prix",
                data=csv_prix,
                file_name=f"prix_dynamiques_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export résultats revenues
            csv_results = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger résultats revenues",
                data=csv_results,
                file_name=f"revenues_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

else:
    # Page d'accueil si pas de fichier
    st.info("👈 Chargez un fichier CSV dans la sidebar pour commencer l'analyse")
    
    st.markdown("""
    ### 🎯 Comment utiliser cette application ?
    
    1. **Chargez votre fichier CSV** avec les colonnes :
       - `date` : Date du créneau
       - `heure_debut` : Heure de début
       - `terrain` : Numéro du terrain
       - `prix_par_joueur` : Prix par joueur
       - `statut` : 'réservé' ou 'libre'
    
    2. **Ajustez les paramètres** dans la sidebar :
       - Seuil de remplissage pour activer les promos
       - Prix plancher minimum
    
    3. **Explorez les résultats** dans les 4 onglets :
       - 📊 Vue d'ensemble : Graphiques et métriques globales
       - 💰 Revenues : Scénarios de CA additionnels
       - 📅 Calendrier : Heatmap des prix par jour/heure
       - 📈 Détails : Analyses approfondies et exports
    
    ### 💡 Les graphiques se mettent à jour en temps réel quand vous changez les paramètres !
    """)
    
    # Exemple de fichier
    st.markdown("---")
    st.markdown("### 📄 Exemple de structure CSV attendue :")
    
    example_df = pd.DataFrame({
        'date': ['2025-07-01', '2025-07-01', '2025-07-01'],
        'heure_debut': ['09:00', '10:30', '12:00'],
        'terrain': [1, 1, 1],
        'prix_par_joueur': [12, 12, 12],
        'statut': ['libre', 'réservé', 'libre'],
        'user_id': ['', 'USER_1234', '']
    })
    
    st.dataframe(example_df, use_container_width=True, hide_index=True)
