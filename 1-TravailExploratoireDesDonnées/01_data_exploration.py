#!/usr/bin/env python3
# streamlit pour analyses eda 

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# configuration de la page
st.set_page_config(
    page_title="EDA AccesLibre",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# css personnalise pour harmoniser avec soutenance.html
st.markdown("""
<style>
/* harmonisation avec le template soutenance.html */
.main {
    background: #40505f;
    color: #2c3e50;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.stApp {
    background: #40505f;
}

/* header style */
.stTitle {
    color: #2c3e50;
    font-weight: 300;
    letter-spacing: 0.5px;
    background: rgba(255, 255, 255, 0.95);
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(52, 73, 94, 0.1);
    box-shadow: 0 2px 20px rgba(0, 0, 0, 0.05);
}

/* sections style */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #2c3e50;
    background: rgba(255, 255, 255, 0.95);
    padding: 15px 20px;
    border-radius: 8px;
    margin: 10px 0;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(52, 73, 94, 0.1);
}

/* metriques style */
.metric-container {
    background: rgba(255, 255, 255, 0.95);
    padding: 15px;
    border-radius: 8px;
    border: 1px solid rgba(52, 73, 94, 0.1);
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

/* dataframe style */
.stDataFrame {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 8px;
    border: 1px solid rgba(52, 73, 94, 0.1);
}

/* sidebar style */
.css-1d391kg {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
}

/* graphiques plotly */
.js-plotly-plot {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 8px;
    border: 1px solid rgba(52, 73, 94, 0.1);
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

/* footer style */
.footer {
    background: rgba(255, 255, 255, 0.95);
    padding: 15px;
    border-radius: 8px;
    margin-top: 20px;
    text-align: center;
    color: #7f8c8d;
    font-size: 12px;
    border: 1px solid rgba(52, 73, 94, 0.1);
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_dataset():
    """chargement du dataset avec cache streamlit"""
    file_path = os.path.join('..', 'SourceData', 'acceslibre-with-web-url.csv')
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        return df
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin-1', low_memory=False)
        return df

def analyze_completeness(df):
    # detection automatique des colonnes d'accessibilite, a voir supprimer la variable accessibility_vars
    accessibility_vars = []
    
    # colonnes avec 'pmr' dans le nom
    accessibility_keywords = [
        'pmr', 'handicap', 'accessib', 'entree', 'largeur', 'marche', 'rampe',
        'sanitaire', 'ascenseur', 'stationnement', 'cheminement', 'accueil'
    ]
    
    accessibility_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in accessibility_keywords):
            accessibility_cols.append(col)
    
    # calculer la completude pour chaque variable
    completude_data = []
    for col in accessibility_cols:
        total = len(df)
        non_null = df[col].notna().sum()
        completude_pct = (non_null / total) * 100
        
        completude_data.append({
            'variable': col,
            'total': total,
            'renseigne': non_null,
            'manquant': total - non_null,
            'completude_pct': completude_pct
        })
    
    return pd.DataFrame(completude_data).sort_values('completude_pct', ascending=False)

def analyze_public_establishments(df):
    """analyse des etablissements publics"""
    public_activities = ['Mairie', 'Bureau de poste', 'Gendarmerie', 'École primaire', 
                        'Bibliothèque médiathèque', 'Toilettes publiques', 'Hôpital', 
                        'Centre médical', 'Pharmacie']
    
    public_stats = {}
    if 'activite' in df.columns:
        activites = df['activite'].value_counts()
        for act in public_activities:
            count = activites.get(act, 0)
            if count > 0:
                public_stats[act] = count
    
    # recherche par mots-cles dans les noms
    keyword_stats = {}
    if 'name' in df.columns:
        keywords = ['mairie', 'poste', 'gendarmerie', 'école', 'ecole', 'bibliothèque']
        for keyword in keywords:
            count = df['name'].str.lower().str.contains(keyword, na=False).sum()
            if count > 0:
                keyword_stats[keyword] = count
    
    return public_stats, keyword_stats

def categorize_columns(df):
    """categorise les colonnes par domaine metier"""
    categories = {
        'Identification': ['id', 'name', 'siret'],
        'Localisation': ['postal_code', 'commune', 'numero', 'voie', 'lieu_dit', 'code_insee', 'longitude', 'latitude'],
        'Contact': ['contact_url', 'site_internet', 'web_url'],
        'Activité': ['activite'],
        'Transport': ['transport_station_presence'],
        'Stationnement': [col for col in df.columns if 'stationnement' in col.lower()],
        'Cheminement_Extérieur': [col for col in df.columns if 'cheminement_ext' in col.lower()],
        'Entrée': [col for col in df.columns if 'entree' in col.lower()],
        'Accueil': [col for col in df.columns if 'accueil' in col.lower()],
        'Sanitaires': [col for col in df.columns if 'sanitaires' in col.lower()],
        'Labels': ['labels', 'labels_familles_handicap', 'registre_url', 'conformite']
    }
    
    category_stats = {}
    for cat, cols in categories.items():
        existing_cols = [c for c in cols if c in df.columns]
        if existing_cols:
            category_stats[cat] = {
                'count': len(existing_cols),
                'columns': existing_cols[:5],  # max 5 exemples
                'completeness': [(col, df[col].notna().sum(), df[col].notna().sum()/len(df)*100) for col in existing_cols[:3]]
            }
    
    return category_stats

def get_ml_variables(df):
    """identifie les variables interessantes pour le ml"""
    # variables numeriques
    numeric_vars = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in ['id', 'postal_code', 'siret', 'longitude', 'latitude', 'code_insee']:
            non_null = df[col].notna().sum()
            if non_null > 1000:  # au moins 1000 valeurs
                numeric_vars.append({
                    'column': col,
                    'count': non_null,
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean()
                })
    
    # variables categorielles
    categorical_vars = []
    for col in df.select_dtypes(include=['object']).columns:
        if col not in ['id', 'name', 'voie', 'lieu_dit', 'contact_url', 'site_internet', 'web_url', 'registre_url']:
            non_null = df[col].notna().sum()
            unique_vals = df[col].nunique()
            if non_null > 1000 and 2 <= unique_vals <= 20:
                categorical_vars.append({
                    'column': col,
                    'count': non_null,
                    'unique': unique_vals,
                    'top_values': df[col].value_counts().head(3).to_dict()
                })
    
    return numeric_vars, categorical_vars

def generate_accessibility_insights(df):
    """genere des insights avances sur l'accessibilite"""
    insights = {}
    
    # analyse des marches d'entree
    if 'entree_marches' in df.columns:
        marches_data = df['entree_marches'].dropna()
        if len(marches_data) > 0:
            accessible = (marches_data == 0).sum()
            difficulte_legere = ((marches_data > 0) & (marches_data <= 3)).sum()
            difficulte_moderee = ((marches_data > 3) & (marches_data <= 10)).sum()
            inaccessible = (marches_data > 10).sum()
            
            insights['marches'] = {
                'total': len(marches_data),
                'moyenne': marches_data.mean(),
                'mediane': marches_data.median(),
                'max': marches_data.max(),
                'accessible': {'count': accessible, 'pct': accessible/len(marches_data)*100},
                'difficulte_legere': {'count': difficulte_legere, 'pct': difficulte_legere/len(marches_data)*100},
                'difficulte_moderee': {'count': difficulte_moderee, 'pct': difficulte_moderee/len(marches_data)*100},
                'inaccessible': {'count': inaccessible, 'pct': inaccessible/len(marches_data)*100}
            }
    
    # analyse des largeurs d'entree
    if 'entree_largeur_mini' in df.columns:
        largeur_data = df['entree_largeur_mini'].dropna()
        if len(largeur_data) > 0:
            conforme_pmr = (largeur_data >= 90).sum()
            insights['largeurs'] = {
                'total': len(largeur_data),
                'moyenne': largeur_data.mean(),
                'mediane': largeur_data.median(),
                'conforme_pmr': {'count': conforme_pmr, 'pct': conforme_pmr/len(largeur_data)*100}
            }
    
    return insights

def create_completeness_chart(completude_df):
    """graphique de completude interactif"""
    fig = px.bar(
        completude_df,
        x='variable',
        y='completude_pct',
        title='Complétude des Variables d\'Accessibilité',
        labels={'completude_pct': 'Complétude (%)', 'variable': 'Variables'},
        color='completude_pct',
        color_continuous_scale='RdYlGn',
        text='completude_pct'
    )
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        showlegend=False
    )
    
    return fig

def analyze_accessibility_insights(df):
    """analyse des insights d'accessibilite"""
    insights = {}
    
    # analyse des marches d'entree
    if 'entree_marches' in df.columns:
        marches_data = df['entree_marches'].dropna()
        if len(marches_data) > 0:
            insights['marches'] = {
                'total_documentes': len(marches_data),
                'moyenne': marches_data.mean(),
                'mediane': marches_data.median(),
                'max': marches_data.max(),
                'accessible': (marches_data == 0).sum(),
                'difficulte_legere': ((marches_data > 0) & (marches_data <= 3)).sum(),
                'difficulte_moderee': ((marches_data > 3) & (marches_data <= 10)).sum(),
                'inaccessible': (marches_data > 10).sum()
            }
    
    # analyse des largeurs d'entree
    if 'entree_largeur_mini' in df.columns:
        largeur_data = df['entree_largeur_mini'].dropna()
        if len(largeur_data) > 0:
            insights['largeurs'] = {
                'total_documentes': len(largeur_data),
                'moyenne': largeur_data.mean(),
                'mediane': largeur_data.median(),
                'conforme_pmr': (largeur_data >= 90).sum(),
                'conforme_pmr_pct': (largeur_data >= 90).sum() / len(largeur_data) * 100
            }
    
    return insights

def create_accessibility_pie_chart(insights):
    """graphique en secteurs pour l'accessibilite"""
    if 'marches' not in insights:
        return None
    
    marches = insights['marches']
    
    labels = ['Accessible (0 marches)', 'Difficulté légère (1-3)', 
              'Difficulté modérée (4-10)', 'Inaccessible (>10)']
    values = [marches['accessible'], marches['difficulte_legere'], 
              marches['difficulte_moderee'], marches['inaccessible']]
    
    colors = ['#2E8B57', '#FFD700', '#FF8C00', '#DC143C']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        textinfo='label+percent+value',
        textposition='auto'
    )])
    
    fig.update_layout(
        title='Répartition des Établissements par Niveau d\'Accessibilité',
        height=500
    )
    
    return fig

def main():
    st.title("EDA sur AccesLibre")
        
    # chargement des donnees
    with st.spinner("Chargement du dataset..."):
        df = load_dataset()
    
    # metriques globales
    st.header(" Infos Métriques ")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Établissements", f"{len(df):,}")
    with col2:
        st.metric("Colonnes", len(df.columns))
    with col3:
        st.metric("Taille mémoire", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    with col4:
        missing_global = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        st.metric("Données manquantes", f"{missing_global:.1f}%")
    
    # analyse de completude
    st.header("Analyse d'Accessibilité")
    
    completude_df = analyze_completeness(df)
    
    # affichage des colonnes detectees
    st.subheader("Variables détectées automatiquement")
    st.info(f"**{len(completude_df)} variables d'accessibilité détectées** (PMR + mots-clés)")
    
        
    # graphique de completude
    fig_completude = create_completeness_chart(completude_df)
    st.plotly_chart(fig_completude, use_container_width=True)
    
    # tableau detaille
    st.subheader("Détail par Variable")
    st.dataframe(
        completude_df.style.format({
            'completude_pct': '{:.1f}%',
            'manquantes_pct': '{:.1f}%',
            'valeurs_renseignees': '{:,}',
            'manquantes': '{:,}'
        }),
        use_container_width=True
    )
    
    # observations eda
    st.subheader("Observations EDA")
    
    for _, row in completude_df.iterrows():
        var = row['variable']
        pct = row['completude_pct']
        
        if pct > 50:
            observation = "🟢 Données bien renseignées"
            color = "green"
        elif pct > 10:
            observation = "🟡 Données partiellement renseignées"
            color = "orange"
        else:
            observation = "🔴 Données très rares"
            color = "red"
        
        st.markdown(f"**{var}** ({pct:.1f}%) : :{color}[{observation}]")
    
    # analyse des etablissements publics
    st.header("Analyse Établissements Publics")
    public_stats, keyword_stats = analyze_public_establishments(df)
    
    if public_stats or keyword_stats:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Par Type d'Activité")
            if public_stats:
                for activite, count in list(public_stats.items())[:8]:
                    st.metric(activite, f"{count:,}")
            else:
                st.info("Aucune activité publique détectée")
        
        with col2:
            st.subheader("Par Mots-clés dans le Nom")
            if keyword_stats:
                for keyword, count in keyword_stats.items():
                    st.metric(keyword.title(), f"{count:,}")
            else:
                st.info("Aucun mot-clé détecté")
    
    # categorisation des colonnes
    st.header("Structure du Dataset")
    category_stats = categorize_columns(df)
    
    st.subheader("Colonnes par Domaine Métier")
    for category, stats in category_stats.items():
        with st.expander(f"{category} ({stats['count']} colonnes)"):
            st.write(f"**Colonnes :** {', '.join(stats['columns'])}")
            if stats['completeness']:
                st.write("**Complétude des principales colonnes :**")
                for col, count, pct in stats['completeness']:
                    st.write(f"- {col}: {count:,} valeurs ({pct:.1f}%)")
    
    # variables interessantes pour ml
    st.header("Variables pour Machine Learning")
    numeric_vars, categorical_vars = get_ml_variables(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Variables Numériques ({len(numeric_vars)})")
        if numeric_vars:
            for var in numeric_vars[:10]:  # top 10
                st.write(f"**{var['column']}**")
                st.write(f"- {var['count']:,} valeurs | Min: {var['min']:.1f} | Max: {var['max']:.1f} | Moy: {var['mean']:.1f}")
        else:
            st.info("Aucune variable numérique pertinente détectée")
    
    with col2:
        st.subheader(f"Variables Catégorielles ({len(categorical_vars)})")
        if categorical_vars:
            for var in categorical_vars[:10]:  # top 10
                st.write(f"**{var['column']}**")
                st.write(f"- {var['count']:,} valeurs | {var['unique']} catégories")
                top_vals = ', '.join([f"{k}({v})" for k, v in list(var['top_values'].items())[:3]])
                st.write(f"- Top: {top_vals}")
        else:
            st.info("Aucune variable de catégorie pertinente détectée")
    
    # insights avances sur l'accessibilite
    st.header("Insights Accessibilité Avancés")
    insights = generate_accessibility_insights(df)
    
    if 'marches' in insights:
        st.subheader("Analyse Marches d'Entrée")
        marches = insights['marches']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Établissements documentés", f"{marches['total']:,}")
            st.metric("Moyenne marches", f"{marches['moyenne']:.1f}")
            st.metric("Médiane marches", f"{marches['mediane']:.1f}")
            st.metric("Maximum marches", f"{marches['max']:.0f}")
        
        with col2:
            st.metric("Accessible (0 marches)", f"{marches['accessible']['pct']:.1f}%")
            st.metric("Difficulté légère (1-3)", f"{marches['difficulte_legere']['pct']:.1f}%")
            st.metric("Difficulté modérée (4-10)", f"{marches['difficulte_moderee']['pct']:.1f}%")
            st.metric("Inaccessible PMR (>10)", f"{marches['inaccessible']['pct']:.1f}%")
    
    if 'largeurs' in insights:
        st.subheader("Analyse Largeurs d'Entrée")
        largeurs = insights['largeurs']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Établissements documentés", f"{largeurs['total']:,}")
        with col2:
            st.metric("Largeur moyenne", f"{largeurs['moyenne']:.0f} cm")
        with col3:
            st.metric("Conformes PMR (≥90cm)", f"{largeurs['conforme_pmr']['pct']:.1f}%")
    
    
    
    # footer stylé
    st.markdown("""
    <div class="footer">
        <strong>EDA  - Développeur IA</strong><br>
        <em>Données : AccesLibre (592k établissements)</em><br>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
