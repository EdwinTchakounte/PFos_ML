import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Prédiction d'Orientation Scolaire",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé pour un design moderne
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .prediction-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    .metric-value {
        font-size: 48px;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 16px;
        opacity: 0.9;
    }
    </style>
""", unsafe_allow_html=True)

# Charger le modèle et les métadonnées
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    with open('metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return model, metadata

try:
    model, metadata = load_model()
except Exception as e:
    st.error(f"❌ Erreur lors du chargement du modèle: {e}")
    st.stop()

# En-tête avec design moderne
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #667eea; font-size: 48px; margin-bottom: 10px;'>
            🎓 Système de Prédiction d'Orientation Scolaire
        </h1>
        <p style='color: #666; font-size: 18px;'>
            Découvrez l'orientation académique recommandée basée sur le profil de l'élève
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Formulaire de saisie avec design amélioré
st.markdown("<h2 style='color: #667eea;'>📝 Informations de l'élève</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    ville = st.text_input(
        "🏙️ Ville", 
        placeholder="Entrez la ville de résidence",
        help="Ville où l'élève étudie actuellement"
    )
    
    serie = st.selectbox(
        " Série", 
        options=["", "A", "C", "D", "Autre"],
        help="Série actuelle de l'élève"
    )

with col2:
    souhait = st.text_input(
        " Souhait d'orientation", 
        placeholder="Ex: Sciences, Lettres, Technique...",
        help="Domaine d'études souhaité par l'élève"
    )
    
    code_riasec = st.selectbox(
        "🧬 Code RIASEC",
        options=["", "R", "I", "A", "S", "E", "C"],
        help="Code de personnalité RIASEC (Réaliste, Investigateur, Artistique, Social, Entreprenant, Conventionnel)"
    )

# Bouton de prédiction stylisé
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button(
        "🔮 LANCER LA PRÉDICTION", 
        type="primary",
        use_container_width=True
    )

# Traitement de la prédiction
if predict_button:
    if all([ville, serie, souhait, code_riasec]) and serie != "" and code_riasec != "":
        try:
            # Préparer les données d'entrée
            input_data = pd.DataFrame({
                'Ville': [ville],
                'Série': [serie],
                'souhait d\'orientation': [souhait],
                'Code RIASEC': [code_riasec]
            })
            
            # Faire la prédiction
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]
            max_proba = max(proba) * 100
            
            # Créer un DataFrame des probabilités
            proba_df = pd.DataFrame({
                'Classe': metadata['target_classes'],
                'Probabilité': proba * 100
            }).sort_values('Probabilité', ascending=False)
            
            # Afficher les résultats
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("""
                <div style='text-align: center;'>
                    <h2 style='color: #667eea; font-size: 36px;'>
                        ✨ RÉSULTAT DE LA PRÉDICTION ✨
                    </h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Cartes métriques
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>CLASSE PRÉDITE</div>
                        <div class='metric-value'>{prediction}</div>
                        <div class='metric-label'>Confiance: {max_proba:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Graphique en barres horizontales avec Plotly
            st.markdown("<h3 style='color: #667eea; text-align: center;'>📊 Probabilités par classe</h3>", unsafe_allow_html=True)
            
            fig = go.Figure()
            
            # Définir les couleurs avec un dégradé
            colors = px.colors.sequential.Viridis_r
            
            fig.add_trace(go.Bar(
                y=proba_df['Classe'],
                x=proba_df['Probabilité'],
                orientation='h',
                marker=dict(
                    color=proba_df['Probabilité'],
                    colorscale='Viridis',
                    line=dict(color='white', width=2)
                ),
                text=proba_df['Probabilité'].apply(lambda x: f'{x:.2f}%'),
                textposition='outside',
                textfont=dict(size=14, color='#333', family='Arial Black'),
                hovertemplate='<b>%{y}</b><br>Probabilité: %{x:.2f}%<extra></extra>'
            ))
            
            fig.update_layout(
                title=dict(
                    text='',
                    font=dict(size=24, color='#667eea', family='Arial Black')
                ),
                xaxis=dict(
                    title='Probabilité (%)',
                    title_font=dict(size=16, color='#666'),
                    tickfont=dict(size=12),
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.05)'
                ),
                yaxis=dict(
                    title='',
                    tickfont=dict(size=14, family='Arial Black'),
                    autorange='reversed'
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='white',
                height=500,
                margin=dict(l=80, r=120, t=50, b=50),
                font=dict(family='Arial', color='#333')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Graphique en camembert (pie chart) pour le top 5
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<h3 style='color: #667eea; text-align: center;'>🥧 Répartition Top 5</h3>", unsafe_allow_html=True)
                
                top5_df = proba_df.head(5)
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=top5_df['Classe'],
                    values=top5_df['Probabilité'],
                    hole=.4,
                    marker=dict(
                        colors=px.colors.sequential.Plasma_r,
                        line=dict(color='white', width=3)
                    ),
                    textinfo='label+percent',
                    textfont=dict(size=14, family='Arial Black'),
                    hovertemplate='<b>%{label}</b><br>%{value:.2f}%<extra></extra>'
                )])
                
                fig_pie.update_layout(
                    showlegend=True,
                    height=400,
                    paper_bgcolor='white',
                    font=dict(family='Arial', size=12),
                    legend=dict(
                        orientation="v",
                        yanchor="middle",
                        y=0.5,
                        xanchor="left",
                        x=1.05
                    )
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.markdown("<h3 style='color: #667eea; text-align: center;'>📈 Tableau détaillé</h3>", unsafe_allow_html=True)
                
                # Styliser le dataframe
                styled_df = proba_df.copy()
                styled_df['Probabilité'] = styled_df['Probabilité'].apply(lambda x: f'{x:.2f}%')
                styled_df.index = range(1, len(styled_df) + 1)
                
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=400,
                    column_config={
                        "Classe": st.column_config.TextColumn(
                            "Classe",
                            help="Classe d'orientation",
                            width="medium",
                        ),
                        "Probabilité": st.column_config.TextColumn(
                            "Probabilité",
                            help="Pourcentage de probabilité",
                            width="medium",
                        )
                    }
                )
            
            # Indicateur de confiance
            st.markdown("<br>", unsafe_allow_html=True)
            
            if max_proba >= 70:
                confidence_color = "#10b981"
                confidence_text = "EXCELLENTE"
                confidence_icon = "🌟"
            elif max_proba >= 50:
                confidence_color = "#f59e0b"
                confidence_text = "BONNE"
                confidence_icon = "⭐"
            else:
                confidence_color = "#ef4444"
                confidence_text = "FAIBLE"
                confidence_icon = "⚠️"
            
            st.markdown(f"""
                <div style='text-align: center; padding: 20px; background: white; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.08);'>
                    <span style='font-size: 24px;'>{confidence_icon}</span>
                    <span style='color: {confidence_color}; font-size: 24px; font-weight: bold; margin-left: 10px;'>
                        Confiance {confidence_text}
                    </span>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f" Erreur lors de la prédiction: {e}")
    else:
        st.warning(" Veuillez remplir tous les champs avant de lancer la prédiction")

# Section d'aide en bas de page
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")

with st.expander("ℹSignification des classes d'orientation"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **C** : Cycle court
        - **EEP** : Enseignement et Formation Professionnelle
        - **G0, G1, G2, G3, G4** : Différents niveaux d'enseignement général
        """)
    
    with col2:
        st.markdown("""
        - **ND** : Non Déterminé
        - **UP** : Université Publique
        - **V** : Vie active
        """)

with st.expander("🧬 Code RIASEC - Guide"):
    st.markdown("""
    - **R (Réaliste)** : Préférence pour les activités physiques et manuelles
    - **I (Investigateur)** : Goût pour la recherche et la résolution de problèmes
    - **A (Artistique)** : Intérêt pour les activités créatives et artistiques
    - **S (Social)** : Orientation vers l'aide et l'accompagnement des autres
    - **E (Entreprenant)** : Aptitude pour le leadership et la gestion
    - **C (Conventionnel)** : Préférence pour les tâches organisées et structurées
    """)

# Footer
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p style='font-size: 14px;'>
             Système de Prédiction d'Orientation Scolaire | 
            Propulsé par <strong>XGBoost</strong> et <strong>Streamlit</strong>
        </p>
    </div>
""", unsafe_allow_html=True)