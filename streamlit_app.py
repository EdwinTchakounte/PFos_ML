import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="PFos ML",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS ultra moderne et épuré
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: #0f0f0f;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Header */
    .header {
        text-align: center;
        padding: 4rem 0 3rem 0;
    }
    
    .logo {
        font-size: 2.8rem;
        font-weight: 700;
        color: #fff;
        letter-spacing: -0.04em;
        margin-bottom: 0.75rem;
    }
    
    .logo span {
        color: #3b82f6;
    }
    
    .subtitle {
        color: #a1a1aa;
        font-size: 1rem;
        font-weight: 400;
    }
    
    /* Container */
    .container {
        max-width: 480px;
        margin: 0 auto;
        padding: 0 1.5rem;
    }
    
    /* Inputs modernes */
    .stTextInput > div > div > input {
        background: #1a1a1a !important;
        border: 1px solid #27272a !important;
        border-radius: 12px !important;
        color: #fff !important;
        padding: 1rem 1.25rem !important;
        font-size: 0.95rem !important;
        transition: all 0.2s !important;
    }
    
    .stTextInput > div > div > input:focus {
        background: #1f1f1f !important;
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #52525b !important;
    }
    
    .stTextInput > label {
        color: #e4e4e7 !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        margin-bottom: 0.5rem !important;
    }
    
    .stTextInput {
        margin-bottom: 1.25rem !important;
    }
    
    /* Bouton */
    .stButton > button {
        background: #3b82f6 !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 1rem 2rem !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        margin-top: 0.5rem !important;
        transition: all 0.2s !important;
    }
    
    .stButton > button:hover {
        background: #2563eb !important;
        transform: translateY(-1px);
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.5) !important;
    }
    
    /* Résultat */
    .result {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin: 3rem auto 2rem auto;
        text-align: center;
        max-width: 480px;
    }
    
    .result-label {
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
    }
    
    .result-value {
        color: #fff;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 1rem;
    }
    
    .result-confidence {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    /* Chart container */
    .chart-section {
        background: #1a1a1a;
        border: 1px solid #27272a;
        border-radius: 16px;
        padding: 2rem 1.5rem;
        margin: 2rem auto;
        max-width: 480px;
    }
    
    .chart-title {
        color: #e4e4e7;
        font-size: 0.875rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    /* Messages */
    .stAlert {
        background: #1a1a1a !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 12px !important;
        color: #60a5fa !important;
        padding: 1rem !important;
    }
    
    .stWarning {
        border-color: #f59e0b !important;
        color: #fbbf24 !important;
    }
    
    .stError {
        border-color: #ef4444 !important;
        color: #f87171 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #52525b;
        font-size: 0.8rem;
        padding: 4rem 0 2rem 0;
    }
    
    /* Plotly */
    .js-plotly-plot {
        background: transparent !important;
    }
    
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 3rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Charger le modèle et les encodeurs
@st.cache_resource
def load_model_and_encoders():
    try:
        model = joblib.load('model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        le_target = joblib.load('le_target.pkl')
        
        with open('metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return model, label_encoders, le_target, metadata
    except FileNotFoundError as e:
        st.error(f"Fichier manquant: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Erreur: {e}")
        st.stop()

try:
    model, label_encoders, le_target, metadata = load_model_and_encoders()
except Exception as e:
    st.error(f"Erreur fatale: {e}")
    st.stop()

# Header
st.markdown("""
    <div class='header'>
        <div class='logo'>PFos <span>ML</span></div>
        <div class='subtitle'>Prédiction d'orientation scolaire</div>
    </div>
""", unsafe_allow_html=True)

# Container des inputs
st.markdown("<div class='container'>", unsafe_allow_html=True)

college = st.text_input(
    "Collège",
    placeholder="Collège Bilingue Diderot"
)

ville = st.text_input(
    "Ville",
    placeholder="Yaoundé"
)

serie = st.text_input(
    "Série",
    placeholder="A, C, D, ESF, ESP"
)

souhait = st.text_input(
    "Souhait d'orientation",
    placeholder="Sciences, Lettres, Ingénierie"
)

code_riasec = st.text_input(
    "Code RIASEC",
    placeholder="R, I, A, S, E, C",
    help="R: Réaliste | I: Investigateur | A: Artistique | S: Social | E: Entreprenant | C: Conventionnel"
)

predict_button = st.button("Prédire")

st.markdown("</div>", unsafe_allow_html=True)

# Traitement de la prédiction
if predict_button:
    if all([college, ville, serie, souhait, code_riasec]):
        try:
            # Préparer les données
            sample_data = [college, ville, serie, souhait, code_riasec]
            columns = ['Collège', 'Ville', 'Série', 'souhait d\'orientation', 'Code RIASEC']
            
            # Encoder les données
            sample_encoded = []
            unknown_values = []
            
            for i, col in enumerate(columns):
                if col in label_encoders:
                    try:
                        encoded_val = label_encoders[col].transform([sample_data[i]])[0]
                        sample_encoded.append(encoded_val)
                    except ValueError:
                        unknown_values.append(f"{col}: '{sample_data[i]}'")
                        sample_encoded.append(-1)
                else:
                    st.error(f"Encodeur manquant pour: {col}")
                    st.stop()
            
            if unknown_values:
                st.warning("Certaines valeurs sont inconnues. Précision réduite.")
            
            # Convertir et prédire
            sample_array = np.array(sample_encoded).reshape(1, -1)
            prediction_encoded = model.predict(sample_array)[0]
            proba = model.predict_proba(sample_array)[0]
            prediction = le_target.inverse_transform([prediction_encoded])[0]
            max_proba = max(proba) * 100
            
            # Afficher le résultat
            st.markdown(f"""
                <div class='result'>
                    <div class='result-label'>Orientation recommandée</div>
                    <div class='result-value'>{prediction}</div>
                    <div class='result-confidence'>
                        <span>●</span>
                        <span>{max_proba:.1f}% de confiance</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Créer le DataFrame des probabilités
            proba_df = pd.DataFrame({
                'Classe': le_target.classes_,
                'Probabilité': proba * 100
            }).sort_values('Probabilité', ascending=False).head(5)
            
            # Graphique minimaliste
            st.markdown("<div class='chart-section'>", unsafe_allow_html=True)
            st.markdown("<div class='chart-title'>TOP 5 ORIENTATIONS</div>", unsafe_allow_html=True)
            
            fig = go.Figure()
            
            colors = ['#3b82f6' if i == 0 else '#27272a' for i in range(len(proba_df))]
            
            fig.add_trace(go.Bar(
                y=proba_df['Classe'],
                x=proba_df['Probabilité'],
                orientation='h',
                marker=dict(color=colors, line=dict(width=0)),
                text=proba_df['Probabilité'].apply(lambda x: f'{x:.0f}%'),
                textposition='outside',
                textfont=dict(size=12, color='#a1a1aa', family='Inter', weight=600),
                hovertemplate='%{y}<br>%{x:.1f}%<extra></extra>'
            ))
            
            fig.update_layout(
                xaxis=dict(
                    showgrid=False,
                    showticklabels=False,
                    zeroline=False,
                    range=[0, max(proba_df['Probabilité']) * 1.2]
                ),
                yaxis=dict(
                    autorange='reversed',
                    tickfont=dict(size=13, color='#a1a1aa', family='Inter')
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=280,
                margin=dict(l=120, r=60, t=0, b=0),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {e}")
    else:
        st.warning("Veuillez remplir tous les champs")

# Footer
st.markdown("<div class='footer'>PFos ML © 2025</div>", unsafe_allow_html=True)