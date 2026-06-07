"""
Application Streamlit principale — immo-dvf-prediction.
"""

import streamlit as st
import pandas as pd

from src.models.predict import ImmoPricePredictor, AppartementInput
from src.agents.parser import AnnonceParser
from src.visualization.plots import (
    plot_prix_distribution,
    plot_prix_par_quartier,
    plot_evolution_prix,
    plot_comparaison_annonce,
)
from src.data.load import load_dvf_features
from src.utils.logging import logger

# -------------------------------------------------------------------
# Configuration de la page
# -------------------------------------------------------------------
st.set_page_config(
    page_title="immo-dvf-prediction",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 immo-dvf-prediction")
st.caption("Analyse et prédiction de prix des appartements à Angers — données DVF + ML + Agents IA")

# -------------------------------------------------------------------
# Chargement du modèle (mis en cache pour la session)
# -------------------------------------------------------------------
@st.cache_resource
def get_predictor() -> ImmoPricePredictor:
    return ImmoPricePredictor()


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_dvf_features()


# -------------------------------------------------------------------
# Navigation
# -------------------------------------------------------------------
mode = st.sidebar.radio(
    "Navigation",
    ["📊 Explorer le marché", "🔎 Analyser une annonce", "🔮 Saisie manuelle"],
)

# -------------------------------------------------------------------
# SECTION 1 — Explorer le marché
# -------------------------------------------------------------------
if mode == "📊 Explorer le marché":
    st.header("Marché immobilier Angers — Appartements")

    try:
        df = get_data()
    except FileNotFoundError as e:
        st.error(f"⚠️ {e}")
        st.stop()

    # Métriques globales
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ventes", f"{len(df):,}")
    col2.metric("Prix médian", f"€{df['prix_m2'].median():,.0f}/m²")
    col3.metric("Prix moyen", f"€{df['prix_m2'].mean():,.0f}/m²")
    col4.metric("Surface médiane", f"{df['surface_m2'].median():.0f} m²")

    # Filtres
    st.sidebar.subheader("Filtres")
    annees = sorted(df["annee_vente"].unique(), reverse=True)
    annee_sel = st.sidebar.multiselect("Année(s)", annees, default=annees)
    df_filtered = df[df["annee_vente"].isin(annee_sel)] if annee_sel else df

    # Graphiques
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(plot_prix_distribution(df_filtered), width="stretch")
    with col_right:
        if "nom_iris" in df_filtered.columns:
            st.plotly_chart(plot_prix_par_quartier(df_filtered), width="stretch")
        else:
            st.info("Données IRIS non disponibles (lancer make data-full).")

    st.plotly_chart(plot_evolution_prix(df_filtered), width="stretch")

    # Tableau stats par quartier
    if "nom_iris" in df_filtered.columns:
        st.subheader("📍 Prix par quartier (IRIS)")
        stats = (
            df_filtered.groupby("nom_iris")["prix_m2"]
            .agg(moyenne="mean", mediane="median", nb_ventes="count")
            .round(0)
            .sort_values("mediane", ascending=False)
            .reset_index()
        )
        st.dataframe(stats, width="stretch")

# -------------------------------------------------------------------
# SECTION 2 — Analyser une annonce (agent IA)
# -------------------------------------------------------------------
elif mode == "🔎 Analyser une annonce":
    st.header("Analyser une annonce immobilière")
    st.info(
        "🤖 Coller l'URL d'une annonce SeLoger, Bien'Ici ou LeBonCoin. "
        "L'agent IA extrait les caractéristiques et compare le prix au marché."
    )

    url = st.text_input("🔗 URL de l'annonce")

    if url and st.button("🔍 Analyser", type="primary"):
        with st.spinner("Extraction des caractéristiques via agent IA..."):
            try:
                parser = AnnonceParser()
                features = parser.parse_url(url)
            except ValueError as e:
                st.error(f"⚠️ {e}")
                st.stop()

        st.subheader("📋 Caractéristiques extraites")
        col1, col2, col3 = st.columns(3)
        col1.metric("Surface", f"{features.surface_m2 or 'N/A'} m²")
        col2.metric("Pièces", features.nombre_pieces or "N/A")
        col3.metric("Prix annonce", f"€{features.prix_annonce:,.0f}" if features.prix_annonce else "N/A")

        col4, col5, col6 = st.columns(3)
        col4.metric("Classe DPE", features.dpe_classe or "N/A")
        col5.metric("Année construction", features.annee_construction or "N/A")
        col6.metric("Charges copro", f"€{features.charges_copro:,.0f}/mois" if features.charges_copro else "N/A")

        # Prédiction
        if features.surface_m2 and features.nombre_pieces:
            with st.spinner("Prédiction du prix de marché..."):
                predictor = get_predictor()
                appart_input = parser.to_appart_input(features)
                result = predictor.predict(appart_input)

            st.subheader("📊 Comparaison avec le marché")

            if features.prix_annonce:
                prix_m2_annonce = features.prix_annonce / features.surface_m2
                ecart_pct = (prix_m2_annonce - result.prix_m2_predit) / result.prix_m2_predit * 100

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Prix annonce", f"€{features.prix_annonce:,.0f}", f"€{prix_m2_annonce:,.0f}/m²")
                col_b.metric("Prix marché prédit", f"€{result.prix_total_predit:,.0f}", f"€{result.prix_m2_predit:,.0f}/m²")
                col_c.metric("Écart", f"{ecart_pct:+.1f}%")

                if ecart_pct > 10:
                    st.warning(f"⚠️ Prix surévalué de {ecart_pct:.1f}% par rapport au marché")
                elif ecart_pct < -10:
                    st.success(f"✅ Prix sous-évalué de {abs(ecart_pct):.1f}% — Bonne affaire potentielle !")
                else:
                    st.info(f"📊 Prix cohérent avec le marché ({ecart_pct:+.1f}%)")

                st.plotly_chart(
                    plot_comparaison_annonce(prix_m2_annonce, result.prix_m2_predit),
                    width="stretch",
                )
        else:
            st.warning("Surface ou nombre de pièces non détectés. Utiliser la saisie manuelle.")

# -------------------------------------------------------------------
# SECTION 3 — Saisie manuelle
# -------------------------------------------------------------------
elif mode == "🔮 Saisie manuelle":
    st.header("Estimer le prix d'un appartement")

    col1, col2 = st.columns(2)
    with col1:
        surface = st.number_input("Surface (m²)", min_value=10, max_value=200, value=55)
        pieces = st.number_input("Nombre de pièces", min_value=1, max_value=8, value=3)
        nb_lots = st.number_input("Lots dans la copropriété", min_value=1, max_value=200, value=20)
    with col2:
        dpe = st.selectbox("Classe DPE", ["Non renseigné", "A", "B", "C", "D", "E", "F", "G"])
        annee_construction = st.number_input("Année de construction", min_value=1850, max_value=2026, value=1980)
        prix_annonce = st.number_input("Prix de l'annonce (€) — optionnel", min_value=0, value=0)

    if st.button("🔮 Estimer le prix", type="primary"):
        from src.data.features import DPE_ORDINAL, ANNEE_REF

        dpe_ordinal = DPE_ORDINAL.get(dpe) if dpe != "Non renseigné" else None
        age_bien = ANNEE_REF - annee_construction

        appart = AppartementInput(
            surface_m2=surface,
            nombre_pieces=pieces,
            nb_lots_copro=nb_lots,
            dpe_ordinal=dpe_ordinal,
            dpe_bonus=int(dpe in ["A", "B"]) if dpe != "Non renseigné" else None,
            dpe_malus=int(dpe in ["F", "G"]) if dpe != "Non renseigné" else None,
            age_bien=age_bien,
        )

        predictor = get_predictor()
        result = predictor.predict(appart)

        st.subheader("📊 Résultat")
        col_a, col_b = st.columns(2)
        col_a.metric("Prix total prédit", f"€{result.prix_total_predit:,.0f}")
        col_b.metric("Prix au m² prédit", f"€{result.prix_m2_predit:,.0f}/m²")

        if prix_annonce > 0:
            prix_m2_annonce = prix_annonce / surface
            st.plotly_chart(
                plot_comparaison_annonce(prix_m2_annonce, result.prix_m2_predit),
                width="stretch",
            )
