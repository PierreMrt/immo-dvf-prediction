"""
Graphiques et visualisations du marché immobilier Angers.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_prix_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogramme de distribution des prix au m²."""
    fig = px.histogram(
        df,
        x="prix_m2",
        nbins=50,
        title="Distribution des prix au m² — Appartements Angers",
        labels={"prix_m2": "Prix (€/m²)"},
        color_discrete_sequence=["#1f77b4"],
    )
    fig.add_vline(
        x=df["prix_m2"].median(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Médiane : {df['prix_m2'].median():,.0f} €/m²",
    )
    return fig


def plot_prix_par_quartier(df: pd.DataFrame) -> go.Figure:
    """Boxplot prix au m² par quartier IRIS."""
    fig = px.box(
        df.sort_values("prix_m2"),
        x="nom_iris",
        y="prix_m2",
        title="Prix au m² par quartier (IRIS) — Appartements Angers",
        labels={"prix_m2": "Prix (€/m²)", "nom_iris": "Quartier"},
        color="nom_iris",
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-45)
    return fig


def plot_evolution_prix(df: pd.DataFrame) -> go.Figure:
    """Evolution des prix médians au m² par trimestre."""
    df = df.copy()
    df["trimestre"] = pd.to_datetime(df["date_vente"]).dt.to_period("Q").astype(str)
    evolution = df.groupby("trimestre")["prix_m2"].median().reset_index()

    fig = px.line(
        evolution,
        x="trimestre",
        y="prix_m2",
        title="Évolution du prix médian au m² — Appartements Angers",
        labels={"prix_m2": "Prix médian (€/m²)", "trimestre": "Trimestre"},
        markers=True,
    )
    return fig


def plot_comparaison_annonce(
    prix_m2_annonce: float, prix_m2_predit: float, nom_quartier: str = ""
) -> go.Figure:
    """
    Graphique comparant le prix de l'annonce avec le prix prédit.

    Args:
        prix_m2_annonce: Prix au m² demandé dans l'annonce
        prix_m2_predit: Prix au m² prédit par le modèle
        nom_quartier: Nom du quartier (pour le titre)
    """
    ecart_pct = (prix_m2_annonce - prix_m2_predit) / prix_m2_predit * 100
    couleur_annonce = "#e74c3c" if ecart_pct > 5 else ("#2ecc71" if ecart_pct < -5 else "#f39c12")

    fig = go.Figure(
        data=[
            go.Bar(name="Prix annonce", x=["Annonce"], y=[prix_m2_annonce], marker_color=couleur_annonce),
            go.Bar(name="Prix marché prédit", x=["Marché"], y=[prix_m2_predit], marker_color="#3498db"),
        ]
    )
    titre = f"Comparaison prix annonce vs marché{' — ' + nom_quartier if nom_quartier else ''}"
    fig.update_layout(
        title=titre,
        yaxis_title="Prix (€/m²)",
        barmode="group",
        annotations=[
            dict(
                x=0.5, y=1.05, xref="paper", yref="paper",
                text=f"Écart : {ecart_pct:+.1f}% vs marché",
                showarrow=False,
                font=dict(size=14, color=couleur_annonce),
            )
        ],
    )
    return fig
