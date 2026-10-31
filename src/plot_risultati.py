import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_risultati_finestre(
    df_analisi: pd.DataFrame, output_path: str = "output/analisi_finestre.pdf"
):
    """
    Genera un grafico combinato: MedAE e Copertura in funzione della finestra.
    """
    # GESTIONE COLORI DINAMICI
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    sensori = df_analisi["sensore"].unique()

    colori = sns.color_palette("husl", n_colors=len(sensori))

    palette = dict(zip(sensori, colori))

    # TRASFORMANDO NUMERI IN STRINGHE EVITO SPAZIATURA VALORI SU ASSE X
    df_analisi["finestra_h_str"] = df_analisi["finestra_h"].astype(str)

    # 1. PLOT SUPERIORE: MedAE vs Finestra_h
    sns.lineplot(
        data=df_analisi,
        x="finestra_h_str",
        y="medae_ore",
        hue="sensore",
        marker="o",
        markersize=8,
        linewidth=2.5,
        ax=ax1,
        palette=palette,
    )
    ax1.set_title(
        "Andamento dell'Errore Mediano (MedAE) al variare della finestra di analisi",
        fontweight="bold",
        pad=15,
    )
    ax1.set_ylabel("MedAE [ore]")
    ax1.legend(title="Sensore", loc="upper left")

    # 2. PLOT INFERIORE: Copertura vs Finestra_h
    sns.lineplot(
        data=df_analisi,
        x="finestra_h_str",
        y="copertura_perc",
        hue="sensore",
        marker="s",
        markersize=8,
        linewidth=2.5,
        ax=ax2,
        palette=palette,
        legend=False,
    )
    ax2.set_title(
        "Copertura delle Previsioni Valide (R² > Soglia)", fontweight="bold", pad=15
    )
    ax2.set_ylabel("Copertura [%]")
    ax2.set_xlabel("Finestra di Analisi [ore]")
    finestre_uniche = sorted(df_analisi["finestra_h_str"].unique(), key=int)
    ax2.set_xticks(finestre_uniche)
    ax2.set_xticklabels(finestre_uniche)

    plt.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Grafico salvato con successo in: {output_path}")
    plt.close()
