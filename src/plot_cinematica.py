from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def genera_grafici_cinematici_pdf(config):
    """Genera i PDF cinematici (serie storica + zoom pre-rottura) per ciascun sensore."""
    out_dir = Path(config.output_dir)
    plot_dir = out_dir / "grafici_cinematici_pdf"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Caricamento dati
    df_s = pd.read_parquet(out_dir / "spostamenti_trazione.parquet")
    df_v = pd.read_parquet(out_dir / "velocita.parquet")
    df_inv = pd.read_parquet(out_dir / "inverso_velocita.parquet")

    df_s = df_s - df_s.iloc[0]

    # Date di riferimento umane
    t_crollo = df_v.index[-1]
    t_zoom = t_crollo - pd.Timedelta(days=1)

    for sensore in df_inv.columns:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), layout="constrained")

        # ---------------------------------------------------------
        # 1. SERIE STORICA (Spostamento vs Velocità)
        # ---------------------------------------------------------
        ax1_v = ax1.twinx()
        ax1.plot(
            df_s.index,
            df_s[sensore],
            color="gray",
            lw=2,
            label=r"Spostamento $\Delta s$ [mm]",
        )
        ax1_v.plot(
            df_v.index,
            df_v[sensore],
            color="crimson",
            lw=1.2,
            alpha=0.7,
            label=r"Velocità $v$ [mm/giorno]",
        )

        ax1.set_title(
            f"Evoluzione Cinematica Globale - {sensore}", fontsize=12, fontweight="bold"
        )
        ax1.set_ylabel(r"Spostamento $\Delta s$ [mm]", color="dimgray")
        ax1_v.set_ylabel(r"Velocità $v$ [mm/giorno]", color="crimson")
        ax1.grid(True, ls=":", alpha=0.5)

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax1_v.get_legend_handles_labels()
        ax1_v.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.9)

        # ---------------------------------------------------------
        # 2. ZOOM PRE-ROTTURA (Velocità vs 1/Velocità)
        # ---------------------------------------------------------
        ax2_inv = ax2.twinx()
        v_zoom = df_v[sensore].loc[t_zoom:]
        inv_zoom = df_inv[sensore].loc[t_zoom:]

        ax2.plot(
            v_zoom.index,
            v_zoom,
            color="crimson",
            lw=1.8,
            label=r"Velocità $v$ [mm/giorno]",
        )
        ax2_inv.scatter(
            inv_zoom.index,
            inv_zoom,
            color="royalblue",
            s=12,
            alpha=0.8,
            label=r"$1/v$ [giorni/mm]",
        )

        ax2_inv.axhline(0, color="black", alpha=0.8, lw=0.8, ls="--")
        ax2.axvline(
            t_crollo,
            color="black",
            alpha=0.8,
            lw=0.8,
            ls="--",
            label=f"Rottura ({t_crollo:%d/%m %H:%M})",
        )

        ax2.set_title(
            "Correlazione $v$ e $1/v$ nella fase pre-rottura",
            fontsize=12,
            fontweight="bold",
        )
        ax2.set_ylabel(r"Velocità $v$ [mm/giorno]", color="crimson")
        ax2_inv.set_ylabel(r"$1/v$ [giorni/mm]", color="royalblue")
        ax2.grid(True, ls=":", alpha=0.5)

        h3, l3 = ax2.get_legend_handles_labels()
        h4, l4 = ax2_inv.get_legend_handles_labels()
        ax2_inv.legend(h3 + h4, l3 + l4, loc="upper left", framealpha=0.9)

        # ---------------------------------------------------------
        # FORMATTAZIONE DATE E SALVATAGGIO
        # ---------------------------------------------------------

        # Forza le date visibili su entrambi gli assi e applica la inclinazione
        for ax in [ax1, ax2]:
            ax.tick_params(axis="x", labelbottom=True)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        fig.savefig(plot_dir / f"cinematica_{sensore}.pdf")
        plt.show()

        print(f"PDF salvato: cinematica_{sensore}.pdf")
