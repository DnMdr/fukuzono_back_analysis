import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from .config import ConfigurazioneAnalisiFukuzono, PlotConfig


def plot_previsione(
    config: ConfigurazioneAnalisiFukuzono,
    df_risultati: pd.DataFrame,
    plot_config: PlotConfig,
) -> None:

    os.makedirs(config.output_dir, exist_ok=True)

    # 1. AZZERAMENTO SPOSTAMENTO CUMULATO SUL PRIMO RECORD DEL DATASET
    df_spostamenti_trazione = pd.read_parquet("output/spostamenti_trazione.parquet")
    df_spostamenti_cumulato = df_spostamenti_trazione - df_spostamenti_trazione.iloc[0]

    colori = {ore: plt.cm.tab10(i) for i, ore in enumerate(config.finestre_analisi_ore)}

    sensori = df_risultati.sensore.unique()

    for sensore in sensori:
        fig, ax1 = plt.subplots(figsize=(14, 8))

        # ------------------------------------------------------------------
        # 1. RETTE DI RIFERIMENTO CROLLO REALE
        # ------------------------------------------------------------------
        ax1.axhline(
            y=config.last_measure,
            color="black",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
        )

        ax1.axvline(
            x=config.last_measure,
            color="black",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label=f"Rottura ({config.last_measure.strftime('%d/%m %H:%M')})",
        )

        # ------------------------------------------------------------------
        # 2. PLOT PREVISIONI (ASSE Y PRINCIPALE - SINISTRA)
        # ------------------------------------------------------------------
        df_filtrato = df_risultati.query(
            f"sensore == @sensore and r2 >= {plot_config.SOGLIA_R2} and previsione_crollo.notnull()"
        )

        for ore in config.finestre_analisi_ore:
            df_plot = df_filtrato.query("finestra_h == @ore")

            if not df_plot.empty:
                ax1.scatter(
                    df_plot.index,
                    df_plot.previsione_crollo,
                    label=f"Finestra {ore}h",
                    color=colori[ore],
                    s=15,
                    alpha=0.7,
                    edgecolors="none",
                )

        ax1.set_title(
            f"Convergenza e Spostamento Cumulato - {sensore}\n(Previsioni filtrate per R² > {plot_config.SOGLIA_R2})",
            fontsize=14,
            pad=15,
        )
        ax1.set_xlabel(r"Istante di analisi (Tempo $t$)", fontsize=12)
        ax1.set_ylabel(r"Data di crollo prevista ($t_f$)", fontsize=12)

        ax1.set_ylim(
            config.last_measure - pd.Timedelta(days=plot_config.ZOOM_Y_GIORNI),
            config.last_measure + pd.Timedelta(days=plot_config.ZOOM_Y_GIORNI),
        )
        ax1.set_xlim(
            config.last_measure - pd.Timedelta(days=plot_config.ZOOM_X_GIORNI_INDIETRO),
            config.last_measure + pd.Timedelta(hours=plot_config.ZOOM_X_ORE_AVANTI),
        )

        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
        ax1.yaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, linestyle=":", alpha=0.6)

        # ------------------------------------------------------------------
        # 3. PLOT SPOSTAMENTO CUMULATO Δs (ASSE Y SECONDARIO - DESTRA)
        # ------------------------------------------------------------------
        ax2 = ax1.twinx()
        ax2.grid(False)

        start_time = config.last_measure - pd.Timedelta(
            days=plot_config.ZOOM_X_GIORNI_INDIETRO
        )
        end_time = config.last_measure + pd.Timedelta(
            hours=plot_config.ZOOM_X_ORE_AVANTI
        )
        mask_tempo = (df_spostamenti_cumulato.index >= start_time) & (
            df_spostamenti_cumulato.index <= end_time
        )

        df_spost_plot = df_spostamenti_cumulato.loc[mask_tempo]

        if not df_spost_plot.empty and sensore in df_spost_plot.columns:
            spostamento_cum = df_spost_plot[sensore]

            ax2.plot(
                df_spost_plot.index,
                spostamento_cum,
                color="#555555",
                linestyle="-",
                linewidth=2.5,
                alpha=0.4,
                label="Spostamento Δs [mm]",
            )

            # --- ANNOTAZIONE VALORE Δs AL CROLLO ---
            if config.last_measure in spostamento_cum.index:
                spost_finale = spostamento_cum.loc[config.last_measure]
            else:
                spost_finale = spostamento_cum.dropna().iloc[-1]

            # Marker sulla retta di crollo reale
            ax2.scatter(
                config.last_measure,
                spost_finale,
                color="#333333",
                s=65,
                zorder=5,
            )

            # Box con freccia
            ax2.annotate(
                f"Δs al crollo = {spost_finale:.2f} mm",
                xy=(config.last_measure, spost_finale),
                xytext=(-140, -45),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color="#333333",
                bbox={
                    "boxstyle": "round,pad=0.4",
                    "facecolor": "#f8f9fa",
                    "alpha": 0.5,
                    "edgecolor": "#888888",
                },
                arrowprops={
                    "arrowstyle": "->",
                    "connectionstyle": "arc3,rad=-0.15",
                    "color": "#333333",
                    "lw": 1.5,
                },
            )

        ax2.set_ylabel(
            "Spostamento Δs [mm]",
            fontsize=12,
            color="#555555",
            rotation=270,
            labelpad=20,
        )
        ax2.tick_params(axis="y", colors="#555555")

        # ------------------------------------------------------------------
        # 4. UNIFICAZIONE LEGENDE
        # ------------------------------------------------------------------
        handles_1, labels_1 = ax1.get_legend_handles_labels()
        handles_2, labels_2 = ax2.get_legend_handles_labels()

        ax1.legend(
            handles_1 + handles_2,
            labels_1 + labels_2,
            loc="upper left",
            markerscale=1.5,
            framealpha=0.9,
            facecolor="white",
            edgecolor="none",
        )

        plt.tight_layout()

        percorso_file = os.path.join(config.output_dir, f"{sensore}_convergenza.pdf")
        fig.savefig(percorso_file, dpi=300, bbox_inches="tight")
        plt.close()
