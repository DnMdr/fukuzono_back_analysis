# %%
from src.config import ConfigurazioneAnalisiFukuzono, PlotConfig

config = ConfigurazioneAnalisiFukuzono(
    input_file="data/dati_lanzada_geo3.csv",
    output_dir="output",
    save_path=True,
    verbose=True,
    colonna_timestamp=["time"],
    colonne_sensori=["F1", "F2", "F3", "F4"],
    last_measure="2025-11-11 08:00:00",
    start_analysis="2025-11-05 08:00:00",
    velocita_stazionaria=0.02,
    max_giorni_previsione=7,
    finestre_analisi_ore=[2 * 1, 2 * 2, 2 * 3, 2 * 4, 2 * 5, 2 * 6, 2 * 12, 2 * 24],
)

plot_config = PlotConfig(
    SOGLIA_R2=0.3, ZOOM_Y_GIORNI=2, ZOOM_X_GIORNI_INDIETRO=1, ZOOM_X_ORE_AVANTI=1
)

# %% LOAD DATASET E 1/V
from src.dataframe_init import preparazione_dataset

df_inverso_velocita = preparazione_dataset(config)

if config.verbose:
    print("\n--- DEBUG: Preparazione Dataset (1/v) ---")
    print(f"Dimensioni dataframe: {df_inverso_velocita.shape}")
    print(f"Sensori sopravvissuti ai filtri:\n{df_inverso_velocita.columns.tolist()}")
    print("Conteggio valori validi per sensore (non-NaN):")
    print(df_inverso_velocita.notna().sum())
    print("Anteprima dati (1/v):")
    print(df_inverso_velocita.head())


# %% ANALISI FUKUZONO
from src.fukuzono_method import esegui_fukuzono_globale

df_previsioni = esegui_fukuzono_globale(df_inverso_velocita, config)

if config.verbose:
    print("\n--- DEBUG: Analisi Fukuzono Globale ---")
    print(f"Dimensioni dataframe previsioni: {df_previsioni.shape}")

    if "previsione_crollo" in df_previsioni.columns:
        tot_previsioni = df_previsioni["previsione_crollo"].notna().sum()
        print(f"Totale date di crollo stimate valide: {tot_previsioni}")

# %% GRAFICI DI PLOT
from src.plot_previsione import plot_previsione

plot_previsione(config, df_previsioni, plot_config)

if config.verbose:
    print("\n--- DEBUG: Plot Previsioni generati con successo ---")


# %% ANALISI QUANTITATIVA PREVISIONI
import pandas as pd

from src.analisi_previsioni import analisi_previsioni

inizio_creep_terziario = pd.to_datetime("2025-11-10 00:00:00")
df_analisi_previsioni = analisi_previsioni(
    df_previsioni, config, inizio_creep_terziario
)


# %% PLOT RISULTATI FINESTRE
from src.plot_risultati import plot_risultati_finestre

plot_risultati_finestre(df_analisi_previsioni)
