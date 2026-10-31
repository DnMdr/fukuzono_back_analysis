# %%
from src.config import ConfigurazioneAnalisiFukuzono, PlotConfig

config = ConfigurazioneAnalisiFukuzono(
    input_file="data/dati_lanzada_geo3.csv",
    output_dir="output",
    save_path=True,
    colonna_timestamp=["time"],
    colonne_sensori=[
        "fessurimetro_f1",
        "fessurimetro_f2",
        "fessurimetro_f3",
        "fessurimetro_f4",
    ],
    last_measure="2025-11-11 08:00:00",
    start_analysis="2025-11-05 08:00:00",
    velocita_stazionaria=0.02,
    max_giorni_previsione=7,
    finestre_analisi_ore=[2, 4, 5, 6, 7, 8, 10, 24, 48],
)

plot_config = PlotConfig(
    SOGLIA_R2=0.3, ZOOM_Y_GIORNI=2, ZOOM_X_GIORNI_INDIETRO=1, ZOOM_X_ORE_AVANTI=1
)

# %% LOAD DATASET E 1/V
from src.dataframe_init import preparazione_dataset

df_inverso_velocita = preparazione_dataset(config)

# %% ANALISI FUKUZONO
from src.fukuzono_method import esegui_fukuzono_globale

df_previsioni = esegui_fukuzono_globale(df_inverso_velocita, config)

# %% GRAFICI DI PLOT
from src.plot_previsione import plot_previsione

plot_previsione(config, df_previsioni, plot_config)

# %% ANALISI QUANTITATIVA PREVISIONI
import pandas as pd

from src.analisi_previsioni import analisi_previsioni

inizio_creep_terziario = pd.to_datetime("2025-11-10 00:00:00")
df_analisi_previsioni = analisi_previsioni(
    df_previsioni, config, inizio_creep_terziario
)

# %%
from src.plot_risultati import plot_risultati_finestre

plot_risultati_finestre(df_analisi_previsioni)
