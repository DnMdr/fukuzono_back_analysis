from dataclasses import dataclass

import pandas as pd


@dataclass
class ConfigurazioneAnalisiFukuzono:
    """
    Configurazioni per il metodo di fukuzono
    """

    # --- DIRECTORY E FILE ---
    input_file: str
    output_dir: str

    # --- SAVE INTERMEDIO ---
    save_path: bool

    # --- STRUTTURA DATI ---
    colonna_timestamp: list[str]
    colonne_sensori: list[str]

    # --- LIMITI TEMPORALI ---
    last_measure: str | pd.Timestamp
    start_analysis: str | pd.Timestamp

    # --- PARAMETRI GEOTECNICI E PREDITTIVI ---
    velocita_stazionaria: float
    max_giorni_previsione: int
    finestre_analisi_ore: list[int]

    def __post_init__(self) -> None:
        """
        Converte le stringhe in oggetti datetime di Pandas
        """
        self.last_measure = pd.to_datetime(self.last_measure)
        self.start_analysis = pd.to_datetime(self.start_analysis)


@dataclass
class PlotConfig:
    SOGLIA_R2: float
    ZOOM_Y_GIORNI: int
    ZOOM_X_GIORNI_INDIETRO: int
    ZOOM_X_ORE_AVANTI: int
