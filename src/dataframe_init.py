from pathlib import Path

import pandas as pd

from .config import ConfigurazioneAnalisiFukuzono


# ==========================================
# PRIVATE
# ==========================================
def _carica_dati(config: ConfigurazioneAnalisiFukuzono) -> pd.DataFrame:
    """Carica i dati da file .csv"""
    colonne_da_leggere = config.colonna_timestamp + config.colonne_sensori

    return pd.read_csv(
        config.input_file,
        parse_dates=config.colonna_timestamp,
        index_col=config.colonna_timestamp[0],
        usecols=colonne_da_leggere,
    )


def _taglia_dati(
    df: pd.DataFrame, config: ConfigurazioneAnalisiFukuzono
) -> pd.DataFrame:
    """Taglia un dataframe alla data di crollo"""
    return df.loc[: config.last_measure].copy()


def _interpola_dati(df: pd.DataFrame, config: ConfigurazioneAnalisiFukuzono):
    """Interpolazione temporale delle serie di dati per gestione buchi"""
    return df[config.colonne_sensori].interpolate(method="time").round(2)


def _calcola_velocita(
    df_spostamenti: pd.DataFrame, config: ConfigurazioneAnalisiFukuzono
) -> pd.DataFrame:
    """Calcolo velocità con derivata prima dello spostamento"""
    delta_giorni = df_spostamenti.index.to_series().diff() / pd.Timedelta(days=1)
    return df_spostamenti.diff().div(delta_giorni, axis=0).bfill()


def _filtra_sensori_trazione(df_spostamenti: pd.DataFrame) -> pd.DataFrame:
    """
    scarta tutti i sensori in compressione
    confrontando se l'ultimo valore di spostamento
    in una serie è maggiore del primo
    """
    return df_spostamenti.loc[
        :, df_spostamenti.iloc[-1] > df_spostamenti.iloc[0]
    ].copy()


def _estrai_creep_terziario(
    df_velocita: pd.DataFrame, config: ConfigurazioneAnalisiFukuzono
) -> pd.DataFrame:
    """
    Taglia al creep terziario
    in realtà taglia ad una data prefissata nei config
    visto che i dataset sono offline
    """
    return df_velocita.loc[config.start_analysis :].copy()


def _calcola_velocita_inversa(
    df_tagliato: pd.DataFrame, config: ConfigurazioneAnalisiFukuzono
) -> pd.DataFrame:
    """
    Calcola il reciproco della velocità per i valori
    superiori a config.velocita_stazionaria
    """
    return df_tagliato.where(df_tagliato > config.velocita_stazionaria).rdiv(1)


# ==========================================
# PUBLIC
# ==========================================
def preparazione_dataset(config: ConfigurazioneAnalisiFukuzono) -> pd.DataFrame:
    """Prepara il dataset di velocità inversa per l'analisi Fukuzono"""

    # 1. Pipeline di calcolo
    df = _carica_dati(config)
    df = _taglia_dati(df, config)
    df = _interpola_dati(df, config)
    df_trazione = _filtra_sensori_trazione(df)
    df_velocita = _calcola_velocita(df_trazione, config)
    df_creep_terziario = _estrai_creep_terziario(df_velocita, config)
    df_inverso_velocita = _calcola_velocita_inversa(df_creep_terziario, config)

    # 2. Salvataggio
    if getattr(config, "save_path", False) and config.output_dir:
        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        df_trazione.to_parquet(out_dir / "spostamenti_trazione.parquet")
        df_velocita.to_parquet(out_dir / "velocita.parquet")
        df_inverso_velocita.to_parquet(out_dir / "inverso_velocita.parquet")

    return df_inverso_velocita
