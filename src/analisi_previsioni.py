import pandas as pd

from src.config import ConfigurazioneAnalisiFukuzono


def _calcola_errori(df: pd.DataFrame, t_crollo_reale: pd.Timestamp) -> pd.DataFrame:
    """Calcola lo scarto temporale dal crollo (in ore) e il suo valore assoluto"""
    delta_ore = (df["previsione_crollo"] - t_crollo_reale) / pd.Timedelta(hours=1)
    return df.assign(errore_ore=delta_ore, errore_assoluto=delta_ore.abs())


def _applica_filtro_r2(
    df: pd.DataFrame, colonna_r2: str = "r2", soglia_minima: float = 0.3
) -> pd.DataFrame:
    """Filtra le previsioni in funzione di una soglia R2 arbitraria"""
    return df[df[colonna_r2] >= soglia_minima]  # type: ignore


def _aggrega_metriche(df: pd.DataFrame) -> pd.DataFrame:
    """Aggrega le metriche statistiche calcolando MedAE, Media e Std con .groupby()"""
    return df.groupby(["sensore", "finestra_h"]).agg(
        previsioni_valide=("r2", "count"),
        medae_ore=("errore_assoluto", "median"),
        errore_medio_ore=("errore_ore", "mean"),
        std_errore=("errore_ore", "std"),
    )  # type: ignore


def _calcola_copertura_percentuale(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola la percentuale di previsioni valide rispetto al massimo teorico"""
    copertura = (df["previsioni_valide"] / df["previsioni_teoriche"]) * 100
    return df.assign(copertura_perc=copertura.round(2))


def analisi_previsioni(
    df_previsioni: pd.DataFrame,
    config: ConfigurazioneAnalisiFukuzono,
    inizio_creep_terziario: str | pd.Timestamp = "2025-11-10 00:00:00",
) -> pd.DataFrame:
    """
    Calcola errore assoluto mediano (MedAE), direzione dell'errore,
    tempo di allertamento e lead time.
    """

    # cast di tipizzazione timestamp
    t_creep = pd.to_datetime(inizio_creep_terziario)

    # isolo il creep terziario
    df_base = df_previsioni[df_previsioni.index >= t_creep]

    # calcolo il numero massimo di previsioni possibili
    df_teoriche = (
        df_base.groupby(["sensore", "finestra_h"])
        .size()
        .to_frame(name="previsioni_teoriche")
    )

    # orchestratore e join delle metriche nel df
    df_analisi = (
        df_base.pipe(_calcola_errori, t_crollo_reale=config.last_measure)  # type: ignore
        .pipe(_applica_filtro_r2)
        .pipe(_aggrega_metriche)
        .join(df_teoriche)
        .pipe(_calcola_copertura_percentuale)
        .reset_index()
    )

    return df_analisi
