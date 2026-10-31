import numpy as np
import pandas as pd


# ==========================================
# MOTORE MATEMATICO (Fukuzono e OLS)
# ==========================================
def _inizializza_spazio_temporale(
    serie_inv_vel_sensore: pd.Series,
) -> pd.DataFrame:
    """Converte l'indice datetime in un asse continuo di secondi trascorsi dall'inizio."""
    inizio_acquisizione = serie_inv_vel_sensore.index[0]

    df_spazio_fasi = serie_inv_vel_sensore.to_frame(name="inv_vel")
    df_spazio_fasi["tempo_sec"] = (
        df_spazio_fasi.index - inizio_acquisizione
    ).total_seconds()

    return df_spazio_fasi


def _calcola_momenti_statistici(df_spazio_fasi: pd.DataFrame, finestra_h: int) -> dict:
    """
    Calcola le grandezze statistiche necessarie per la regressione lineare (OLS)
    utilizzando una finestra temporale mobile (rolling window).
    """
    # min_periods=2 per calcolare una varianza o una covarianza (gradi di libertà N-1).
    finestra = df_spazio_fasi.rolling(f"{finestra_h}h", min_periods=2)

    return {
        "media_tempo": finestra["tempo_sec"].mean(),
        "media_inv_vel": finestra["inv_vel"].mean(),
        "varianza_tempo": finestra["tempo_sec"].var(),
        "varianza_inv_vel": finestra["inv_vel"].var(),
        "covarianza_mista": finestra["tempo_sec"].cov(df_spazio_fasi["inv_vel"]),
        "punti_utilizzati": finestra["inv_vel"].count(),
    }


def _calcola_pendenza(
    covarianza_mista: pd.Series, varianza_tempo: pd.Series
) -> pd.Series:
    """Formula OLS: m = Cov(t, 1/v) / Var(t)"""
    return covarianza_mista / varianza_tempo


def _calcola_intercetta(
    media_inv_vel: pd.Series, pendenza: pd.Series, media_tempo: pd.Series
) -> pd.Series:
    """Formula OLS: q = media(1/v) - (m * media(t))"""
    return media_inv_vel - (pendenza * media_tempo)


def _calcola_r2(
    covarianza_mista: pd.Series, varianza_tempo: pd.Series, varianza_inv_vel: pd.Series
) -> pd.Series:
    """Formula OLS: R^2 = Cov(t, 1/v)^2 / (Var(t) * Var(1/v))"""
    return (covarianza_mista**2) / (varianza_tempo * varianza_inv_vel)


def _calcola_intersezione_zero(pendenza: pd.Series, intercetta: pd.Series) -> pd.Series:
    """
    Estrapolazione Fukuzono: t_crollo = -alpha / beta (istante in cui 1/v tocca lo zero)
    con alpha = q intercetta e beta = m coeff. angolare
    sostituisce lo zero con NaN per evitare divisione per zero
    la pendenza può essere 0 se sensori non misurano accelerazione
    """
    return -intercetta / pendenza.replace(0, np.nan)


# ==========================================
# CRITERI DI PREVISIONE
# ==========================================
def _verifica_condizione_accelerazione(pendenza: pd.Series) -> pd.Series:
    """CRITERIO 1: La parete deve essere in accelerazione (pendenza di 1/v negativa)"""
    return pendenza < 0


def _verifica_crollo_nel_futuro(
    istante_crollo_sec: pd.Series, tempo_attuale_sec: pd.Series
) -> pd.Series:
    """CRITERIO 2: L'istante di rottura previsto (t_f) deve essere successivo al momento dell'analisi (t_0)"""
    return istante_crollo_sec > tempo_attuale_sec


def _verifica_crollo_entro_limite_imposto(
    istante_crollo_sec: pd.Series, tempo_attuale_sec: pd.Series, max_giorni: int
) -> pd.Series:
    """CRITERIO 3: Il crollo deve avvenire entro il limite imposto (7gg)"""
    max_secondi_ammessi = pd.Timedelta(days=max_giorni).total_seconds()
    return (istante_crollo_sec - tempo_attuale_sec) <= max_secondi_ammessi


# ==========================================
# CONVERSIONI TEMPORALI
# ==========================================
def _converti_secondi_in_data_assoluta(
    secondi_da_inizio: pd.Series, inizio_acquisizione: pd.Timestamp
) -> pd.Series:
    """Converte l'asse in secondi in date assolute"""
    delta_temporale = pd.to_timedelta(secondi_da_inizio, unit="s")
    return (inizio_acquisizione + delta_temporale).dt.round("s")


# ==========================================
# ANALISI SINGOLO SENSORE
# ==========================================
def _analizza_singolo_sensore(
    serie_inv_vel_sensore: pd.Series, finestra_h: int, max_giorni_previsione: int
) -> pd.DataFrame:
    """
    Esegue l'algoritmo completo su una singola serie storica,
    combinando matematica OLS e filtri fisici.
    """
    # 1. Inizializzazione e statistica
    df_spazio_fasi = _inizializza_spazio_temporale(serie_inv_vel_sensore)
    inizio_acquisizione = df_spazio_fasi.index[0]

    momenti_statistici = _calcola_momenti_statistici(df_spazio_fasi, finestra_h)

    # 2. Matematica del ordinary least squares
    pendenza = _calcola_pendenza(
        momenti_statistici["covarianza_mista"], momenti_statistici["varianza_tempo"]
    )

    intercetta = _calcola_intercetta(
        momenti_statistici["media_inv_vel"], pendenza, momenti_statistici["media_tempo"]
    )

    r2 = _calcola_r2(
        momenti_statistici["covarianza_mista"],
        momenti_statistici["varianza_tempo"],
        momenti_statistici["varianza_inv_vel"],
    )

    # 3. Previsione pura
    istante_crollo_sec = _calcola_intersezione_zero(pendenza, intercetta)

    # 4. Maschere geotecniche
    mask_accelera = _verifica_condizione_accelerazione(pendenza)

    mask_futuro = _verifica_crollo_nel_futuro(
        istante_crollo_sec, df_spazio_fasi["tempo_sec"]
    )

    mask_orizzonte = _verifica_crollo_entro_limite_imposto(
        istante_crollo_sec, df_spazio_fasi["tempo_sec"], max_giorni_previsione
    )

    # 5. Filtraggio del dato fisicamente ammissibile
    maschera_globale_valida = mask_accelera & mask_futuro & mask_orizzonte
    istanti_validi_sec = istante_crollo_sec.where(maschera_globale_valida)

    previsione_crollo = _converti_secondi_in_data_assoluta(
        istanti_validi_sec, inizio_acquisizione
    )

    # 6. Costruzione dataset risultati per la serie
    return pd.DataFrame(
        {
            "r2": r2,
            "punti_utilizzati": momenti_statistici["punti_utilizzati"],
            "pendenza": pendenza,
            "intercetta": intercetta,
            "previsione_crollo": previsione_crollo,
        },
        index=serie_inv_vel_sensore.index,
    )


# ==========================================
# HELPER DI FORMATTAZIONE
# ==========================================
def _formatta_risultati_finali(df_grezzo: pd.DataFrame) -> pd.DataFrame:
    """Riordina il df finale e ottimizza i tipi delle colonne"""
    colonne_ordinate = [
        "sensore",
        "finestra_h",
        "r2",
        "punti_utilizzati",
        "pendenza",
        "intercetta",
        "previsione_crollo",
    ]

    tipi_ottimizzati = {
        "sensore": "category",
        "finestra_h": "int8",
        "r2": "float32",
        "punti_utilizzati": "UInt16",
    }

    return df_grezzo[colonne_ordinate].astype(tipi_ottimizzati)


# ==========================================
# PUBLIC
# ==========================================
def esegui_fukuzono_globale(df_inverso_velocita: pd.DataFrame, config) -> pd.DataFrame:
    """Applica il metodo di Fukuzono all'intero dataset"""
    lista_risultati = []

    for sensore in df_inverso_velocita.columns:
        serie_inv_vel_sensore = df_inverso_velocita[sensore]

        for finestra_h in config.finestre_analisi_ore:
            df_risultati_sensore = _analizza_singolo_sensore(
                serie_inv_vel_sensore=serie_inv_vel_sensore,
                finestra_h=finestra_h,
                max_giorni_previsione=config.max_giorni_previsione,
            )

            df_risultati_sensore["sensore"] = sensore
            df_risultati_sensore["finestra_h"] = finestra_h
            lista_risultati.append(df_risultati_sensore)

    df_finale = pd.concat(lista_risultati)
    return _formatta_risultati_finali(df_finale)
