# PREVISIONE INV IN BACK ANALYSIS

Pipeline di calcolo in back analysis per il metodo dell'inverso della velocità (Fukuzono - Voight), completamente vettorializzato con pandas.

Prende in ingresso un file .csv contenente i dati di spostamento con struttura:

> time,sensore_1,...,sensore_n

Installa con pip le librerie richieste:

> pip install requirements.txt

L'output sono diversi dataframe contenenti previsione per rolling window ed analisi statistiche sulle previsioni effettuate per la calibrazione della finestra e del sensore ottimale.
