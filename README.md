ITALIANO

# Analisi e Prevenzione delle Frodi Bancarie: un Approccio Multidimensionale.

## Descrizione del progetto

Questo progetto nasce con l'obiettivo di sviluppare un modello di machine learning per la rilevazione e prevenzione delle frodi bancarie.
Utilizzando tecniche avanzate di data preprocessing, trasformazione dei dati non numerici, bilanciamento delle classi (SMOTE),
e una, seppur automatizzata e ancora preliminare (senz'altro ottimizzabile), operazione di feature engineering, insieme all'impiego di algoritmi di classificazione come Random Forest,
il progetto mira a ottenere un modello efficace e computazionalmente ottimizzato. 

L'obiettivo primario è dimostrare che, con un approccio metodico e organizzato, è possibile ottenere un modello altamente performante con un costo computazionale relativamente basso,
pur lasciando spazio a ulteriori ottimizzazioni future.


## Dataset - "Bank Account Fraud Dataset Suite (NeurIPS 2022)"

- **Dimensioni:** Il dataset di addestramento contiene 1.000.000 di righe e 50 colonne (dopo la fase di feature engineering), rappresentanti variabili numeriche, categoriche e alcune derivate.
- **Bilanciamento:** La variabile target `fraud_bool` era fortemente sbilanciata, con una distribuzione di circa 96% di transazioni legittime e 4% di transazioni fraudolente.
- **Caratteristiche principali:**
  - **Realistico**, basato su dati reali per il rilevamento delle frodi.
  - **Sbilanciato**, con una bassa prevalenza della classe positiva (frode).
  - **Dinamico**, con dati temporali che riflettono distribuzioni mutevoli.
  - **Privacy preserving**, grazie all'applicazione di tecniche di privacy differenziale (aggiunta di rumore), codifica delle feature e utilizzo di un modello generativo (CTGAN).
- **Bilanciamento finale:** Il bilanciamento delle classi è stato ottenuto tramite l'algoritmo SMOTE, portando il dataset bilanciato a 1.979.742 righe.

LINK TO DATAFRAME: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022


## Struttura del progetto

Il progetto è organizzato come segue:

```
ANALISI_FRODI/
├── data/
│   ├── raw/                          # Dati grezzi
│   ├── processed/                    # Dati preprocessati
│       ├── processed_data_cleaned.csv
│       ├── processed_data_derived_features.csv
│       ├── processed_data_resampled.csv
├── notebooks/
│   ├── data_cleaning.ipynb           # Notebook per il preprocessing iniziale
│   ├── EDA.ipynb                     # Analisi esplorativa dei dati
│   ├── RandomForest.ipynb            # Addestramento e validazione del modello Random Forest
│   ├── SMOTE.ipynb                   # Applicazione di SMOTE e bilanciamento delle classi
├── src/
│   ├── __init__.py                   # File di inizializzazione del modulo
│   ├── func.py                       # Funzioni utilizzate nei notebook
├── .gitignore                        # File per ignorare file/cartelle non necessari
├── environment.yml                   # Specifica delle dipendenze dell'ambiente
├── README.md                         # Documentazione del progetto
```

## Risultati

Il modello Random Forest ha raggiunto prestazioni elevate:
- **ROC-AUC medio:** 0.9991
- **Precisione media:** 0.9921
- **Recall media:** 0.9921
- **F1-score medio:** 0.9921

Questi risultati sono stati ottenuti grazie a un bilanciamento accurato dei dati (SMOTE) e una validazione incrociata con Starfield, garantendo un modello robusto e generalizzabile.

## Dipendenze

Le dipendenze del progetto sono specificate nel file `environment.yml`. Per ricreare l'ambiente:
```bash
conda env create -f environment.yml
```


## Come eseguire il progetto

1. **Preprocessing dei dati:**
   - Eseguire il notebook `data_cleaning.ipynb` per pulire i dati.
   - Eseguire il notebook `EDA.ipynb` per l'analisi esplorativa.

2. **Bilanciamento dei dati:**
   - Utilizzare `SMOTE.ipynb` per applicare SMOTE e salvare i dati bilanciati.

3. **Addestramento del modello:**
   - Eseguire `RandomForest.ipynb` per addestrare e validare il modello Random Forest.


## Ulteriori sviluppi

Sebbene il progetto abbia raggiunto risultati notevoli, sono certamente identificabili alcuni possibili miglioramenti per future iterazioni:

- **Bayesian Optimization:** Ottimizzare ulteriormente i parametri del modello con tecniche avanzate di ricerca degli iperparametri.
- **Confronto con altri algoritmi:** Implementare e valutare modelli come XGBoost e LightGBM per un confronto delle prestazioni.
- **Data Augmentation avanzata:** Utilizzare GAN o tecniche simili per generare dati sintetici più realistici e diversificati.
- **Estensione del dataset:** Integrare ulteriori dati per migliorare la generalizzazione del modello.

Con più tempo e risorse, questi miglioramenti potrebbero portare a un sistema ancora più accurato e robusto per la rilevazione delle frodi bancarie.

---

Questo progetto rappresenta un punto di partenza affidabile per affrontare il complesso problema delle frodi bancarie, dimostrando che con un metodo ben strutturato è possibile ottenere risultati significativi bilanciando prestazioni e costi computazionali. Ad ogni modo, il machine learning è un terreno in continua evoluzione, che offre costantemente nuove opportunità per migliorare e ottimizzare, anche laddove i risultati sembrano già molto soddisfacenti.  Questo progetto non è solo una tappa, ma un invito a spingersi oltre, a ottimizzare ulteriormente e a esplorare soluzioni innovative per portare il modello a livelli di accuratezza ancora più alti.










ENGLISH

# Banking Fraud Analysis and Prevention: A Multidimensional Approach


## Project Description

This project was designed with the goal of developing a machine learning model to detect and prevent banking fraud.
By leveraging advanced data preprocessing techniques, class balancing (SMOTE), and classification algorithms such as Random Forest,
the project aims to achieve an effective and computationally optimized model.

The primary objective is to demonstrate that, with a methodical and organized approach, it is possible to achieve a highly performant model
with relatively low computational costs, while leaving room for further future optimizations.

Additionally, this project integrates the transformation of non-numerical data and an initial, automated operation of feature engineering, involving the creation of derived features.


## Dataset - "Bank Account Fraud Dataset Suite (NeurIPS 2022)"

- **Size:** The training dataset contains 1,000,000 rows and 50 columns (after feature engineering), representing numerical, categorical, and some derived variables.
- **Class Imbalance:** The target variable `fraud_bool` was highly imbalanced, with approximately 96% legitimate transactions and 4% fraudulent transactions.
- **Key Features:**
  - **Realistic**, based on real-world data for fraud detection.
  - **Imbalanced**, with a low prevalence of the positive class (fraud).
  - **Dynamic**, including temporal data that reflects changing distributions.
  - **Privacy preserving**, achieved through differential privacy techniques (noise addition), feature encoding, and the use of a generative model (CTGAN).
- **Final Balancing:** Class balancing was achieved using the SMOTE algorithm, resulting in a balanced dataset with 1,979,742 rows.

LINK TO DATAFRAME: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022


## Project Structure

The project is organized as follows:

```
ANALISI_FRODI/
├── data/
│   ├── raw/                          # Raw data
│   ├── processed/                    # Processed data
│       ├── processed_data_cleaned.csv
│       ├── processed_data_derived_features.csv
│       ├── processed_data_resampled.csv
├── notebooks/
│   ├── data_cleaning.ipynb           # Notebook for initial preprocessing
│   ├── EDA.ipynb                     # Exploratory Data Analysis
│   ├── RandomForest.ipynb            # Training and validation of the Random Forest model
│   ├── SMOTE.ipynb                   # Application of SMOTE and class balancing
├── src/
│   ├── __init__.py                   # Module initialization file
│   ├── func.py                       # Functions used in the notebooks
├── .gitignore                        # File to ignore unnecessary files/folders
├── environment.yml                   # Environment dependency specification
├── README.md                         # Project documentation
```


## Results

The Random Forest model achieved excellent performance:
- **Mean ROC-AUC:** 0.9991
- **Mean Precision:** 0.9921
- **Mean Recall:** 0.9921
- **Mean F1-score:** 0.9921

These results were obtained through careful data balancing (SMOTE) and cross-validation with Starfield, ensuring a robust and generalizable model.


## Dependencies

The project dependencies are specified in the `environment.yml` file. To recreate the environment:
```bash
conda env create -f environment.yml
```


## How to Execute the Project

1. **Data Preprocessing:**
   - Run the `data_cleaning.ipynb` notebook to clean the data.
   - Run the `EDA.ipynb` notebook for exploratory analysis.

2. **Data Balancing:**
   - Use `SMOTE.ipynb` to apply SMOTE and save the balanced data.

3. **Model Training:**
   - Run `RandomForest.ipynb` to train and validate the Random Forest model.


## Further Developments

While the project achieved remarkable results, some possible improvements have been identified for future iterations:

- **Bayesian Optimization:** Further optimize model parameters with advanced hyperparameter search techniques.
- **Comparison with Other Algorithms:** Implement and evaluate models like XGBoost and LightGBM for performance comparison.
- **Advanced Data Augmentation:** Use GANs or similar techniques to generate more realistic and diversified synthetic data.
- **Dataset Expansion:** Integrate additional data to improve the model's generalization.

With more time and resources, these improvements could lead to an even more accurate and robust system for detecting banking fraud.

---

This project serves as a reliable starting point for tackling the complex challenge of banking fraud, demonstrating that with a well-structured approach, it is possible to achieve significant results while balancing performance and computational costs. That said, machine learning is an ever-evolving field, constantly offering new opportunities for improvement and optimization, even when the results already seem highly satisfactory. This project is not just a milestone but an invitation to go further, to optimize even more, and to explore innovative solutions to elevate the model to even greater levels of accuracy.
