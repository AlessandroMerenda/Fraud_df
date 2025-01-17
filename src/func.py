import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV




df = pd.DataFrame()  #placeholder vuoto




def identify_missing_values(df, potential_missing_values):
    """
    Identifica e calcola la percentuale di valori mancanti personalizzati in ogni colonna del DataFrame.

    Args:
        df (pd.DataFrame): Il DataFrame da analizzare.
        potential_missing_values (list): Lista di valori da considerare come mancanti.

    Returns:
        pd.DataFrame: Un DataFrame con la percentuale di valori mancanti per colonna.
    """
    # Dizionario per memorizzare la percentuale di valori mancanti per ogni colonna
    missing_custom = {}
    
    # Itera su ogni colonna del DataFrame
    for col in df.columns:
        # Conta il numero di valori mancanti personalizzati nella colonna
        count = df[col].isin(potential_missing_values).sum()
         # Se ci sono valori mancanti, calcola la percentuale e aggiungila al dizionario
        if count > 0:
            missing_custom[col] = count / len(df) * 100
    
    # Crea un DataFrame per il sommario dei valori mancanti
    missing_custom_summary = pd.DataFrame({
        'Colonna': missing_custom.keys(),
        'Percentuale Valori Mancanti (%)': missing_custom.values()
    }).sort_values(by='Percentuale Valori Mancanti (%)', ascending=False)
    
    return missing_custom_summary

# Identificazione e gestione dei valori mancanti
potential_missing_values = [-1]
missing_custom_summary = identify_missing_values(df, potential_missing_values)




def handle_missing_values(df, missing_custom_summary, threshold_high=70, threshold_low=1):
    """
    Elimina le colonne con una percentuale di valori mancanti superiore a threshold_high
    e imputa le colonne con una percentuale di valori mancanti inferiore a threshold_low.

    Args:
        df (pd.DataFrame): Il DataFrame da processare.
        missing_custom_summary (pd.DataFrame): Il sommario dei valori mancanti.
        threshold_high (float): La soglia superiore per l'eliminazione delle colonne.
        threshold_low (float): La soglia inferiore per l'imputazione delle colonne.

    Returns:
        pd.DataFrame: Il DataFrame processato.
    """
    # Eliminazione delle colonne con troppi valori mancanti
    columns_to_drop = missing_custom_summary[missing_custom_summary['Percentuale Valori Mancanti (%)'] > threshold_high]['Colonna']
    df.drop(columns=columns_to_drop, inplace=True)

    # Imputazione delle colonne con pochi valori mancanti
    columns_to_impute = missing_custom_summary[missing_custom_summary['Percentuale Valori Mancanti (%)'] < threshold_low]['Colonna']
    for col in columns_to_impute:
        df[col] = df[col].replace(-1, df[col].mean())

    return df




def encode_categorical_features(df):
    """
    Codifica le variabili categoriche nel DataFrame utilizzando One-Hot Encoding e Label Encoding.

    Args:
        df (pd.DataFrame): Il DataFrame da codificare.

    Returns:
        pd.DataFrame: Il DataFrame codificato.
    """
    # Codifica con One-Hot Encoding per variabili nominali
    df = pd.get_dummies(df, columns=['payment_type', 'employment_status', 'housing_status', 'device_os'], drop_first=True)

    # Codifica con Label Encoding per 'source'
    le = LabelEncoder()
    df['source'] = le.fit_transform(df['source'])

    return df




def generate_derived_features(df, target_col='fraud_bool', threshold=0.1):
    """
    Genera feature derivate dalle colonne numeriche del DataFrame e calcola la loro correlazione con la variabile target.

    Args:
        df (pd.DataFrame): Il DataFrame originale.
        target_col (str): Il nome della variabile target.
        threshold (float): La soglia per considerare una correlazione significativa.

    Returns:
        pd.DataFrame: Un DataFrame con le feature derivate e le loro correlazioni significative.
    """
    # Seleziona le colonne numeriche
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col != target_col]  # Escludi la variabile target

    # Liste per salvare le nuove feature e le loro correlazioni
    new_features = []
    correlations = []

    # Generazione automatica di feature derivate
    for i, col1 in enumerate(numerical_cols):
        for j, col2 in enumerate(numerical_cols):
            if i >= j:  # Evita duplicati e combinazioni inverse (A+B e B+A)
                continue

            # Operazioni matematiche
            try:
                # Somma
                feature_name = f'{col1}_plus_{col2}'
                new_feature = df[col1] + df[col2]
                corr = new_feature.corr(df[target_col])
                if abs(corr) > threshold:  # Verifica solo se supera la soglia
                    new_features.append(feature_name)
                    correlations.append(corr)

                # Differenza
                feature_name = f'{col1}_minus_{col2}'
                new_feature = df[col1] - df[col2]
                corr = new_feature.corr(df[target_col])
                if abs(corr) > threshold:
                    new_features.append(feature_name)
                    correlations.append(corr)

                # Prodotto
                feature_name = f'{col1}_times_{col2}'
                new_feature = df[col1] * df[col2]
                corr = new_feature.corr(df[target_col])
                if abs(corr) > threshold:
                    new_features.append(feature_name)
                    correlations.append(corr)

                # Rapporto (gestendo divisioni per zero)
                feature_name = f'{col1}_div_{col2}'
                new_feature = df[col1] / (df[col2] + 1e-5)
                corr = new_feature.corr(df[target_col])
                if abs(corr) > threshold:
                    new_features.append(feature_name)
                    correlations.append(corr)

            except Exception as e:
                print(f"Errore nella combinazione {col1} e {col2}: {e}")

    # Crea un DataFrame per le nuove feature e le correlazioni
    significant_features = pd.DataFrame({
        'Feature': new_features,
        'Correlation': correlations
    })

    return significant_features




def add_derived_features(df, significant_features):
    """
    Aggiunge le feature derivate significative al DataFrame originale.

    Args:
        df (pd.DataFrame): Il DataFrame originale.
        significant_features (pd.DataFrame): Il DataFrame con le feature derivate e le loro correlazioni.

    Returns:
        pd.DataFrame: Il DataFrame originale con le feature derivate significative aggiunte.
    """
    # Lista delle feature derivate significative
    derived_features_list = significant_features['Feature'].tolist()

    # Aggiungi tutte le feature derivate al DataFrame originale
    for feature in derived_features_list:
        if feature not in df.columns:
            try:
                # Suddivisione del nome della feature con rsplit
                col1, operation, col2 = feature.rsplit('_', 2)  # Dividi dalla destra
                if operation == 'plus':
                    df[feature] = df[col1] + df[col2]
                elif operation == 'minus':
                    df[feature] = df[col1] - df[col2]
                elif operation == 'times':
                    df[feature] = df[col1] * df[col2]
                elif operation == 'div':
                    df[feature] = df[col1] / (df[col2] + 1e-5)  # Gestione divisioni per zero
            except KeyError as e:
                print(f"Errore: colonna non trovata per la feature {feature}. Dettagli: {e}")
            except Exception as e:
                print(f"Errore durante l'aggiunta della feature {feature}: {e}")

    return df




def plot_class_distribution(y_before, y_after):
    """
    Genera un grafico della distribuzione delle classi prima e dopo SMOTE.

    Args:
        y_before (pd.Series): Distribuzione delle classi prima di SMOTE.
        y_after (pd.Series): Distribuzione delle classi dopo SMOTE.
    """
    # Crea una figura con due sottotrame
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Grafico della distribuzione delle classi prima di SMOTE    axes[0].bar(y_before.index, y_before.values, color='blue')
    axes[0].set_title('Distribuzione delle classi (Prima di SMOTE)')
    axes[0].set_xlabel('Classi')
    axes[0].set_ylabel('Numero di campioni')
    
    # Grafico della distribuzione delle classi dopo SMOTE
    axes[1].bar(y_after.index, y_after.values, color='green')
    axes[1].set_title('Distribuzione delle classi (Dopo SMOTE)')
    axes[1].set_xlabel('Classi')
    axes[1].set_ylabel('Numero di campioni')
    
    # Migliora il layout della figura
    plt.tight_layout()
    plt.show()




def apply_smote(X, y, random_state=42, plot_distribution=False):
    """
    Applica SMOTE per bilanciare le classi nel dataset.

    Args:
        X (pd.DataFrame): Features del dataset.
        y (pd.Series): Target del dataset.
        random_state (int): Random state per SMOTE.
        plot_distribution (bool): Se True, genera un grafico della distribuzione delle classi prima e dopo SMOTE.

    Returns:
        X_resampled (pd.DataFrame): Features bilanciate.
        y_resampled (pd.Series): Target bilanciato.
    """
    # Distribuzione originale delle classi
    class_distribution_before = y.value_counts()

    # Applica SMOTE per bilanciare le classi
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Distribuzione delle classi dopo SMOTE
    class_distribution_after = pd.Series(y_resampled).value_counts()

    # Visualizza la distribuzione delle classi
    if plot_distribution:
        plot_class_distribution(class_distribution_before, class_distribution_after)

    return X_resampled, y_resampled




def train_random_forest(X_train, y_train, random_state=42, **kwargs):
    """
    Addestra un modello Random Forest con i parametri specificati.

    Args:
        X_train (pd.DataFrame): Features di training.
        y_train (pd.Series): Target di training.
        random_state (int): Random state per garantire riproducibilità.
        **kwargs: Altri parametri per il modello Random Forest.

    Returns:
        RandomForestClassifier: Modello addestrato.
    """
    # Creazione del modello Random Forest con i parametri specificati
    rf_model = RandomForestClassifier(random_state=random_state, n_jobs=-1, **kwargs)
    
    # Addestramento del modello
    rf_model.fit(X_train, y_train)
    
    return rf_model




def evaluate_model(model, X_test, y_test):
    """
    Valuta il modello sul test set e ritorna i risultati in formato DataFrame.
    
    Args:
        model: Modello addestrato.
        X_test: Features del test set.
        y_test: Target del test set.
        
    Returns:
        dict: Risultati ROC-AUC e Classification Report.
        pd.DataFrame: Classification Report in formato tabellare.
    """
    from sklearn.metrics import roc_auc_score
    
    # Predizioni del modello
    y_pred = model.predict(X_test)
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred)
    
    # Classification Report
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    
    return {"roc_auc": roc_auc, "classification_report": report_df}




def plot_feature_importance(model, feature_names, top_n=10):
    """
    Visualizza l'importanza delle feature di un modello Random Forest.

    Args:
        model: Modello addestrato.
        feature_names (list): Lista dei nomi delle feature.
        top_n (int): Numero di feature principali da visualizzare.

    Returns:
        None
    """
    # Calcolo dell'importanza delle feature
    feature_importances = pd.Series(model.feature_importances_, index=feature_names)
    
    # Ordinamento delle feature per importanza e selezione delle top_n
    feature_importances = feature_importances.sort_values(ascending=False).head(top_n)

    # Visualizzazione dell'importanza delle feature
    feature_importances.plot(kind='bar', figsize=(10, 6))
    plt.title('Importanza delle Feature')
    plt.ylabel('Importanza')
    plt.xlabel('Feature')
    plt.show()

# Configura logging
logging.basicConfig(level=logging.INFO)




def bayesian_optimization_rf(X, y, n_iter=50, cv_splits=5, random_state=42):
    """
    Esegue la Bayesian Optimization per il modello Random Forest.

    Args:
        X (pd.DataFrame): Features del dataset.
        y (pd.Series): Target del dataset.
        n_iter (int): Numero di iterazioni per la Bayesian Optimization.
        cv_splits (int): Numero di split per la validazione incrociata.
        random_state (int): Random state per la riproducibilità.

    Returns:
        dict: Migliori iperparametri trovati e il modello ottimizzato.
    """
    # Configura il logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Spazio di ricerca per i parametri
        param_grid = {
            'n_estimators': (50, 300),         # Numero di alberi
            'max_depth': (5, 50),             # Profondità massima
            'min_samples_split': (2, 10),     # Campioni minimi per split
            'min_samples_leaf': (1, 10),      # Campioni minimi per foglia
            'max_features': ['sqrt', 'log2', None]  # Numero massimo di feature
        }

        # Modello Random Forest
        rf = RandomForestClassifier(random_state=random_state)

        # Validazione incrociata stratificata
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

        # Bayesian Optimization
        bayes_search = BayesSearchCV(
            estimator=rf,
            search_spaces=param_grid,
            n_iter=n_iter,
            scoring='roc_auc',
            cv=cv,
            n_jobs=-1,
            random_state=random_state,
            verbose=1
        )

        # Esegue l'ottimizzazione
        logging.info("Inizio della Bayesian Optimization...")
        bayes_search.fit(X, y)
        logging.info("Bayesian Optimization completata.")

        # Migliori parametri trovati
        best_params = bayes_search.best_params_
        best_model = bayes_search.best_estimator_

        logging.info(f"Migliori iperparametri trovati: {best_params}")
        return {'best_params': best_params, 'best_model': best_model}

    except Exception as e:
        logging.error(f"Errore durante la Bayesian Optimization: {e}")
        return None




def starfield_with_smote(X, y, model, smote_function, k=5, random_state=42):
    """
    Applica validazione incrociata Starfield con SMOTE.
    
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        model: Modello machine learning da allenare.
        smote_function (function): Funzione SMOTE.
        k (int): Numero di fold per la validazione incrociata.
        random_state (int): Random state per la stratificazione.

    Returns:
        dict: Risultati medi (ROC-AUC, precision, recall, F1-score).
    """
    # Inizializza StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    metrics = {'roc_auc': [], 'precision': [], 'recall': [], 'f1_score': []}

    fold = 1
    for train_index, test_index in skf.split(X, y):
        try:
            logging.info(f"Fold {fold}...")
            
            # Dividi i dati in train e test
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Applica SMOTE sul training set
            X_resampled, y_resampled = smote_function(X_train, y_train)
            if X_resampled is None or y_resampled is None:
                raise ValueError("SMOTE function did not return valid resampled data.")

            # Addestra il modello
            model.fit(X_resampled, y_resampled)

            # Fai predizioni sul test set
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Calcola le metriche
            roc_auc = roc_auc_score(y_test, y_prob)
            report = classification_report(y_test, y_pred, output_dict=True)
            if not report:
                raise ValueError("Classification report generation failed.")

             # Aggiungi le metriche al dizionario
            metrics['roc_auc'].append(roc_auc)
            metrics['precision'].append(report['weighted avg']['precision'])
            metrics['recall'].append(report['weighted avg']['recall'])
            metrics['f1_score'].append(report['weighted avg']['f1-score'])

            logging.info(f"Fold {fold} complete. ROC-AUC: {roc_auc:.4f}")
        except Exception as e:
            logging.error(f"Errore durante il Fold {fold}: {e}")
        fold += 1

    # Calcola le medie delle metriche
    results = {metric: np.mean(values) for metric, values in metrics.items()}
    return results