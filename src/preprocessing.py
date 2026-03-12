import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def create_master_table(demo_df, diet_df, quest_df):

    # Definire Target
    target_data = quest_df[quest_df['DIQ010'].isin([1, 2])].copy()
    target_data['diabetes_target'] = target_data['DIQ010'].map({2: 0, 1: 1})

    # Merge Demographics + Target
    master_df = pd.merge(demo_df[['SEQN', 'RIAGENDR', 'RIDAGEYR']],
                         target_data[['SEQN', 'diabetes_target']], on='SEQN')

    # Agregare Dieta
    daily_nutrition = diet_df.groupby('SEQN').agg({
        'DR1TKCAL': 'sum', 'DR1TSUGR': 'sum', 'DR1TFIBE': 'sum'
    }).reset_index()

    # Final Merge
    master_df = pd.merge(master_df, daily_nutrition, on='SEQN')
    return master_df


def clean_and_split(master_df):

    # Outliers
    master_df = master_df[(master_df['DR1TKCAL'] >= 500) & (master_df['DR1TKCAL'] <= 8000)]

    # Imputare și Encoding
    master_df['RIAGENDR'] = master_df['RIAGENDR'].map({1: 0, 2: 1})
    master_df = master_df.fillna(master_df.median())

    # Split
    X = master_df.drop(columns=['SEQN', 'diabetes_target'])
    y = master_df['diabetes_target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scaling
    scaler = StandardScaler()
    cols = ['RIDAGEYR', 'DR1TKCAL', 'DR1TSUGR', 'DR1TFIBE']
    X_train[cols] = scaler.fit_transform(X_train[cols])
    X_test[cols] = scaler.transform(X_test[cols])

    return X_train, X_test, y_train, y_test