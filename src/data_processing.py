import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split

# Procesado a raw
df = pd.read_excel('/Users/manuelfranceschi/Desktop/clases_thebridge/ONLINE_DS_THEBRIDGE_MANUEL/Proyectos/ProyectoML-prov/data/raw/Patients Data ( Used for Heart Disease Prediction ).xlsx', index_col=0)

# Al ser tantos registros que tienen accidente cerebral 0, vamos a descartar el 90% para poder trabajar y balancear el modelo
condition_rows = df[df['HadStroke'] == 0]
rows_to_drop_count = int(0.93 * len(condition_rows))
drop_indices = condition_rows.sample(n=rows_to_drop_count, random_state=1).index
df = df.drop(drop_indices)

df.drop(columns=['CovidPos', 'HighRiskLastYear', 'TetanusLast10Tdap', 'PneumoVaxEver', 'FluVaxLast12', 'HIVTesting', 'ChestScan', 'ECigaretteUsage', 'DifficultyErrands', 'DifficultyDressingBathing', 'DifficultyWalking','DifficultyConcentrating','BlindOrVisionDifficulty', 'DeafOrHardOfHearing', 'HadSkinCancer', 'HadAsthma','State'], inplace=True)
df.to_csv('/Users/manuelfranceschi/Desktop/clases_thebridge/ONLINE_DS_THEBRIDGE_MANUEL/Proyectos/ProyectoML-prov/data/raw/healthcare-dataset-raw.csv', index=False)

# Procesado a train y test

df = pd.read_csv('/Users/manuelfranceschi/Desktop/clases_thebridge/ONLINE_DS_THEBRIDGE_MANUEL/Proyectos/ProyectoML-prov/data/raw/healthcare-dataset-raw.csv')
pd.set_option('display.max_columns', 500)

# Feature engineering

df.insert(1, 'gender_num', np.where(df['Sex'] == 'Male',1, 0))

generalHealth_map = {
    "Poor": 5,
    "Fair": 4,
    "Good": 3,
    "Very good": 2,
    "Excellent": 1
}
df.insert(3, 'GeneralHealth_type', df['GeneralHealth'].map(generalHealth_map))

conditions = [
    (df['AgeCategory'].str.contains("18|24|25|29|30|34|35|39")),
    (df['AgeCategory'].str.contains("40|44|45|49|50|54|55|59")),
    (df['AgeCategory'].str.contains("60|64|65|69|70|74|75|79]")),
    (df['AgeCategory'].str.contains("80|older"))
]
values = [1, 2, 3, 4]
df.insert(5, 'AgeGroup', np.select(conditions, values))

diabetes_map = {
    "No": 0,
    "Yes, but only during pregnancy (female)": 1,
    "No, pre-diabetes or borderline diabetes": 2,
    "Yes": 3
}
df.insert(df.columns.get_loc('HadDiabetes') + 1, 'diabetesGroup', df['HadDiabetes'].map(diabetes_map))

smoker_map = {
    "Never smoked": 0,
    "Former smoker": 1,
    "Current smoker - now smokes some days": 2,
    "Current smoker - now smokes every day": 3
}
df.insert(df.columns.get_loc('SmokerStatus') + 1, 'SmokerGroup', df['SmokerStatus'].map(smoker_map))

race_ethnicity_map = {
    "White only, Non-Hispanic": 1,
    "Other race only, Non-Hispanic": 1,
    "Hispanic": 2,
    "Multiracial, Non-Hispanic": 3,
    "Black only, Non-Hispanic": 4
}
df.insert(df.columns.get_loc('RaceEthnicityCategory') + 1, 'raceEthnicityGroup', df['RaceEthnicityCategory'].map(race_ethnicity_map))

bins = [0,18.5, 25, 30, 40, float('inf')] 
etiquetas = [0, 1, 2, 3, 4]
df.insert(df.columns.get_loc('BMI') + 1, 'BmiRange', pd.cut(df['BMI'], bins=bins, labels=etiquetas, right=False).astype(int))

# Guardado de CSV ya procesado
df.to_csv('/Users/manuelfranceschi/Desktop/clases_thebridge/ONLINE_DS_THEBRIDGE_MANUEL/Proyectos/ProyectoML-prov/data/processed/healthcare-dataset-processed.csv', index=False)

# Separación de train y test
train,test = train_test_split(df, test_size=0.20, random_state=42)
train.to_csv('/Users/manuelfranceschi/Desktop/clases_thebridge/ONLINE_DS_THEBRIDGE_MANUEL/Proyectos/ProyectoML-prov/data/train/train.csv', index=False)
test.to_csv('/Users/manuelfranceschi/Desktop/clases_thebridge/ONLINE_DS_THEBRIDGE_MANUEL/Proyectos/ProyectoML-prov/data/test/test.csv', index=False)