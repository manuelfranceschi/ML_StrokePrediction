import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

df_train = pd.read_csv('/Users/manuelfranceschi/Desktop/clases_thebridge/ONLINE_DS_THEBRIDGE_MANUEL/Proyectos/ProyectoML-prov/data/train/train.csv')
X_train = df_train[['AgeGroup', 'HadHeartAttack', 'diabetesGroup','SmokerGroup', 'AlcoholDrinkers', 'HadArthritis','HadKidneyDisease', 'HadDepressiveDisorder']]
y_train = df_train['HadStroke']

# Modelo final
rf_model = RandomForestClassifier(n_estimators= 200,min_samples_split=4,max_depth=6,criterion="gini",class_weight="balanced", random_state=42)
rf_model.fit(X_train,y_train)

# Modelo más simple, pero funcional
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_model.fit(X_train, y_train)

with open('/Users/manuelfranceschi/Desktop/clases_thebridge/ONLINE_DS_THEBRIDGE_MANUEL/Proyectos/ProyectoML-prov/models/final_model_rf.pkl', "wb") as archivo_salida:
    pickle.dump(rf_model, archivo_salida)
with open('/Users/manuelfranceschi/Desktop/clases_thebridge/ONLINE_DS_THEBRIDGE_MANUEL/Proyectos/ProyectoML-prov/models/trained_model_lr.pkl', "wb") as archivo_salida:
    pickle.dump(lr_model, archivo_salida)