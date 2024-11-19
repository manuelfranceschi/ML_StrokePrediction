import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score
def show_metrics(model,x_test, y_test):
    if isinstance(model, GridSearchCV) or isinstance(model, RandomizedSearchCV) :
        print('Best estimator: ',model.best_estimator_)
        print('Best params: ',model.best_params_)
        print('Best score: ',model.best_score_)
        
    y_pred = model.predict(x_test)
    print(f"{type(model).__name__}:")
    print('accuracy_score', accuracy_score(y_test,y_pred))
    print('precision_score', precision_score(y_test,y_pred))
    print('recall_score', recall_score(y_test,y_pred))
    print('f1_score', f1_score(y_test,y_pred))
    print('confusion matrix \n', confusion_matrix(y_test,y_pred))
    print("\n")

lr_model = pickle.load(open('/Users/manuelfranceschi/Desktop/clases_thebridge/ONLINE_DS_THEBRIDGE_MANUEL/Proyectos/ProyectoML-prov/models/trained_model_lr.pkl','rb'))
rf_model = pickle.load(open('/Users/manuelfranceschi/Desktop/clases_thebridge/ONLINE_DS_THEBRIDGE_MANUEL/Proyectos/ProyectoML-prov/models/final_model_rf.pkl','rb'))

df_test = pd.read_csv('/Users/manuelfranceschi/Desktop/clases_thebridge/ONLINE_DS_THEBRIDGE_MANUEL/Proyectos/ProyectoML-prov/data/test/test.csv')
X_test = df_test[['AgeGroup', 'HadHeartAttack', 'diabetesGroup','SmokerGroup', 'AlcoholDrinkers', 'HadArthritis','HadKidneyDisease', 'HadDepressiveDisorder']]
y_test = df_test['HadStroke']

show_metrics(lr_model, X_test,y_test)
show_metrics(rf_model, X_test,y_test)