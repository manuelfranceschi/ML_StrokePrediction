from utils import *
#Dataframe
df = pd.read_csv('../data/raw/gym_members_exercise_tracking.csv')

# Conversión de columnas
#Gender Male or female
#df.insert(2, 'Gender_Male', np.where(df['Gender'] == 'Male', 1, 0))
#df.insert(3, 'Gender_Female', np.where(df['Gender'] == 'Female', 1, 0))

#Workout Type true or false 
#df_dummies =  pd.get_dummies(df['Workout_Type']).astype(int)
#df = pd.merge(df, df_dummies, how='inner', left_index=True,right_index=True)

#df.to_csv('../data/processed/gym_members_processed', index=False)