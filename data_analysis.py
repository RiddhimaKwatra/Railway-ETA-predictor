import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
df=pd.read_csv("dataset.csv")
#print(df) #1000 rows x 11 columns
#print(df.columns) #['Train_ID', 'Station_Code', 'Train_Priority', 'Weather_Condition',
       #'Time_of_Day', 'Day_of_Week', 'Network_Congestion', 'Active_Incident',
       #'Scheduled_Arrival_Mins', 'Actual_Arrival_Mins', 'Delay_Minutes']

#print(df.isnull().sum()) --> 0 null values
#print(df.duplicated().sum()) --> 0 duplicate values
#print(df.dtypes)
#df['check'] = df['Actual_Arrival_Mins'] - df['Scheduled_Arrival_Mins']
#print((df['check'] - df['Delay_Minutes']).abs().max()) --> 0, lets gooo
#for col in ['Train_Priority', 'Weather_Condition', 'Time_of_Day', 
#            'Day_of_Week', 'Network_Congestion', 'Active_Incident']:
#    print(col, ':', df[col].unique())
#yeah yea, no mispelling of NO as no or anything

y=df.Delay_Minutes
X=df.drop(['Delay_Minutes', 'Actual_Arrival_Mins', 'Train_ID'], axis=1)

#print(y.head())
#print(X.head())
#a=(X.dtypes=="string")
#cato_cols=list(a[a].index)
cato_cols = list(X.select_dtypes(include=['object', 'string']).columns)
print(cato_cols)
num_cols=X.drop(cato_cols,axis=1)

from sklearn.preprocessing import OneHotEncoder
One_hot_Encoder= OneHotEncoder(handle_unknown="ignore", sparse_output=False)
OH_cols_X=pd.DataFrame(One_hot_Encoder.fit_transform(X[cato_cols]))
OH_cols_X.index=X.index
OH_X=pd.concat([num_cols,OH_cols_X],axis=1)
OH_X.columns=OH_X.columns.astype(str)
#print(OH_X.shape)



demo_rows = df.sample(5, random_state=42)
demo_indices = demo_rows.index

OH_X_train = OH_X.drop(demo_indices)
y_train = y.drop(demo_indices)


demo_rows.to_csv('demo_examples.csv', index=False)
model= XGBRegressor(n_estimators=700, learning_rate=0.03)
scores=-1*cross_val_score(model,OH_X_train,y_train,
                          cv=6,
                          scoring="neg_mean_absolute_error")
print("MAE:\n", scores)
print("Mean",scores.mean())
model.fit(OH_X_train,y_train)

import joblib
joblib.dump(model, 'model.pkl')
joblib.dump(One_hot_Encoder, 'encoder.pkl')