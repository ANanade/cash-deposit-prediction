# --------------
# Import Libraries
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Code starts here

df =  pd.read_csv(path)
df.head(5)
df.columns = map(str.lower, df.columns)
df.columns = df.columns.str.replace(" ", "_")
df.replace('NaN', np.nan)
# Code ends here


# --------------
from sklearn.model_selection import train_test_split
df.columns = df.columns.str.lower().str.replace(' ','_')
df.set_index(keys='serial_number',inplace=True,drop=True)


# Code starts here
# convert timestamps to datetime to work with time data
df.established_date = pd.to_datetime(df.established_date)
df.acquired_date = pd.to_datetime(df.acquired_date)

# Independent variable and dependent variable
X = df.drop('2016_deposits',1)
y = df['2016_deposits']

# Train and test data
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.25,random_state=3)


# --------------
# time_col = X_train.select_dtypes(exclude=[np.number,'O']).columns
time_col = ['established_date', 'acquired_date']

# Code starts here

for i in time_col:
    new_col_name = "since_" + i
    print(new_col_name)
    X_train[new_col_name] = pd.datetime.now() - X_train[i]
    X_train[new_col_name] = X_train[new_col_name].apply(lambda x: float(x.days)/365)
    X_train.drop(columns=i,inplace=True)
    
  
# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
cat = X_train.select_dtypes('object').columns.tolist()
print(cat)

# Code starts here
X_train.fillna(0,inplace = True)
X_val.fillna(0,inplace=True)
le = LabelEncoder()

for df in [X_train, X_val]:
    for col in cat:
        df[col] = le.fit_transform(df[col])

X_train_temp = pd.get_dummies(data = X_train, columns = cat)
X_val_temp = pd.get_dummies(data = X_val, columns = cat)

print(X_train_temp.head())





# Code ends here


# --------------
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Code starts here

dt = DecisionTreeRegressor(random_state = 5)
dt.fit(X_train, y_train)

accuracy = dt.score(X_val, y_val)

y_pred = dt.predict(X_val)
print(y_pred)
rmse = np.sqrt(mean_squared_error(y_pred, y_val))

print("rmse for DecisionTreeRegressor: ",rmse)



# --------------
from xgboost import XGBRegressor


# Code starts here
xgb = XGBRegressor(max_depth = 50, learning_rate = 0.83, n_estimators = 100)

xgb.fit(X_train, y_train)
accuracy = xgb.score(X_val, y_val)
y_pred = xgb.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_pred, y_val))

print("rmse for XGBRegressor: ",rmse)




