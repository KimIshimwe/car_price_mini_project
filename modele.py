import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
sklearn.set_config(transform_output="pandas")



df = pd.read_csv('car_price_prediction.csv', sep = ',')
print(df.head())
print(df.columns)
df["Mileage"] = df["Mileage"].str.extract('(\d+)').astype(float) 
df["Engine volume"] = df["Engine volume"].apply(lambda x: float(str(x)[0]) if pd.notnull(x) and str(x)[0].isdigit() else None)

# Construction du modèle
df = df.dropna(axis = 0)
y = df.Price 
print(y) 

# pour l'erreur "The feature names should match those that were passed during fit"
df = df.rename(columns= {'Prod. year': 'prod_year', 'Engine volume': 'engine_volume', 'Fuel type': 'fuel_type', 'Airbags': 'Airbags'})
features = ['Manufacturer','Model', 'prod_year', 'Category', 'fuel_type', 'engine_volume','Mileage', 'Cylinders', 'Airbags' ]
print(df.dtypes)
X = df[features]
print(X.head())

from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42)
print(train_X.head())

# variables catégorielles
s = (train_X.dtypes == 'object')
object_cols = list(s[s].index)
print(object_cols)
from sklearn.preprocessing import OneHotEncoder
#help(OneHotEncoder)

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output =False)
OH_encoder.fit(train_X[object_cols])

encoded_col_names = OH_encoder.get_feature_names_out(object_cols)

OH_cols_train = pd.DataFrame(OH_encoder.transform(train_X[object_cols]), columns=encoded_col_names, index=train_X.index)
OH_cols_valid = pd.DataFrame(OH_encoder.transform(val_X[object_cols]), columns=encoded_col_names, index=val_X.index)

OH_cols_train.index = train_X.index
OH_cols_valid.index = val_X.index

num_X_train = train_X.drop(object_cols, axis=1)
num_X_valid = val_X.drop(object_cols, axis=1)

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

OH_X_train.columns = OH_X_train.columns.astype(str) 
OH_X_valid.columns = OH_X_valid.columns.astype(str)

forest_model = RandomForestRegressor(random_state=42)
forest_model.fit(OH_X_train, train_y)
predictions = forest_model.predict(OH_X_valid)

print(mean_absolute_error(val_y, predictions))

# stocker mon modèle, encodeur et l'ordre des features
import joblib

feature_order = OH_X_train.columns.to_list() # pour l'ordre des colonnes features

joblib.dump(forest_model, "my_model.joblib", compress = 3)
joblib.dump(OH_encoder, "encoder.joblib")
joblib.dump(feature_order, "feature_order.joblib")
