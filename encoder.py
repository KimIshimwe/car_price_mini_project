from sklearn.preprocessing import OneHotEncoder
import joblib
import pandas as pd

df = pd.read_csv('car_price_prediction.csv', sep=',')
df = df.rename(columns= {'Prod. year': 'prod_year', 'Engine volume': 'engine_volume', 'Fuel type': 'fuel_type', 'Airbags': 'Airbags'})

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(df[["Manufacturer","Model", 'Category', 'fuel_type', 'engine_volume']])

joblib.dump(encoder, "encoder.joblib")
feature_names = encoder.get_feature_names_out(["Manufacturer","Model", 'Category', 'fuel_type', 'engine_volume'])
joblib.dump(feature_names, "encoded_feature_names.joblib")
