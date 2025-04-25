import streamlit as st
import numpy as np
import pandas as pd
import joblib
import base64
from sklearn.preprocessing import OneHotEncoder
df = pd.read_csv('car_price_prediction.csv', sep = ',')

df = df.rename(columns= {'Prod. year': 'prod_year', 'Engine volume': 'engine_volume', 'Fuel type': 'fuel_type', 'Airbags': 'Airbags'})

model = joblib.load(filename = 'my_model.joblib')
encoder = joblib.load(filename = 'encoder.joblib')
feature_order = joblib.load('feature_order.joblib')

st.set_page_config(page_title="Prédiction du Prix de Voiture", layout="wide")

# En-tête de page stylisé
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>Prédiction du prix des voitures</h1>
    <hr>
""", unsafe_allow_html=True)
# Chargement de l'image en fond (à placer au début)

def set_background(image_file):
    # Ouvrir l'image en mode binaire
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    # Utilisation de l'image encodée en base64
    img_data = f"data:image/jpg;base64,{encoded}"
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("{img_data}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        </style>
    """, unsafe_allow_html=True)

# Chemin de l'image (en utilisant le chemin absolu que tu as mentionné)
#set_background("breaking_bad.jpeg")

import streamlit as st
import pandas as pd
import joblib
import base64

# convertir une image en base64
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img_path = "breaking_bad.jpeg"
img_base64 = get_img_as_base64(img_path)

#  image à droite
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Special+Elite&display=swap');

    .stApp {{
        background-color: #0a0a0a;
        color: #bfffbf;
        font-family: 'Special Elite', monospace;
    }}

    .container {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
    }}

    .left {{
        flex: 1;
        padding-right: 20px;
    }}

    .right {{
        width: 35%;
    }}

    .right img {{
        width: 100%;
        border-radius: 10px;
        box-shadow: 0 0 20px #00ff88;
    }}

    h1, h2 {{
        color: #00ff88;
        text-shadow: 0 0 5px #00ff88;
    }}
    </style>

    <div class="container">
        <div class="left">
            <h1>Breaking Cars</h1>
            <h2>Analyse & Prédiction de Prix</h2>
        </div>
        <div class="right">
            <img src="data:image/jpeg;base64,{img_base64}" alt="Image voiture Breaking Bad">
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
            <p style='text-align: center; font-size:18px;'>Remplissez les champs ci-dessous pour obtenir une estimation du prix</p>

            """, unsafe_allow_html = True)
def inference(model, encoder,Manufacturer, Model , prod_year, Category,fuel_type, engine_volume, Mileage, Cylinders, Airbags):
     # var. catégorielles
    var_cat = pd.DataFrame([[Manufacturer,Model, Category, fuel_type, engine_volume]], columns=['Model', 'Category', 'fuel_type', 'engine_volume'])
    var_cat_encod = encoder.transform(var_cat)
    col_names = encoder.get_feature_names_out(['Manufacturer','Model', 'Category', 'fuel_type', 'engine_volume'])
    var_cat_encod_df = pd.DataFrame(var_cat_encod, columns=col_names)

    # var. numériques
    var_num = pd.DataFrame([[prod_year, Mileage, Cylinders, Airbags]],
                           columns=['prod_year', 'Mileage', 'Cylinders', 'Airbags'])

    var = pd.concat([var_cat_encod_df, var_num], axis = 1)
    var = var[feature_order]
    
    pred = model.predict(var)
    
    return pred

st.markdown(
    """
    <style>
    label {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

Manufacturer = st.selectbox(
    "Le constructeur", df["Manufacturer"].unique())

filtered_models = df[df["Manufacturer"] == Manufacturer]["Model"].unique()

Model = st.selectbox(
    "Le modèle", filtered_models
)
prod_year = st.selectbox(
    "L'année de production", sorted(df["prod_year"].unique(), reverse=True)
)
Category = st.selectbox(
    "Catégorie", df["Category"].unique()
)

fuel_type = st.selectbox(
    "Le type de carburant", df["fuel_type"].unique()
)
engine_volume = st.selectbox(
    "La capacité du réservoir", sorted(df["engine_volume"].unique(), reverse=True)
)
Mileage = st.number_input("Kilométrage", min_value = 0, max_value = 200000, value = 0)

Cylinders = st.selectbox(
    "Le nombre de cylindres", sorted(df["Cylinders"].unique(), reverse=True)
)

Airbags = st.selectbox(
    "Le nombre d'airbags", sorted(df["Airbags"].unique(), reverse=True)
)


st.markdown(
    """
    <style>
    
    div.stButton > button {
        background-color: #32CD32 !important; 
        color: white !important; 
        border-radius: 5px; 
        border: none; 
        padding: 10px 20px; 
        font-size: 16px; 
        font-weight: bold; 
        cursor: pointer; 
    }
    div.stButton > button:hover {
        background-color: #2E8B57 !important; 
    }
    </style>
    """,
    unsafe_allow_html=True
)
if st.button("Prédire le prix"):
    
    prediction = inference(model, encoder, Manufacturer,Model , prod_year, Category,fuel_type, engine_volume, Mileage, Cylinders, Airbags)
    st.success(f"Le prix prédit est de {prediction[0]} $")
    
