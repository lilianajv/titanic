import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import scikit-learn

data = pd.read_csv("clean_titanic".csv)
modelo = pickle.load(open("modelo.pickle", "rb"))

# titulo
                   
st.title("Habras sobrevivido al naufragio del titanic?")
         
# Exploracion inicial
         
st. header("Exploracion inical")
st.subheader("los primeros datos")
st.dataframe(data.head())
st.subheader("un descriptivo")
st.dataframe(data.describe())

#Visualizacion
                   
st.header("Visualizacion")
fig, ax = plt.subplots(1,4, sharey=True, figsize=(10,6))
ax[0].set_yalabel("%")
for idx, col in enumerate(["Pclass", "SibSp", "Parch", "Sex_male"]):
  data[col].value_counts(normalize=True).plot(kind="bar", ax=ax[idx], title=col)
plt.show()
st.pyplot(fig)
                   
fig, ax = plt.subplots(1,4, sharey=True, figsize=(10,6))
ax[0].set_yalabel("%")
for idx, col in enumerate(["Pclass", "SibSp", "Parch", "Sex_male"]):
  pd.crosstab(df[col], df["Survived"]).plot(kind="bar", ax=ax[idx], title=col)
plt.show()
st.pyplot(fig)

#Seleccion de datos

st.header("Verifica si habras sobrevivido")
col1, col2 = st.columns(2)
with col1:
  st.header("Caracteristicas")
  sexo = st.selectbox("genero", ("M", "F"))
  if sexo == "M":
    sexo = 1
  else:
    sexo = 0
  par_hijos = st.slider("Número entre padres e hijos", 0, 8)
  hermanos_esposos = st.slider("Número entre hermanos(as) y esposo(a)", 0, 5)

with col2:
                              
  st.header("Boleto")
  clase = st.selectbox("clase", (1, 2, 3))
  edad = st.slider("Edad", 0, 99)
  fare = st.slider("Disposicion a pagar el boleto", 0, 500)

#Prediccion
if st.button("Predecir"):
  pred=modelo.predict_proba(np.array([[clase, edad, hermanos_esposos, par_hijos, fare, sexo]]))
  st.text(f"Habrias sobrevivido con una probabilidad de {round(pred[0][1]*100, 1)}%.")
else:
  st.text("selecciona entre las opciones y oprime predecir")
