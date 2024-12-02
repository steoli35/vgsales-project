
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,TargetEncoder
from sklearn.compose import ColumnTransformer
from category_encoders import CatBoostEncoder,BinaryEncoder,CountEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor,export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso,BayesianRidge,LogisticRegressionCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error,accuracy_score
import xgboost
from xgboost import XGBRegressor
from scipy.special import inv_boxcox
import shap

st.set_page_config(layout="wide")
st.title("ANALYSE EXPLORATOIRE ET MODELISATION DES VENTES GLOBALES DE JEUX VIDEO AVANT 2017")
st.write("François Dumont, Thomas Bouffay, Olivier Steinbauer")
st.sidebar.title("Sommaire")
pages=["Exploration", "DataVizualization", "Modélisation - base", "Machine Learning - base"]
page=st.sidebar.radio("Aller vers", pages)

########################################################## CODE POUR LES PAGES 2 ET 3 ##########
df=pd.read_csv('cleaned_by_script_vgsales.csv')

#on passe les Name en minuscules dans df_uvlist et df_no_year
df.loc[:,'Name'] = df['Name'].str.lower()
df.loc[:,'Publisher'] = df['Publisher'].str.lower()
#on retire toutes les informations inutiles dans le nom de df_no_year elles sont entre parenthèses (JP sales)etc...
df.loc[:,'Name'] = df['Name'].str.split('(').str[0]
df.loc[:,'Publisher'] = df['Publisher'].str.split('(').str[0]

# On ne conserve que les mots et les espaces dans les Names
df.loc[:,'Name'] = df['Name'].apply(lambda x: re.sub(r'[^\w\s]','', x))
df.loc[:,'Publisher'] = df['Publisher'].apply(lambda x: re.sub(r'[^\w\s]','', x))

# on remplace les espaces doubles par des simples
df.loc[:,'Name'] = df['Name'].str.replace("  "," ")
df.loc[:,'Publisher'] = df['Publisher'].str.replace("  "," ")

# on retire tous les espaces en début et fin de Name
df.loc[:,'Name'] = df['Name'].str.strip()
df.loc[:,'Publisher'] = df['Publisher'].str.strip()

li_salon = ['Wii','NES','X360','PS3','PS2','SNES','PS4','N64','PS','XB','PC','2600','XOne','GC','GEN','DC','SAT','SCD','NG','TG16','3DO','PCFX']
li_portable = ['GB','DS','GBA','3DS','PSP','WiiU','PSV','WS','GG']
df['Type'] = np.where(df['Platform'].isin(li_salon), 'Salon', 'Portable')
df['Year'] = df['Year'].astype(int)
### Définition de 'durée de vie' pour les Plateformes et les éditeurs
def assign_longevite(group):
  plat_long = group.max() - group.min()
  return plat_long
df['Game_Sales_Period'] = df.groupby('Platform')['Year'].transform(assign_longevite)

def assign_longevite(group):
  plat_long = group.max() - group.min()
  return plat_long

df['Publisher_Sales_Period'] = df.groupby('Publisher')['Year'].transform(assign_longevite)

#Création de combinaisons de variables
df['Pub_Plat'] = df['Publisher'] + '_' + df['Platform']
df['Pub_Genre'] = df['Publisher'] + '_' + df['Genre']
df['Plat_Year'] = df['Platform'] + '_' + df['Year'].astype(str)
df['Plat_Genre'] = df['Platform'] + '_' + df['Genre']
df['Genre_Year'] = df['Genre'] + '_' + df['Year'].astype(str)

df['PSP_x_GSP'] = df['Publisher_Sales_Period'] * df['Game_Sales_Period']

############################################################################# FIN DU CODE POUR LES PAGES 2 ET 3 ########################################

####################################### PAGE 2 (MODELISATION (1)) ###################################
if page == pages[2] : 
  st.write("### Modélisation")
  st.write("Suite à l'exploration initiale des données et nos premières constatations, nous ajoutons le type de plateforme Salon/Portable.")
  
  
  code = '''
  li_salon = ['Wii','NES','X360','PS3','PS2','SNES','PS4','N64','PS','XB','PC','2600','XOne','GC','GEN','DC','SAT','SCD','NG','TG16','3DO','PCFX']
  li_portable = ['GB','DS','GBA','3DS','PSP','WiiU','PSV','WS','GG']
  df['Type'] = np.where(df['Platform'].isin(li_salon), 'Salon', 'Portable')'''
  
  st.code(code, language="python")

  st.write("Nous passons aussi l'année en entier")
  code = '''df['Year'] = df['Year'].astype(int)'''
  st.code(code, language="python")
  ### FIN PARTIE 1
  ### Affichage des premières lignes du df
  st.dataframe(df.head())

  ### PARTIE 2 - étude la répartition de Global_Sales

  st.write("## Répartition de la variable Global_Sales")
  st.write("Les différents modèles que nous avons essayés lors de notre première tentative renvoyait des résultats nuls ou négatifs, quelques fussent les variations !")
  st.write("Il est apparu clair que la distribution de la variable cible empêchait toute modélisation, à notre niveau comme on peut le voir ci-dessous.")

  # Créer les sous-graphiques
  fig = make_subplots(rows=1, cols=2)

  # Ajouter un Scatter plot
  i=1
  for colonne in ['Global_Sales']:
      fig.add_trace(
          go.Scatter(x=df[colonne], name=colonne),
          row=1, col=i
      )
      i+=1

  i=2
  for colonne in ['Global_Sales']:
      fig.add_trace(
          go.Histogram(x=df[colonne], name=colonne),
          row=1, col=i
      )
      i+=1

  fig.update_layout(width=1200,height=400)

  # Afficher le graphique dans Streamlit
  st.plotly_chart(fig)

  ### FIN DE LA PARTIE 2

  ### PARTIE 3
  ### APPLICATION DE LA METHODE BOX-COX ET VISUALISATION DE LA TRANSFORMATION
  st.write("## Application de la méthode Box-Cox sur la variable cible")
  st.write("Cette méthode est employée car nous n'avons aucune valeur négative, autrement il eut fallu utiliser la méthode Yeo-Johnson qui supporte de telles valeurs.")

  pt = PowerTransformer(method='box-cox',standardize=False)

  pt.fit(df[['Global_Sales']])

  df['Global_Sales_boxed'] = pt.transform(df[['Global_Sales']])

  only_those = ['Name','Global_Sales','Global_Sales_boxed']
  df_only_those = df[only_those]
  st.dataframe(df_only_those.head(5))

  fig = make_subplots(
    rows=1, cols=2
    )

  i=1
  for colonne in ['Global_Sales_boxed']:
      fig.add_trace(
          go.Scatter(x=df[colonne], name=colonne),
          row=1, col=i
      )
      i+=1

  i=2
  for colonne in ['Global_Sales_boxed']:
      fig.add_trace(
          go.Histogram(x=df[colonne], name=colonne),
          row=1, col=2
      )
      i+=1

  fig.update_layout(width=1200,height=400)
  st.write("Nous voyons l'effet de la 'normalisation' de la variable cible plus clairement sur les graphiques ci-dessous.")
  # Afficher le graphique dans Streamlit
  st.plotly_chart(fig)

  ### FIN DE LA PARTIE 3

  ### PARTIE 4
  ### FEATURE ENGINEERING, CREATION DE NOUVELLES VARIABLES
  st.write("## Feature Engineering à partir des données de base")
  st.write("Compte tenu du nombre limité de variables à disposition, nous avons essayé d'en ajouter de nouvelles à partir des existantes.")
  st.write("Période d'existence au sein du jeu de données des éditeurs et des plateformes ainsi que des associations potentiellement utiles.")
  
  ### checkbox pour afficher le code
  afficher_code = st.checkbox('Afficher le code')
  
  code = '''
  def assign_longevite(group):
    plat_long = group.max() - group.min()
    return plat_long
  df['Game_Sales_Period'] = df.groupby('Platform')['Year'].transform(assign_longevite)

  def assign_longevite(group):
    plat_long = group.max() - group.min()
    return plat_long
  df['Publisher_Sales_Period'] = df.groupby('Publisher')['Year'].transform(assign_longevite)

  df['Pub_Plat'] = df['Publisher'] + '_' + df['Platform']
  df['Pub_Genre'] = df['Publisher'] + '_' + df['Genre']
  df['Plat_Year'] = df['Platform'] + '_' + df['Year'].astype(str)
  df['Plat_Genre'] = df['Platform'] + '_' + df['Genre']
  df['Genre_Year'] = df['Genre'] + '_' + df['Year'].astype(str)
  '''
  ### si on checkbox
  if afficher_code:
    st.code(code, language="python")

  


####################################### PAGE 3 (MACHINE LEARNING (1)) ###################################
if page == pages[3] :
  st.write("### Machine Learning - Données de base") 
  st.write("Nous allons procéder sur deux jeux de test et d'entrainement, un normalisé par Box-Cox et l'autre non.")

   #### Séparation du jeu de données 
  X_scaled = df.drop(['Rank', 'NA_Sales',
       'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales','Year'
       ], axis=1)
  y_scaled = df['Global_Sales']

  X_non_scaled = df.drop(['Rank', 'NA_Sales',
       'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales','Year'
      ], axis=1)
  y_non_scaled = df['Global_Sales']

  X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=43)

  X_train_non_scaled, X_test_non_scaled, y_train_non_scaled, y_test_non_scaled = train_test_split(X_non_scaled, y_non_scaled, test_size=0.3, random_state=43)

  pt = PowerTransformer(method='box-cox',standardize=False).set_output(transform="pandas")

  y_train_scaled_trans = pt.fit_transform(y_train_scaled.values.reshape(-1,1))
  y_test_scaled_trans = pt.transform(y_test_scaled.values.reshape(-1,1))

  global_sales_lambda = pt.lambdas_[0]
  afficher_xtrain_scaled = st.checkbox('Afficher X_train et y_train scaled')
  if afficher_xtrain_scaled:
    st.write("X_train_scaled")
    st.dataframe(X_train_scaled.head(2))
    st.write("y_train_scaled")
    st.write(y_train_scaled_trans.head(2))
  afficher_xtrain_non_scaled = st.checkbox('Afficher X_train et y_train non_scaled')
  if afficher_xtrain_non_scaled:
    st.write("X_train_non_scaled")
    st.dataframe(X_train_scaled.head(2))
    st.write("y_train_non_scaled")
    st.write(y_train_non_scaled.head(2))

  ### ENCODAGE DES VARIABLES
  #Target_Encoder - SCALED
  te_cat = ['Name'] 

  te = TargetEncoder(categories='auto', target_type='continuous', smooth='auto', cv=5, shuffle=False).set_output(transform="pandas")

  X_train_scaled[te_cat] = te.fit_transform(X_train_scaled[te_cat], y_train_scaled)
  X_test_scaled[te_cat] = te.transform(X_test_scaled[te_cat])

  ### FREQUENCY ENCODER - SCALED
  freq_cat =['Publisher','Platform','Genre','Pub_Plat', 'Plat_Year', 'Plat_Genre','Type', 'Pub_Genre','Genre_Year']

  fr = CountEncoder(normalize=True).set_output(transform="pandas")
  X_train_scaled_encoded = fr.fit_transform(X_train_scaled[freq_cat])
  X_test_scaled_encoded = fr.transform(X_test_scaled[freq_cat])

  X_train_scaled = pd.concat([X_train_scaled.drop(freq_cat, axis=1), X_train_scaled_encoded], axis=1)
  X_test_scaled = pd.concat([X_test_scaled.drop(freq_cat, axis=1), X_test_scaled_encoded], axis=1)
  
  #Target_Encoder - NON SCALED
  te_ns_cat = ['Name']

  te_ns = TargetEncoder(categories='auto', target_type='continuous', smooth='auto', cv=5, shuffle=False).set_output(transform="pandas")

  X_train_non_scaled[te_ns_cat] = te_ns.fit_transform(X_train_non_scaled[te_ns_cat], y_train_non_scaled)
  X_test_non_scaled[te_ns_cat] = te_ns.transform(X_test_non_scaled[te_ns_cat])

  ### FREQUENCY ENCODER - NON SCALED
  freq_cat_ns = ['Publisher','Platform','Genre','Pub_Plat', 'Plat_Year', 'Plat_Genre','Type', 'Pub_Genre','Genre_Year']

  fr_ns = CountEncoder(normalize=True).set_output(transform="pandas")
  X_train_non_scaled_encoded = fr_ns.fit_transform(X_train_non_scaled[freq_cat_ns])
  X_test_non_scaled_encoded = fr_ns.transform(X_test_non_scaled[freq_cat_ns])

  X_train_non_scaled = pd.concat([X_train_non_scaled.drop(freq_cat_ns, axis=1), X_train_non_scaled_encoded], axis=1)
  X_test_non_scaled = pd.concat([X_test_non_scaled.drop(freq_cat_ns, axis=1), X_test_non_scaled_encoded], axis=1)

  X_train_scaled['Pub_x_PSP'] = X_train_scaled['Publisher'] * X_train_scaled['Publisher_Sales_Period']
  X_test_scaled['Pub_x_PSP'] = X_test_scaled['Publisher'] * X_test_scaled['Publisher_Sales_Period']

  X_train_scaled['Plat_x_GSP'] = X_train_scaled['Platform'] * X_train_scaled['Game_Sales_Period']
  X_test_scaled['Plat_x_GSP'] = X_test_scaled['Platform'] * X_test_scaled['Game_Sales_Period']


  X_train_non_scaled['Pub_x_PSP'] = X_train_non_scaled['Publisher'] * X_train_non_scaled['Publisher_Sales_Period']
  X_test_non_scaled['Pub_x_PSP'] = X_test_non_scaled['Publisher'] * X_test_non_scaled['Publisher_Sales_Period']

  X_train_non_scaled['Plat_x_GSP'] = X_train_non_scaled['Platform'] * X_train_non_scaled['Game_Sales_Period']
  X_test_non_scaled['Plat_x_GSP'] = X_test_non_scaled['Platform'] * X_test_non_scaled['Game_Sales_Period']

  ### FEATURE ENGINEERING, CREATION DE NOUVELLES VARIABLES
  st.write("## Feature Engineering à partir des données de base")
  st.write("Compte tenu du nombre limité de variables à disposition, nous avons essayé d'en ajouter de nouvelles à partir des existantes.")
  st.write("Une fois l'encodage réalisé")
  
  ### checkbox pour afficher le code
  afficher_code = st.checkbox('Afficher le code')
  
  code = '''
  X_train_scaled['Pub_x_PSP'] = X_train_scaled['Publisher'] * X_train_scaled['Publisher_Sales_Period']
  X_test_scaled['Pub_x_PSP'] = X_test_scaled['Publisher'] * X_test_scaled['Publisher_Sales_Period']

  X_train_scaled['Plat_x_GSP'] = X_train_scaled['Platform'] * X_train_scaled['Game_Sales_Period']
  X_test_scaled['Plat_x_GSP'] = X_test_scaled['Platform'] * X_test_scaled['Game_Sales_Period']


  X_train_non_scaled['Pub_x_PSP'] = X_train_non_scaled['Publisher'] * X_train_non_scaled['Publisher_Sales_Period']
  X_test_non_scaled['Pub_x_PSP'] = X_test_non_scaled['Publisher'] * X_test_non_scaled['Publisher_Sales_Period']

  X_train_non_scaled['Plat_x_GSP'] = X_train_non_scaled['Platform'] * X_train_non_scaled['Game_Sales_Period']
  X_test_non_scaled['Plat_x_GSP'] = X_test_non_scaled['Platform'] * X_test_non_scaled['Game_Sales_Period']
  '''
  ### si on checkbox
  if afficher_code:
    st.code(code, language="python")
  
  afficher_xtrain_encoded = st.checkbox('Afficher X_train_scaled et X_train_non_scaled encodés')
  if afficher_xtrain_encoded:
    st.dataframe(X_train_scaled.head(2))
    st.dataframe(X_train_non_scaled.head(2))
  
  scaler = StandardScaler().set_output(transform="pandas")
  minmaxscaler = MinMaxScaler().set_output(transform="pandas")

  # ### X_scaled
  X_train_scaled = scaler.fit_transform(X_train_scaled)
  X_test_scaled = scaler.transform(X_test_scaled)

  # X_train_scaled = minmaxscaler.fit_transform(X_train_scaled)
  # X_test_scaled = minmaxscaler.transform(X_test_scaled)

  ### X_non_scaled
  X_train_non_scaled = scaler.fit_transform(X_train_non_scaled)
  X_test_non_scaled = scaler.transform(X_test_non_scaled)

  # X_train_non_scaled = minmaxscaler.fit_transform(X_train_non_scaled)
  # X_test_non_scaled = minmaxscaler.transform(X_test_non_scaled)

  # Charger les modèles
  # rf_non_scaled = joblib.load('rf_non_scaled.joblib')
  # rf_scaled = joblib.load('rf_scaled.joblib')
  # xg_non_scaled = joblib.load('xg_non_scaled.joblib')
  # xg_scaled = joblib.load('xg_scaled.joblib')


  # Titre de l'application
  st.title('Modèles Pré-Entraînés')
  # Dictionnaire des modèles disponibles
  model_dict = {
    'RandomForrest scaled': 'rf_scaled.joblib',
    'XGBregressor scaled': 'xg_scaled.joblib',
    'RandomForrest non scaled': 'rf_non_scaled.joblib',
    'XGBregressor non scaled': 'xg_non_scaled.joblib',
  }

  # Créez une selectbox pour choisir le modèle
  selected_model_name = st.selectbox('Sélectionnez un modèle', list(model_dict.keys()))

  # Chargez le modèle sélectionné
  selected_model_path = model_dict[selected_model_name]
  model = joblib.load(selected_model_path)

  # Fonction de prédiction utilisant le modèle sélectionné
  def predict(input_data):
      return model.predict(input_data)

  # Entrée utilisateur pour la prédiction
  if model_dict[selected_model_name]=="rf_scaled.joblib" or model_dict[selected_model_name]=="xg_scaled.joblib":
    input_data = X_test_scaled
    input_data_train = X_train_scaled

  if model_dict[selected_model_name]=="rf_non_scaled.joblib" or model_dict[selected_model_name]=="xg_non_scaled.joblib":
    input_data = X_test_non_scaled
    input_data_train = X_train_non_scaled

  if st.button('Prédire'):
    result = predict(input_data)
    result_train = predict(input_data_train)

    ######################################################## Prédictions et résultats sur le jeu scaled avec RANDOM FORREST
    if model_dict[selected_model_name]=="rf_scaled.joblib":
      
      mse = mean_squared_error(y_test_scaled_trans, result)
      rmse = mse ** 0.5
      mae = mean_absolute_error(y_test_scaled_trans, result)
      r2 = r2_score(y_test_scaled_trans, result)
      medae = median_absolute_error(y_test_scaled_trans, result)

      mse_t = mean_squared_error(y_train_scaled_trans, result_train)
      rmse_t = mse_t ** 0.5
      mae_t = mean_absolute_error(y_train_scaled_trans, result_train)
      r2_t = r2_score(y_train_scaled_trans, result_train)
      medae_t = median_absolute_error(y_train_scaled_trans, result_train)
      
      st.write('Résultat de la prédiction sur test et train:\n\n')
      st.write('R2 (test):', r2,'R2 (train):', r2_t, '\n')
      st.write('MSE (test):', mse,'MSE (train):', mse_t, '\n')
      st.write('MAE (test):', mae,'MAE (train):', mae_t, '\n')
      st.write('RMSE (test):', rmse,'RMSE (train):', rmse_t, '\n')
      st.write('MedAE (test):', medae,'MedAE (train):', medae_t, '\n')

      st.write('# Valeurs réelles VS Valeurs résiduelles:\n\n')

      le_dict = {
        'Global_Sales': global_sales_lambda
      }
      y_pred = inv_boxcox(result, [global_sales_lambda])
      y_test = inv_boxcox(y_test_scaled_trans, [global_sales_lambda])

      residuals = y_test['x0'] - y_pred

      plt.figure(figsize=(10, 6))
      plt.scatter(y_test, residuals, alpha=0.3, color="g")
      plt.axhline(y=0, color='r', linestyle='--')
      plt.xlabel('Valeurs réelles')
      plt.ylabel('Résidus')
      plt.title('Résidus vs Valeurs réelles sur Test')
      st.pyplot(plt)

      comparison_df = pd.DataFrame({'Valeurs Réelles': y_test['x0'], 'Valeurs Prédites': y_pred})
      comparison_df.sort_values(by='Valeurs Réelles',ascending=True, inplace=True)
      st.dataframe(comparison_df.head(10))
      st.dataframe(comparison_df.describe())
      # y_pred = inv_boxcox(result_train, [global_sales_lambda])
      # y_test = inv_boxcox(y_train_scaled_trans, [global_sales_lambda])

      # residuals = y_test['x0'] - y_pred

      # plt.figure(figsize=(10, 6))
      # plt.scatter(y_test, residuals, alpha=0.3, color="g")
      # plt.axhline(y=0, color='r', linestyle='--')
      # plt.xlabel('Valeurs réelles')
      # plt.ylabel('Résidus')
      # plt.title('Résidus vs Valeurs réelles sur Train')
      # st.pyplot(plt)

      st.write('# SHAP values:\n\n')
      shap_values_test = shap.TreeExplainer(model).shap_values(X_test_scaled)

      X_test_scaled_array = X_test_scaled
      X_test_scaled_array = X_test_scaled.values
      plt.figure()
      shap.summary_plot(shap_values_test, X_test_scaled_array, feature_names=X_test_scaled.columns)
      st.pyplot(plt)

      # shap_values_test = shap.TreeExplainer(model).shap_values(X_train_scaled)

      # # X_test_scaled_array = X_test_scaled
      # X_train_scaled_array = X_train_scaled.values
      # plt.figure()
      # shap.summary_plot(shap_values_test, X_train_scaled_array, feature_names=X_train_scaled.columns)
      # st.pyplot(plt)
      st.write('# Matrice de corrélations:\n\n')
      y_pred = inv_boxcox(result_train, [global_sales_lambda])
      y_pred = pd.Series(result_train, name='Predictions', index=X_train_scaled.index)

      X_all = pd.concat([X_train_scaled, y_pred], axis=1)

      corr_matrix = X_all.corr()
      plt.figure(figsize=(12, 12))
      sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
      plt.title('Correlation Matrix of Features')
      # plt.show()
      st.pyplot(plt)


      ################################################################## Prédictions et résultats sur le jeu scaled avec XGB
    if model_dict[selected_model_name]=="xg_scaled.joblib":
      
      mse = mean_squared_error(y_test_scaled_trans, result)
      rmse = mse ** 0.5
      mae = mean_absolute_error(y_test_scaled_trans, result)
      r2 = r2_score(y_test_scaled_trans, result)
      medae = median_absolute_error(y_test_scaled_trans, result)

      mse_t = mean_squared_error(y_train_scaled_trans, result_train)
      rmse_t = mse_t ** 0.5
      mae_t = mean_absolute_error(y_train_scaled_trans, result_train)
      r2_t = r2_score(y_train_scaled_trans, result_train)
      medae_t = median_absolute_error(y_train_scaled_trans, result_train)
      
      st.write('Résultat de la prédiction sur test et train:\n\n')
      st.write('R2 (test):', r2,'R2 (train):', r2_t, '\n')
      st.write('MSE (test):', mse,'MSE (train):', mse_t, '\n')
      st.write('MAE (test):', mae,'MAE (train):', mae_t, '\n')
      st.write('RMSE (test):', rmse,'RMSE (train):', rmse_t, '\n')
      st.write('MedAE (test):', medae,'MedAE (train):', medae_t, '\n')
      st.write('# Valeurs réelles VS Valeurs résiduelles:\n\n')

      le_dict = {
        'Global_Sales': global_sales_lambda
      }
      y_pred = inv_boxcox(result, [global_sales_lambda])
      y_test = inv_boxcox(y_test_scaled_trans, [global_sales_lambda])

      residuals = y_test['x0'] - y_pred

      plt.figure(figsize=(10, 6))
      plt.scatter(y_test, residuals, alpha=0.3, color="g")
      plt.axhline(y=0, color='r', linestyle='--')
      plt.xlabel('Valeurs réelles')
      plt.ylabel('Résidus')
      plt.title('Résidus vs Valeurs réelles sur Test')
      st.pyplot(plt)
      comparison_df = pd.DataFrame({'Valeurs Réelles': y_test['x0'], 'Valeurs Prédites': y_pred})
      comparison_df.sort_values(by='Valeurs Réelles',ascending=True, inplace=True)
      st.dataframe(comparison_df.head(10))
      st.dataframe(comparison_df.describe())
      # y_pred = inv_boxcox(result_train, [global_sales_lambda])
      # y_test = inv_boxcox(y_train_scaled_trans, [global_sales_lambda])

      # residuals = y_test['x0'] - y_pred

      # plt.figure(figsize=(10, 6))
      # plt.scatter(y_test, residuals, alpha=0.3, color="g")
      # plt.axhline(y=0, color='r', linestyle='--')
      # plt.xlabel('Valeurs réelles')
      # plt.ylabel('Résidus')
      # plt.title('Résidus vs Valeurs réelles sur Train')
      # st.pyplot(plt)

      st.write('# SHAP values:\n\n')
      shap_values_test = shap.TreeExplainer(model).shap_values(X_test_scaled)

      X_test_scaled_array = X_test_scaled
      X_test_scaled_array = X_test_scaled.values
      plt.figure()
      shap.summary_plot(shap_values_test, X_test_scaled_array, feature_names=X_test_scaled.columns)
      st.pyplot(plt)

      # shap_values_test = shap.TreeExplainer(model).shap_values(X_train_scaled)

      # # X_test_scaled_array = X_test_scaled
      # X_train_scaled_array = X_train_scaled.values
      # plt.figure()
      # shap.summary_plot(shap_values_test, X_train_scaled_array, feature_names=X_train_scaled.columns)
      # st.pyplot(plt)
      st.write('# Matrice de corrélations:\n\n')
      y_pred = inv_boxcox(result_train, [global_sales_lambda])
      y_pred = pd.Series(result_train, name='Predictions', index=X_train_scaled.index)

      X_all = pd.concat([X_train_scaled, y_pred], axis=1)

      corr_matrix = X_all.corr()
      plt.figure(figsize=(12, 12))
      sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
      plt.title('Correlation Matrix of Features')
      # plt.show()
      st.pyplot(plt)

      ################################################################ Prédictions et résultats sur le jeu non scaled avec RANDOM FORREST
    if model_dict[selected_model_name]=="rf_non_scaled.joblib":
      
      mse = mean_squared_error(y_test_non_scaled, result)
      rmse = mse ** 0.5
      mae = mean_absolute_error(y_test_non_scaled, result)
      r2 = r2_score(y_test_non_scaled, result)
      medae = median_absolute_error(y_test_non_scaled, result)
            
      mse_t = mean_squared_error(y_train_non_scaled, result_train)
      rmse_t = mse_t ** 0.5
      mae_t = mean_absolute_error(y_train_non_scaled, result_train)
      r2_t = r2_score(y_train_non_scaled, result_train)
      medae_t = median_absolute_error(y_train_non_scaled, result_train)

      st.write('Résultat de la prédiction sur test et train:\n\n')
      st.write('R2 (test):', r2,'R2 (train):', r2_t, '\n')
      st.write('MSE (test):', mse,'MSE (train):', mse_t, '\n')
      st.write('MAE (test):', mae,'MAE (train):', mae_t, '\n')
      st.write('RMSE (test):', rmse,'RMSE (train):', rmse_t, '\n')
      st.write('MedAE (test):', medae,'MedAE (train):', medae_t, '\n')

      st.write('# Valeurs réelles VS Valeurs résiduelles:\n\n')

      y_pred = result
      y_test = y_test_non_scaled

      residuals = y_test - y_pred

      plt.figure(figsize=(10, 6))
      plt.scatter(y_test, residuals, alpha=0.3, color="g")
      plt.axhline(y=0, color='r', linestyle='--')
      plt.xlabel('Valeurs réelles')
      plt.ylabel('Résidus')
      plt.title('Résidus vs Valeurs réelles sur Test')
      st.pyplot(plt)
      comparison_df = pd.DataFrame({'Valeurs Réelles': y_test, 'Valeurs Prédites': y_pred})
      comparison_df.sort_values(by='Valeurs Réelles',ascending=True, inplace=True)
      st.dataframe(comparison_df.head(10))
      st.dataframe(comparison_df.describe())
      # y_pred = inv_boxcox(result_train, [global_sales_lambda])
      # y_test = inv_boxcox(y_train_scaled_trans, [global_sales_lambda])

      # residuals = y_test['x0'] - y_pred

      # plt.figure(figsize=(10, 6))
      # plt.scatter(y_test, residuals, alpha=0.3, color="g")
      # plt.axhline(y=0, color='r', linestyle='--')
      # plt.xlabel('Valeurs réelles')
      # plt.ylabel('Résidus')
      # plt.title('Résidus vs Valeurs réelles sur Train')
      # st.pyplot(plt)

      st.write('# SHAP values:\n\n')
      shap_values_test = shap.TreeExplainer(model).shap_values(X_test_non_scaled)

      X_test_non_scaled_array = X_test_non_scaled
      X_test_non_scaled_array = X_test_non_scaled.values
      plt.figure()
      shap.summary_plot(shap_values_test, X_test_non_scaled_array, feature_names=X_test_non_scaled.columns)
      st.pyplot(plt)

      # shap_values_test = shap.TreeExplainer(model).shap_values(X_train_scaled)

      # # X_test_scaled_array = X_test_scaled
      # X_train_scaled_array = X_train_scaled.values
      # plt.figure()
      # shap.summary_plot(shap_values_test, X_train_scaled_array, feature_names=X_train_scaled.columns)
      # st.pyplot(plt)
      st.write('# Matrice de corrélations:\n\n')
      y_pred = result_train
      y_pred = pd.Series(result_train, name='Predictions', index=X_train_scaled.index)

      X_all = pd.concat([X_train_scaled, y_pred], axis=1)

      corr_matrix = X_all.corr()
      plt.figure(figsize=(12, 12))
      sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
      plt.title('Correlation Matrix of Features')
      # plt.show()
      st.pyplot(plt)

      #################################################################### Prédictions et résultats sur le jeu non scaled avec XGB
    if model_dict[selected_model_name]=="xg_non_scaled.joblib":
      
      mse = mean_squared_error(y_test_non_scaled, result)
      rmse = mse ** 0.5
      mae = mean_absolute_error(y_test_non_scaled, result)
      r2 = r2_score(y_test_non_scaled, result)
      medae = median_absolute_error(y_test_non_scaled, result)
            
      mse_t = mean_squared_error(y_train_non_scaled, result_train)
      rmse_t = mse_t ** 0.5
      mae_t = mean_absolute_error(y_train_non_scaled, result_train)
      r2_t = r2_score(y_train_non_scaled, result_train)
      medae_t = median_absolute_error(y_train_non_scaled, result_train)

      st.write('Résultat de la prédiction sur test et train:\n\n')
      st.write('R2 (test):', r2,'R2 (train):', r2_t, '\n')
      st.write('MSE (test):', mse,'MSE (train):', mse_t, '\n')
      st.write('MAE (test):', mae,'MAE (train):', mae_t, '\n')
      st.write('RMSE (test):', rmse,'RMSE (train):', rmse_t, '\n')
      st.write('MedAE (test):', medae,'MedAE (train):', medae_t, '\n')

      st.write('# Valeurs réelles VS Valeurs résiduelles:\n\n')

      y_pred = result
      y_test = y_test_non_scaled

      residuals = y_test - y_pred

      plt.figure(figsize=(10, 6))
      plt.scatter(y_test, residuals, alpha=0.3, color="g")
      plt.axhline(y=0, color='r', linestyle='--')
      plt.xlabel('Valeurs réelles')
      plt.ylabel('Résidus')
      plt.title('Résidus vs Valeurs réelles sur Test')
      st.pyplot(plt)
      comparison_df = pd.DataFrame({'Valeurs Réelles': y_test, 'Valeurs Prédites': y_pred})
      comparison_df.sort_values(by='Valeurs Réelles',ascending=True, inplace=True)
      st.dataframe(comparison_df.head(10))
      st.dataframe(comparison_df.describe())
      # y_pred = inv_boxcox(result_train, [global_sales_lambda])
      # y_test = inv_boxcox(y_train_scaled_trans, [global_sales_lambda])

      # residuals = y_test['x0'] - y_pred

      # plt.figure(figsize=(10, 6))
      # plt.scatter(y_test, residuals, alpha=0.3, color="g")
      # plt.axhline(y=0, color='r', linestyle='--')
      # plt.xlabel('Valeurs réelles')
      # plt.ylabel('Résidus')
      # plt.title('Résidus vs Valeurs réelles sur Train')
      # st.pyplot(plt)

      st.write('# SHAP values:\n\n')
      shap_values_test = shap.TreeExplainer(model).shap_values(X_test_non_scaled)

      X_test_non_scaled_array = X_test_non_scaled
      X_test_non_scaled_array = X_test_non_scaled.values
      plt.figure()
      shap.summary_plot(shap_values_test, X_test_non_scaled_array, feature_names=X_test_non_scaled.columns)
      st.pyplot(plt)

      # shap_values_test = shap.TreeExplainer(model).shap_values(X_train_scaled)

      # # X_test_scaled_array = X_test_scaled
      # X_train_scaled_array = X_train_scaled.values
      # plt.figure()
      # shap.summary_plot(shap_values_test, X_train_scaled_array, feature_names=X_train_scaled.columns)
      # st.pyplot(plt)
      st.write('# Matrice de corrélations:\n\n')
      y_pred = result_train
      y_pred = pd.Series(result_train, name='Predictions', index=X_train_scaled.index)

      X_all = pd.concat([X_train_scaled, y_pred], axis=1)

      corr_matrix = X_all.corr()
      plt.figure(figsize=(12, 12))
      sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
      plt.title('Correlation Matrix of Features')
      # plt.show()
      st.pyplot(plt)

  else:
    st.write('Veuillez choisir un modèle.')


  


