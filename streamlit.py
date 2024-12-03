
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

st.set_page_config(layout="centered")

st.title("ANALYSE EXPLORATOIRE ET MODELISATION DES VENTES GLOBALES DE JEUX VIDEO AVANT 2017")
st.write("François Dumont, Thomas Bouffay, Olivier Steinbauer")
st.sidebar.title("Sommaire")
pages=["Exploration / Webscraping", "DataVizualization", "Modélisation - base", "Machine Learning - base"]
page=st.sidebar.radio("Aller vers", pages)

####################################### PAGE 0 (EXPLORATION (0)) ###################################

if page == pages[0] : 
  st.write("### Exploration")

  df = pd.read_csv("vgsales-original.csv")
  st.write("Celui-ci contient 16598 lignes de 0 à 16597 Il est composé de 3 types de données :<ul>", unsafe_allow_html=True)
  st.write("<li>6 colonnes de type float (Year, NA_Sales, EU_Sales, JP_Sales, Other_Sales et Global_Sales)</li>", unsafe_allow_html=True)
  st.write("<li>1 colonne de type int (Rank)</li>", unsafe_allow_html=True)
  st.write("<li>4 colonnes de type object (Name, Platform, Genre et Publisher)</li></ul><br>", unsafe_allow_html=True)
  afficher_code = st.checkbox('Afficher les premières lignes du jeu de données original')

  if afficher_code:
    st.dataframe(df.head(10))
    
  st.write("### Constations")

  st.write("<ul><li>Valeurs manquantes pour la variable Year (271) et Publisher (58)</li><li>Valeurs Unknown (203) pour Publisher</li></ul>", unsafe_allow_html=True)
  afficher_dfunk = st.checkbox('Afficher les lignes en question')
  if afficher_dfunk:
    df_unknown = df.loc[(df['Publisher']=='Unknown')|(df['Publisher'].isna())|df['Year'].isna()]
    st.dataframe(df_unknown)
  
  st.write("### WebScrapping")
  st.write("Les données ont été récupérées et exportées dans des fichiers csv afin de pouvoir compléter les informations du jeu de données original")
  st.write("<ul><li>VGChartz - MetaCritic - UVLIST (scripts et csv en annexes du rendu final)</li></ul>", unsafe_allow_html=True)
  
  afficher_vg =  st.checkbox('Afficher les données VGChartz')
  
  if afficher_vg:
    df_vg = pd.read_csv("vgsales_new.csv")
    st.text(f"VGChartz : {df_vg.shape[0]} lignes")
    st.dataframe(df_vg.head(50))
    df_plat_count = df["Platform"].value_counts().reset_index()
    df_plat_count.columns = ["Platform", "nb_games_origine"]
    df_vg_plat_count = df_vg["Platform"].value_counts().reset_index()
    df_vg_plat_count.columns = ["Platform", "nb_games_scrap"]

    df_both = df_plat_count.merge(df_vg_plat_count, on="Platform", how="outer")
    df_both = df_both.loc[df_both["nb_games_origine"].notna()]
    st.dataframe(df_both)
    df_both['Platform'] = df_both['Platform'].astype(str).astype('category')

    fig = px.bar(df_both, y='Platform', x='nb_games_scrap', text_auto='.2s',
             hover_data=['nb_games_origine', 'nb_games_scrap'], color='nb_games_origine',
             labels={'Nb jeux':'Jeux par plateforme'},height=800)
    fig.update_layout(barmode='group')
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title="Platform", yaxis_title="Nombre de jeux")
    st.plotly_chart(fig)

  afficher_meta = st.checkbox('Afficher les données MetaCritic')
  
  if afficher_meta:
    df_meta = pd.read_csv('Scores_Metacritic_V2.csv')
    update_platform_name = {'Dreamcast': 'DC',
                        'Game Boy Advance': 'GBA',
                        'GameCube': 'GC',
                        'Meta Quest': 'MQ',
                        'Nintendo 64': 'N64',
                        'Nintendo Switch': 'SWITCH',
                        'PlayStation' : 'PS',
                        'PlayStation 2': 'PS2',
                        'PlayStation 3': 'PS3',
                        'PlayStation 4': 'PS4',
                        'PlayStation 5': 'PS5',
                        'PlayStation Vita': 'PSV',
                        'Wii': 'Wii',
                        'Wii U': 'WiiU',
                        'Xbox': 'XB',
                        'Xbox 360': 'X360',
                        'Xbox One': 'XOne',
                        'Xbox Series X': 'XBSX',
                        'iOS (iPhone/iPad)': 'IOS'}
    df_meta.Platform = df_meta.Platform.replace(update_platform_name)
    # np.sort(df_meta.Platform.unique()), np.sort(df_meta.Platform.unique())
    st.text(f"MetaCritic : {df_meta.shape[0]} lignes")
    st.dataframe(df_meta.head(50))
    df_plat_count = df["Platform"].value_counts().reset_index()
    df_plat_count.columns = ["Platform", "nb_games_origine"]
    df_meta_plat_count = df_meta["Platform"].value_counts().reset_index()
    df_meta_plat_count.columns = ["Platform", "nb_games_scrap"]

    df_both = df_plat_count.merge(df_meta_plat_count, on="Platform", how="outer")
    df_both = df_both.loc[df_both["nb_games_origine"].notna()]
    st.dataframe(df_both)
    df_both['Platform'] = df_both['Platform'].astype(str).astype('category')
    
    fig = px.bar(df_both, y='Platform', x='nb_games_scrap', text_auto='.2s',
             hover_data=['nb_games_origine', 'nb_games_scrap'], color='nb_games_origine',
             labels={'Nb jeux':'Jeux par plateforme'},height=800)
    fig.update_layout(barmode='group')
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title="Platform", yaxis_title="Nombre de jeux")
    st.plotly_chart(fig)

  afficher_uvlist = st.checkbox('Afficher les données UVLIST')
  if afficher_uvlist:
    df_uvlist0 = pd.read_csv('base_uvlist.csv')
    df_uvlist1 = pd.read_csv('base_uvlist1.csv')
    df_uvlist2 = pd.read_csv('base_uvlist2.csv')
    df_uvlist3 = pd.read_csv('base_uvlist3.csv')
    df_uvlist = pd.concat([df_uvlist0, df_uvlist1, df_uvlist2, df_uvlist3],ignore_index=True).drop(columns='Unnamed: 0')
    # df_uvlist['Year'] = df_uvlist['Year'].str.extract('(\d{4})').astype(float)
    df_uvlist['Name'] = df_uvlist['Name'].str.strip()
    df_uvlist['Platform'] = df_uvlist['Platform'].str.strip().astype(str)
    df_uvlist['Publisher'] = df_uvlist['Publisher'].str.strip()
    #on itialise deux listes de correspondances pour remplacer les valeurs de uvlist par celles de vgchartz
    uvlist_platforms = ['Wii','NES','GB','Nintendo DS','X360','PS3','PS2','SNES','GBA',
                        '3DS','PS4','N64','PS','Xbox','Windows','Atari 2600','PSP','Xbox One','GameCube',
                        'Wii U','Mega Drive','Dreamcast','PS Vita','Saturn','Mega-CD','WonderSwan','Neo-Geo','PC Engine',
                        '3DO','Game Gear','PC-FX']

    vgchartz_platforms = ['Wii', 'NES', 'GB', 'DS', 'X360', 'PS3', 'PS2', 'SNES', 'GBA',
                          '3DS', 'PS4', 'N64', 'PS', 'XB', 'PC', '2600', 'PSP', 'XOne', 'GC',
                          'WiiU', 'GEN', 'DC', 'PSV', 'SAT', 'SCD', 'WS', 'NG', 'TG16',
                          '3DO', 'GG', 'PCFX']

    #on remplace dans uvlist par les valeurs de Platform de vgchartz
    df_uvlist['Platform'] = df_uvlist['Platform'].replace(uvlist_platforms, vgchartz_platforms)

    df_uvlist = df_uvlist.dropna(subset=['Year', 'Publisher'], how='any')

    st.text(f"UVLIST : {df_uvlist.shape[0]} lignes")
    st.dataframe(df_uvlist.head(50))
    df_plat_count = df["Platform"].value_counts().reset_index()
    df_plat_count.columns = ["Platform", "nb_games_origine"]
    df_uvlist_plat_count = df_uvlist["Platform"].value_counts().reset_index()
    df_uvlist_plat_count.columns = ["Platform", "nb_games_scrap"]

    df_both = df_plat_count.merge(df_uvlist_plat_count, on="Platform", how="outer")
    df_both = df_both.loc[df_both["nb_games_origine"].notna()]
    st.dataframe(df_both)
    df_both['Platform'] = df_both['Platform'].astype(str).astype('category')
    
    fig = px.bar(df_both, y='Platform', x='nb_games_scrap', text_auto='.2s',
             hover_data=['nb_games_origine', 'nb_games_scrap'], color='nb_games_origine',
             labels={'Nb jeux':'Jeux par plateforme'},height=800)
    fig.update_layout(barmode='group')
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title="Platform", yaxis_title="Nombre de jeux")
    st.plotly_chart(fig)



  st.write("<br><ul><li>Quatre lignes ont été mises à jour manuellement</li></ul>", unsafe_allow_html=True)
  df_last_miss = df[df.Year>2016]
  st.dataframe(df_last_miss)


  st.write("<br><ul><li>Suppression des 60 lignes dont la correspondance n'a pu être faîte malgré la somme d'informations récupérée.</li></ul>", unsafe_allow_html=True)
  afficher_clean = st.checkbox('Afficher les données nettoyées')
  if afficher_clean:
    df_clean = pd.read_csv("cleaned_by_script_vgsales.csv")
    st.text(f"Fichier complété : {df_clean.shape[0]} lignes")
    st.dataframe(df_clean.head(50))
######################################## PAGE 1 DATAVIZUALIZATION #################################################################
if page == pages[1]:
  st.write("### Evolution du nombre de jeux sortis par année")
  df = pd.read_csv("cleaned_by_script_vgsales.csv")
  fig = go.Figure()
  fig.add_trace(go.Histogram(x=df.Year[df.Year.notna()],
                            marker_color='darkorange',
                            marker_line=dict(width=2, color='black')))
  fig.update_layout(bargap=0.2, title='Évolution du nombre de jeux sortis par année',
                    xaxis_title = "Année de sortie",
                    yaxis_title = "Nombre de jeux",
                    height = 500)
  st.plotly_chart(fig)

  st.write("### Comparaison du nombre de jeux sortis et celui du nombre de ventes médian par année de sortie")
  fig, ax = plt.subplots(figsize=(15, 10))
  sns.set_style("whitegrid", {'axes.grid' : False})
  sns.lineplot(x='Year', y='Global_Sales', data=df, ax=ax, label='Nombre de ventes médian des jeux\n(en million)\nAvec la répartition inter-percentiles\n(2.5%-97.5%)', errorbar="pi", estimator="median")
  sns.move_legend(ax, "upper left")
  ax.set_xlabel('Année de sortie des jeux', labelpad = 15, fontsize = 16)
  ax.set_ylabel('Nombre de ventes\n(en million)', labelpad = 15, fontsize = 16)
  ax.set_title('Nombre de ventes médian des jeux par année de sortie\nNombre de jeux sortis par année', fontsize = 16);
  ax2 = ax.twinx()
  game_counts = df.Year.value_counts().sort_index()
  # sns.lineplot(x=game_counts.index, y=game_counts.values, ax = ax2, label='Nombre de jeux sortis')
  df.Year.value_counts().sort_index().plot(ax=ax2, color='orange', kind='area', alpha=0.2, legend = 'Nombre de jeux sortis')
  # print(data_no_nan.Year.value_counts().sort_index())
  # sns.countplot(x='Year', data=data_no_nan, ax=ax2, color ='red', alpha=0.2, edgecolor='black', label='Nombre de jeux')
  # ax.sharex(ax2)
  ax2.set_xlabel('')
  ax2.set_ylabel('Nombre de jeux',  fontsize = 16)
  ax2.legend(['Nombre de jeux sortis'], fontsize = 9)
  st.pyplot(fig)

  ################## GRAPH TOP 10 
  st.write("### Top 10 des jeux par régions")
  show_top_10 = st.checkbox("Afficher TOP 10 des jeux")
  if show_top_10:
    top10_EU = df[['Name', 'EU_Sales', 'Publisher', 'Genre']].sort_values(by='EU_Sales').tail(10)
    top10_JP = df[['Name', 'JP_Sales', 'Publisher', 'Genre']].sort_values(by='JP_Sales').tail(10)
    top10_NA = df[['Name', 'NA_Sales', 'Publisher', 'Genre']].sort_values(by='NA_Sales').tail(10)
    top10_Other = df[['Name', 'Other_Sales', 'Publisher', 'Genre']].sort_values(by='Other_Sales').tail(10)
    top10_Gl = df[['Name', 'Global_Sales', 'Publisher', 'Genre']].sort_values(by='Global_Sales').tail(10)

    fig = make_subplots(rows=5, cols=1,
                        subplot_titles=("Marché Européen",
                                        "Marché Japonais",
                                        "Marché Nord Américain",
                                        "Marché autres région",
                                        "Marché globale"))
    fig.append_trace(
        go.Bar(y=top10_EU["Name"],
              x=top10_EU["EU_Sales"],
              orientation='h',
              text=top10_EU["Publisher"],
              name='Europe'),
        row=1, col=1,
                )
    fig.append_trace(
        go.Bar(y=top10_JP["Name"],
              x=top10_JP["JP_Sales"],
              orientation='h',
              text=top10_JP["Publisher"],
              name='Japon'),
        row=2, col=1
                )
    fig.append_trace(
        go.Bar(y=top10_NA["Name"],
              x=top10_NA["NA_Sales"],
              orientation='h',
              text=top10_NA["Publisher"],
              name='Amérique du Nord'),
        row=3, col=1
                )
    fig.append_trace(
        go.Bar(y=top10_Other["Name"],
              x=top10_Other["Other_Sales"],
              orientation='h',
              text=top10_Other["Publisher"],
              name='Autres régions'),
        row=4, col=1
                )
    fig.append_trace(
        go.Bar(y=top10_Gl["Name"],
              x=top10_Gl["Global_Sales"],
              orientation='h',
              text=top10_Gl["Publisher"],
              name='Monde'),
        row=5, col=1
                )
    fig.update_xaxes(title_text="Nombre de ventes (en million)", row=1, col=1)
    fig.update_xaxes(title_text="Nombre de ventes (en million)", row=2, col=1)
    fig.update_xaxes(title_text="Nombre de ventes (en million)", row=3, col=1)
    fig.update_xaxes(title_text="Nombre de ventes (en million)", row=4, col=1)
    fig.update_xaxes(title_text="Nombre de ventes (en million)", row=5, col=1)

    fig.update_layout(title="Top 10 des jeux par nombre de ventes",
                      xaxis_title="Nombre de ventes (en million)",
                      height=2400,width=800)
    st.plotly_chart(fig)
  st.write("### Top 5 des éditeurs par régions")
  show_top_10_pub = st.checkbox("Afficher TOP 5 des éditeurs")
  if show_top_10_pub:
    pie_data_gl = df.groupby('Publisher').sum().sort_values('Global_Sales', ascending=False).reset_index().head()
    pie_data_na = df.groupby('Publisher').sum().sort_values('NA_Sales', ascending=False).reset_index().head()
    pie_data_eu = df.groupby('Publisher').sum().sort_values('EU_Sales', ascending=False).reset_index().head()
    pie_data_jp = df.groupby('Publisher').sum().sort_values('JP_Sales', ascending=False).reset_index().head()
    pie_data_other = df.groupby('Publisher').sum().sort_values('Other_Sales', ascending=False).reset_index().head()

    fig = make_subplots(rows=3, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]])

    fig.add_trace(go.Pie(values=pie_data_na['NA_Sales'],
                        labels=pie_data_na['Publisher'],
                        pull=[0.15,0,0,0,0], ),
                  1, 1)
    fig.add_trace(go.Pie(values=pie_data_eu['EU_Sales'],
                        labels=pie_data_eu['Publisher'],
                        pull=[0.15,0,0,0,0], ),
                  1, 2)
    fig.add_trace(go.Pie(values=pie_data_jp['JP_Sales'],
                        labels=pie_data_jp['Publisher'],
                        pull=[0.15,0,0,0,0], ),
                  2, 1)
    fig.add_trace(go.Pie(values=pie_data_other['Other_Sales'],
                        labels=pie_data_other['Publisher'],
                        pull=[0.15,0,0,0,0], ),
                  2, 2)
    fig.add_trace(go.Pie(values=pie_data_gl['Global_Sales'],
                        labels=pie_data_gl['Publisher'],
                        pull=[0.15,0,0,0,0], ),
                  3, 1)

    fig.update_traces(hole=.3, hoverinfo="label+percent+name")

    fig.update_layout(
        title_text="Répartition du top 5 des éditeurs par region",
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='NA', x=sum(fig.get_subplot(1, 1).x) / 2, y=(sum(fig.get_subplot(1, 1).y))*1.012 / 2,
                          font_size=20, showarrow=False, xanchor="center"),
                    dict(text='EU', x=sum(fig.get_subplot(1, 2).x) / 2, y=(sum(fig.get_subplot(1, 2).y))*1.012 / 2,
                          font_size=20, showarrow=False, xanchor="center"),
                    dict(text='JP', x=sum(fig.get_subplot(2, 1).x) / 2, y=sum(fig.get_subplot(2, 1).y) / 2,
                          font_size=20, showarrow=False, xanchor="center"),
                    dict(text='Other', x=sum(fig.get_subplot(2, 2).x) / 2, y=sum(fig.get_subplot(2, 2).y) / 2,
                          font_size=20, showarrow=False, xanchor="center"),
                    dict(text='Global', x=sum(fig.get_subplot(3, 1).x) / 2, y=(sum(fig.get_subplot(3, 1).y))*0.98 / 2,
                          font_size=20, showarrow=False, xanchor="center")],
        height=1300, width=800)
    st.plotly_chart(fig)
  st.write("### Top 5 des plateformes par régions")
  show_top_10_plat = st.checkbox("Afficher TOP 5 des plateformes")
  if show_top_10_plat:
    pie_data_gl = df.groupby('Platform').sum().sort_values('Global_Sales', ascending=False).reset_index().head()
    pie_data_na = df.groupby('Platform').sum().sort_values('NA_Sales', ascending=False).reset_index().head()
    pie_data_eu = df.groupby('Platform').sum().sort_values('EU_Sales', ascending=False).reset_index().head()
    pie_data_jp = df.groupby('Platform').sum().sort_values('JP_Sales', ascending=False).reset_index().head()
    pie_data_other = df.groupby('Platform').sum().sort_values('Other_Sales', ascending=False).reset_index().head()

    fig = make_subplots(rows=3, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]])

    fig.add_trace(go.Pie(values=pie_data_na['NA_Sales'],
                        labels=pie_data_na['Platform'],
                        pull=[0.15,0,0,0,0], ),
                  1, 1)
    fig.add_trace(go.Pie(values=pie_data_eu['EU_Sales'],
                        labels=pie_data_eu['Platform'],
                        pull=[0.15,0,0,0,0], ),
                  1, 2)
    fig.add_trace(go.Pie(values=pie_data_jp['JP_Sales'],
                        labels=pie_data_jp['Platform'],
                        pull=[0.15,0,0,0,0], ),
                  2, 1)
    fig.add_trace(go.Pie(values=pie_data_other['Other_Sales'],
                        labels=pie_data_other['Platform'],
                        pull=[0.15,0,0,0,0], ),
                  2, 2)
    fig.add_trace(go.Pie(values=pie_data_gl['Global_Sales'],
                        labels=pie_data_gl['Platform'],
                        pull=[0.15,0,0,0,0], ),
                  3, 1)

    fig.update_traces(hole=.3, hoverinfo="label+percent+name")

    fig.update_layout(
        title_text="Répartition du top 5 des platforme par région",
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='NA', x=sum(fig.get_subplot(1, 1).x) / 2, y=(sum(fig.get_subplot(1, 1).y))*1.012 / 2,
                          font_size=20, showarrow=False, xanchor="center"),
                    dict(text='EU', x=sum(fig.get_subplot(1, 2).x) / 2, y=(sum(fig.get_subplot(1, 2).y))*1.012 / 2,
                          font_size=20, showarrow=False, xanchor="center"),
                    dict(text='JP', x=sum(fig.get_subplot(2, 1).x) / 2, y=sum(fig.get_subplot(2, 1).y) / 2,
                          font_size=20, showarrow=False, xanchor="center"),
                    dict(text='Other', x=sum(fig.get_subplot(2, 2).x) / 2, y=sum(fig.get_subplot(2, 2).y) / 2,
                          font_size=20, showarrow=False, xanchor="center"),
                    dict(text='Global', x=sum(fig.get_subplot(3, 1).x) / 2, y=(sum(fig.get_subplot(3, 1).y))*0.98 / 2,
                          font_size=20, showarrow=False, xanchor="center")],
        height=1300, width=800)
    st.plotly_chart(fig)
  st.write("### Ventes globales (distinction du type de plateforme)")
  show_top_glob_sales = st.checkbox("Afficher les ventes globales par type de plateforme")
  if show_top_glob_sales:
    df_vgchartz = pd.read_csv('cleaned_by_script_vgsales.csv')
    df_vgchartz['Platform'].unique()
    li_salon = ['Wii','NES','X360','PS3','PS2','SNES','PS4','N64','PS','XB','PC','2600','XOne','GC','GEN','DC','SAT','SCD','NG','TG16','3DO','PCFX']
    li_portable = ['GB','DS','GBA','3DS','PSP','WiiU','PSV','WS','GG']

    df_vgchartz['Type'] = np.where(df_vgchartz['Platform'].isin(li_salon), 'Salon', 'Portable')

    df_vgchartz['Year'] = df_vgchartz['Year'].astype(int)
    platform_count = df_vgchartz['Platform'].value_counts()
    platform_count.columns = ['Platform','Count']
    valides_platform = platform_count[platform_count >= 34].index

    df_vgchartz_filter_platform = df_vgchartz[df_vgchartz['Platform'].isin(valides_platform)]

    df_plat_type_global_sales = df_vgchartz_filter_platform.groupby(['Type','Platform'])['Global_Sales'].sum().reset_index().sort_values(by='Global_Sales', ascending=False)

    df_salon_sales = df_plat_type_global_sales.loc[df_plat_type_global_sales['Type'] == "Salon"]
    df_portable_sales = df_plat_type_global_sales.loc[df_plat_type_global_sales['Type'] == "Portable"]

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Ventes mondiales",))

    fig.add_trace(go.Bar(y = df_salon_sales["Global_Sales"], x = df_salon_sales["Platform"],name="Consoles de salon",text=round(df_salon_sales["Global_Sales"],2), textposition='auto'), row=1, col=1)
    fig.add_trace(go.Bar(y = df_portable_sales["Global_Sales"], x = df_portable_sales["Platform"],name="Consoles portables",text=round(df_portable_sales["Global_Sales"],2), textposition='auto'), row=2, col=1)
    fig.update_layout(width=800,height=800,title_text="Ventes globales de jeux par type d'équipment (en millions de copies vendues)")
    st.plotly_chart(fig)
  st.write("### Ventes régionales de jeux par type de support")
  show_top_reg_sales = st.checkbox("Afficher les ventes régionales de jeux par type de support")
  if show_top_reg_sales:
    df_vgchartz = pd.read_csv('cleaned_by_script_vgsales.csv')
    df_vgchartz['Platform'].unique()
    li_salon = ['Wii','NES','X360','PS3','PS2','SNES','PS4','N64','PS','XB','PC','2600','XOne','GC','GEN','DC','SAT','SCD','NG','TG16','3DO','PCFX']
    li_portable = ['GB','DS','GBA','3DS','PSP','WiiU','PSV','WS','GG']

    df_vgchartz['Type'] = np.where(df_vgchartz['Platform'].isin(li_salon), 'Salon', 'Portable')

    df_vgchartz['Year'] = df_vgchartz['Year'].astype(int)
    platform_count = df_vgchartz['Platform'].value_counts()
    platform_count.columns = ['Platform','Count']
    valides_platform = platform_count[platform_count >= 34].index

    df_vgchartz_filter_platform = df_vgchartz[df_vgchartz['Platform'].isin(valides_platform)]
    df_plat_na_sales = df_vgchartz_filter_platform.groupby(['Type','Platform'])['NA_Sales'].sum().reset_index().sort_values(by='NA_Sales', ascending=True)
    df_plat_eu_sales = df_vgchartz_filter_platform.groupby(['Type','Platform'])['EU_Sales'].sum().reset_index().sort_values(by='EU_Sales', ascending=True)
    df_plat_jp_sales = df_vgchartz_filter_platform.groupby(['Type','Platform'])['JP_Sales'].sum().reset_index().sort_values(by='JP_Sales', ascending=True)
    df_plat_other_sales = df_vgchartz_filter_platform.groupby(['Type','Platform'])['Other_Sales'].sum().reset_index().sort_values(by='Other_Sales', ascending=True)

    df_salon_na_sales = df_plat_na_sales.loc[df_plat_na_sales['Type'] == "Salon"]
    df_portable_na_sales = df_plat_na_sales.loc[df_plat_na_sales['Type'] == "Portable"]

    df_salon_eu_sales = df_plat_eu_sales.loc[df_plat_eu_sales['Type'] == "Salon"]
    df_portable_eu_sales = df_plat_eu_sales.loc[df_plat_eu_sales['Type'] == "Portable"]

    df_salon_jp_sales = df_plat_jp_sales.loc[df_plat_jp_sales['Type'] == "Salon"]
    df_portable_jp_sales = df_plat_jp_sales.loc[df_plat_jp_sales['Type'] == "Portable"]

    df_salon_other_sales = df_plat_other_sales.loc[df_plat_other_sales['Type'] == "Salon"]
    df_portable_other_sales = df_plat_other_sales.loc[df_plat_other_sales['Type'] == "Portable"]

    fig = make_subplots(rows=4, cols=1, subplot_titles=("Ventes en Amérique du Nord", "Ventes en Europe", "Ventes au Japon", "Ventes dans les autres régions"))

    fig.add_trace(go.Bar(y = df_salon_na_sales["NA_Sales"], x = df_salon_na_sales["Platform"],name="Ventes NA",
             text=round(df_salon_na_sales["NA_Sales"],2), textposition='auto'), row=1, col=1)
    fig.add_trace(go.Bar(y = df_salon_eu_sales["EU_Sales"], x = df_salon_eu_sales["Platform"],name="Ventes EU",
              text=round(df_salon_eu_sales["EU_Sales"],2), textposition='auto'), row=2, col=1)
    fig.add_trace(go.Bar(y = df_salon_jp_sales["JP_Sales"], x = df_salon_jp_sales["Platform"],name="Ventes JP",
              text=round(df_salon_jp_sales["JP_Sales"],2), textposition='auto'), row=3, col=1)
    fig.add_trace(go.Bar(y = df_salon_other_sales["Other_Sales"], x = df_salon_other_sales["Platform"],name="Ventes autres régions",
              text=round(df_salon_other_sales["Other_Sales"],2), textposition='auto'), row=4, col=1)

    fig.update_layout(width=800,height=1200,title_text="Ventes régionales de jeux sur équipments de salons (en millions de copies vendues)")
    st.plotly_chart(fig)

    fig = make_subplots(rows=2, cols=2, subplot_titles=("Ventes en Amérique du Nord", "Ventes en Europe", "Ventes au Japon", "Ventes dans les autres régions"))

    fig.add_trace(go.Bar(y = df_portable_na_sales["NA_Sales"], x = df_portable_na_sales["Platform"],name="Ventes NA",text=round(df_portable_na_sales["NA_Sales"],2), textposition='auto'), row=1, col=1)
    fig.add_trace(go.Bar(y = df_portable_eu_sales["EU_Sales"], x = df_portable_eu_sales["Platform"],name="Ventes EU",text=round(df_portable_eu_sales["EU_Sales"],2), textposition='auto'), row=1, col=2)
    fig.add_trace(go.Bar(y = df_portable_jp_sales["JP_Sales"], x = df_portable_jp_sales["Platform"],name="Ventes JP",text=round(df_portable_jp_sales["JP_Sales"],2), textposition='auto'), row=2, col=1)
    fig.add_trace(go.Bar(y = df_portable_other_sales["Other_Sales"], x = df_portable_other_sales["Platform"],name="Ventes autres régions",text=round(df_portable_other_sales["Other_Sales"],2), textposition='auto'), row=2, col=2)

    fig.update_layout(width=800,height=800,title_text="Ventes régionales de jeux sur équipements portables (en millions de copies vendues)")
    st.plotly_chart(fig)
  st.markdown("### Volume de jeux édités par plateforme au fil du temps pour le type Salon")
  show_top_vol_sales = st.checkbox("Afficher les volumes de jeux édités par plateforme au fil du temps (Salon)")
  if show_top_vol_sales:
    df_vgchartz = pd.read_csv('cleaned_by_script_vgsales.csv')
    df_vgchartz['Platform'].unique()
    li_salon = ['Wii','NES','X360','PS3','PS2','SNES','PS4','N64','PS','XB','PC','2600','XOne','GC','GEN','DC','SAT','SCD','NG','TG16','3DO','PCFX']
    li_portable = ['GB','DS','GBA','3DS','PSP','WiiU','PSV','WS','GG']

    df_vgchartz['Type'] = np.where(df_vgchartz['Platform'].isin(li_salon), 'Salon', 'Portable')

    df_vgchartz['Year'] = df_vgchartz['Year'].astype(int)
    platform_count = df_vgchartz['Platform'].value_counts()
    platform_count.columns = ['Platform','Count']
    valides_platform = platform_count[platform_count >= 34].index

    df_vgchartz_filter_platform = df_vgchartz[df_vgchartz['Platform'].isin(valides_platform)]
    df_plat_year_count = df_vgchartz_filter_platform.groupby(['Type','Platform', 'Year']).size().reset_index(name='Count').sort_values(by='Count', ascending=True)

    df_salon_sales = df_plat_year_count.loc[df_plat_year_count['Type'] == "Salon"].sort_values(by='Year', ascending=True)

    #vu le nombre de plateforme il faut rajouter plusieurs palettes de couleurs
    #color_sequence = px.colors.qualitative.Dark2 + px.colors.qualitative.Vivid

    fig = px.bar(df_salon_sales,
                y='Platform', x='Count',
                orientation="h",
                hover_data=['Platform', 'Year'], color='Year',
                labels={'Platform'}, height=800, width=800,
                color_continuous_scale=px.colors.sequential.Inferno)


    fig.update_layout(
        title='Volume de jeux édités par plateforme au fil du temps pour le type Salon',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title=dict(
                text="Plateformes de salon",
                font=dict(
                    size=16
                    )
                ),
            tickfont_size=14,
            ),
        legend=dict(
            x=0,
            y=1,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
            ),
        xaxis=dict(
            title=dict(
                text="Nbre de Jeux publiés",
                font=dict(
                    size=16
                    )
                ),
            tickfont_size=14,
            ),

        )
    st.plotly_chart(fig)
  st.write("### Volume de jeux édités par plateforme au fil du temps pour le type Portable")
  show_top_volp_sales = st.checkbox("Afficher les volumes de jeux édités par plateforme au fil du temps (Portable)")
  if show_top_volp_sales:
    df_vgchartz = pd.read_csv('cleaned_by_script_vgsales.csv')
    df_vgchartz['Platform'].unique()
    li_salon = ['Wii','NES','X360','PS3','PS2','SNES','PS4','N64','PS','XB','PC','2600','XOne','GC','GEN','DC','SAT','SCD','NG','TG16','3DO','PCFX']
    li_portable = ['GB','DS','GBA','3DS','PSP','WiiU','PSV','WS','GG']

    df_vgchartz['Type'] = np.where(df_vgchartz['Platform'].isin(li_salon), 'Salon', 'Portable')

    df_vgchartz['Year'] = df_vgchartz['Year'].astype(int)
    platform_count = df_vgchartz['Platform'].value_counts()
    platform_count.columns = ['Platform','Count']
    valides_platform = platform_count[platform_count >= 34].index

    df_vgchartz_filter_platform = df_vgchartz[df_vgchartz['Platform'].isin(valides_platform)]
    df_plat_year_count = df_vgchartz_filter_platform.groupby(['Type','Platform', 'Year']).size().reset_index(name='Count').sort_values(by='Count', ascending=False)

    df_portable_sales = df_plat_year_count.loc[df_plat_year_count['Type'] == "Portable"].sort_values(by='Year', ascending=True)

    #vu le nombre de plateforme il faut rajouter plusieurs palettes de couleurs
    #color_sequence = px.colors.qualitative.Dark2

    fig = px.bar(df_portable_sales,
                y='Platform', x='Count',
                hover_data=['Platform', 'Year'], color='Year',
                labels={'Platform'}, height=800, width=800,orientation="h",
                color_continuous_scale=px.colors.sequential.Inferno)


    fig.update_layout(
        title='Volume de jeux édités par plateforme au fil du temps pour le type portable',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title=dict(
                text="Plateformes portables",
                font=dict(
                    size=16
                    )
                ),
            tickfont_size=14,
            ),
        legend=dict(
            x=0,
            y=1,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
            ),
        xaxis=dict(
            title=dict(
                text="Nbre de Jeux publiés",
                font=dict(
                    size=16
                    )
                ),
            tickfont_size=14,
            ),

        )
    st.plotly_chart(fig)
  st.write("### Durée moyenne du marché de développement d'un jeu sur une plateforme")
  show_top_vie = st.checkbox("Durée moyenne du marché de développement d'un jeu")
  if show_top_vie:
    df_vgchartz = pd.read_csv('cleaned_by_script_vgsales.csv')
    df_vgchartz['Platform'].unique()
    li_salon = ['Wii','NES','X360','PS3','PS2','SNES','PS4','N64','PS','XB','PC','2600','XOne','GC','GEN','DC','SAT','SCD','NG','TG16','3DO','PCFX']
    li_portable = ['GB','DS','GBA','3DS','PSP','WiiU','PSV','WS','GG']

    df_vgchartz['Type'] = np.where(df_vgchartz['Platform'].isin(li_salon), 'Salon', 'Portable')

    df_vgchartz['Year'] = df_vgchartz['Year'].astype(int)
    platform_count = df_vgchartz['Platform'].value_counts()
    platform_count.columns = ['Platform','Count']
    valides_platform = platform_count[platform_count >= 34].index

    df_vgchartz_filter_platform = df_vgchartz[df_vgchartz['Platform'].isin(valides_platform)]
    df_plat_year_count = df_vgchartz_filter_platform.groupby(['Type','Platform', 'Year']).size().reset_index(name='Count').sort_values(by='Count', ascending=False)
    df_salon_sales = df_plat_year_count.loc[df_plat_year_count['Type'] == "Salon"].sort_values(by='Year', ascending=False)

    #on va compter le nombre d'années par Platform
    df_salon_year_count = df_salon_sales.groupby(['Platform']).size().reset_index(name='Count').sort_values(by='Count', ascending=False)

    fig = px.bar(df_salon_year_count,
                y='Platform', x='Count',
                hover_data=['Platform', 'Count'],
                color='Count',
                labels={'Platform'}, height=800, width=800,orientation="h",
                color_continuous_scale=px.colors.sequential.Inferno)


    fig.update_layout(
        title='durée moyenne du marché de développement d\'un jeu sur une plateforme',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title=dict(
                text="Plateformes de salon",
                font=dict(
                    size=16
                    )
                ),
            tickfont_size=14,
            ),
        legend=dict(
            x=0,
            y=1,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
            ),
        xaxis=dict(
            title=dict(
                text="Nbre années",
                font=dict(
                    size=16
                    )
                ),
            tickfont_size=14,
            ),

        )
    st.plotly_chart(fig)
    df_plat_year_count = df_vgchartz.groupby(['Type','Platform', 'Year']).size().reset_index(name='Count').sort_values(by='Year', ascending=False)

    df_portable_sales = df_plat_year_count.loc[df_plat_year_count['Type'] == "Portable"].sort_values(by='Year', ascending=False)

    #on va compter le nombre d'années par Platform
    df_portable_year_count = df_portable_sales.groupby(['Platform']).size().reset_index(name='Count').sort_values(by='Count', ascending=False)

    fig = px.bar(df_portable_year_count,
                y='Platform', x='Count',
                hover_data=['Platform', 'Count'],
                color='Count',
                labels={'Platform'}, height=800, width=800,orientation="h",
                color_continuous_scale=px.colors.sequential.Inferno)


    fig.update_layout(
        title='durée moyenne du marché de développement d\'un jeu sur une plateforme',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title=dict(
                text="Plateformes Portables",
                font=dict(
                    size=16
                    )
                ),
            tickfont_size=14,
            ),
        legend=dict(
            x=0,
            y=1,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
            ),
        xaxis=dict(
            title=dict(
                text="Nbre années",
                font=dict(
                    size=16
                    )
                ),
            tickfont_size=14,
            ),

        )
    st.plotly_chart(fig)
####################################### PAGE 2 (MODELISATION (BASE)) ###################################
if page == pages[2] : 

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

  st.write("### Répartition de la variable Global_Sales")
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

  fig.update_layout(width=800,height=400)

  # Afficher le graphique dans Streamlit
  st.plotly_chart(fig)

  ### FIN DE LA PARTIE 2

  ### PARTIE 3
  ### APPLICATION DE LA METHODE BOX-COX ET VISUALISATION DE LA TRANSFORMATION
  st.write("### Application de la méthode Box-Cox sur la variable cible")
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

  fig.update_layout(width=800,height=400)
  st.write("Nous voyons l'effet de la 'normalisation' de la variable cible plus clairement sur les graphiques ci-dessous.")
  # Afficher le graphique dans Streamlit
  st.plotly_chart(fig)

  ### FIN DE LA PARTIE 3

  ### PARTIE 4
  ### FEATURE ENGINEERING, CREATION DE NOUVELLES VARIABLES
  st.write("### Feature Engineering à partir des données de base")
  st.write("Compte tenu du nombre limité de variables à disposition, nous avons essayé d'en ajouter de nouvelles à partir des existantes.")
  st.write("Période d'existence au sein du jeu de données des éditeurs et des plateformes ainsi que des associations potentiellement utiles.")
  
  ### checkbox pour afficher le code
  afficher_code = st.checkbox('Afficher le code')
  
  code = '''
  def assign_longevite(group):
    plat_long = group.max() - group.min()
    return plat_long

  df['Game_Sales_Period'] = df.groupby('Platform')['Year'].transform(assign_longevite)
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


  st.dataframe(df.head())
  


####################################### PAGE 3 (MACHINE LEARNING (BASE)) ###################################
if page == pages[3] :
  ########################################################## CODE POUR LA PAGE 3 ##########
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

  ############################################################################# FIN DU CODE POUR LA PAGE 3 ########################################
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
  st.write("### Feature Engineering à partir des données de base")
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

  ### X_non_scaled
  X_train_non_scaled = scaler.fit_transform(X_train_non_scaled)
  X_test_non_scaled = scaler.transform(X_test_non_scaled)


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

      st.write('# Valeurs réelles - Valeurs résiduelles:\n\n')

      le_dict = {
        'Global_Sales': global_sales_lambda
      }
      y_pred = inv_boxcox(result, [global_sales_lambda])
      y_test = inv_boxcox(y_test_scaled_trans, [global_sales_lambda])

      residuals = y_test['x0'] - y_pred

      comparison_df = pd.DataFrame({'Valeurs Réelles': y_test['x0'], 'Valeurs Prédites': y_pred,'Residuals': residuals})
      comparison_df.sort_values(by='Valeurs Réelles',ascending=True, inplace=True)  
      

      #Création de 2 colonnes dans streamlit
      col1, col2 = st.columns(2)
      with col1:
        st.write("10 plus petites valeurs réelles")
        st.dataframe(comparison_df.head(10))
      with col2:
        st.write("10 plus grandes valeurs réelles")
        st.dataframe(comparison_df.tail(10))
       
      st.dataframe(comparison_df.describe())
      
      fig = px.scatter(comparison_df, y="Residuals", x="Valeurs Réelles")
      
      fig.update_layout(width=800,height=400)
      
      st.plotly_chart(fig)

      st.write('# Valeurs réelles - Valeurs prédites:\n\n')
      fig = px.scatter(comparison_df, x="Valeurs Réelles", y="Valeurs Prédites")
      
      fig.update_layout(width=800,height=400)
      
      st.plotly_chart(fig)
      

      st.write('# SHAP values:\n\n')
      shap_values_test = shap.TreeExplainer(model).shap_values(X_test_scaled)

      X_test_scaled_array = X_test_scaled
      X_test_scaled_array = X_test_scaled.values
      plt.figure()
      shap.summary_plot(shap_values_test, X_test_scaled_array, feature_names=X_test_scaled.columns)
      st.pyplot(plt)

      
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

      comparison_df = pd.DataFrame({'Valeurs Réelles': y_test['x0'], 'Valeurs Prédites': y_pred,'Residuals': residuals})
      comparison_df.sort_values(by='Valeurs Réelles',ascending=True, inplace=True)  
      
      col1, col2 = st.columns(2)
      with col1:
        st.write("10 plus petites valeurs")
        st.dataframe(comparison_df.head(10))
      with col2:
        st.write("10 plus grandes valeurs")
        st.dataframe(comparison_df.tail(10))
        
      st.dataframe(comparison_df.describe())

      fig = px.scatter(comparison_df, y="Residuals", x="Valeurs Réelles")
      
      fig.update_layout(width=800,height=400)
      
      st.plotly_chart(fig)

      st.write('# Valeurs réelles - Valeurs prédites:\n\n')
      fig = px.scatter(comparison_df, x="Valeurs Réelles", y="Valeurs Prédites")
      
      fig.update_layout(width=800,height=400)
      
      st.plotly_chart(fig)
      



      st.write('# SHAP values:\n\n')
      shap_values_test = shap.TreeExplainer(model).shap_values(X_test_scaled)

      X_test_scaled_array = X_test_scaled
      X_test_scaled_array = X_test_scaled.values
      plt.figure(figsize=(6,12))
      shap.summary_plot(shap_values_test, X_test_scaled_array, feature_names=X_test_scaled.columns,show=False)
      st.pyplot(plt)

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


      comparison_df = pd.DataFrame({'Valeurs Réelles': y_test, 'Valeurs Prédites': y_pred,'Residuals': residuals})
      comparison_df.sort_values(by='Valeurs Réelles',ascending=True, inplace=True)  

      col1, col2 = st.columns(2)
      with col1:
        st.write("10 plus petites valeurs")
        st.dataframe(comparison_df.head(10))
      with col2:
        st.write("10 plus grandes valeurs")
        st.dataframe(comparison_df.tail(10))

      st.dataframe(comparison_df.describe())
      
      fig = px.scatter(comparison_df, y="Residuals", x="Valeurs Réelles")
      
      fig.update_layout(width=800,height=400)
      
      st.plotly_chart(fig)

      st.write('# Valeurs réelles - Valeurs prédites:\n\n')
      fig = px.scatter(comparison_df, x="Valeurs Réelles", y="Valeurs Prédites")
      
      fig.update_layout(width=800,height=400)
      
      st.plotly_chart(fig)

      
      
      st.write('# SHAP values:\n\n')
      shap_values_test = shap.TreeExplainer(model).shap_values(X_test_non_scaled)

      X_test_non_scaled_array = X_test_non_scaled
      X_test_non_scaled_array = X_test_non_scaled.values
      plt.figure()
      shap.summary_plot(shap_values_test, X_test_non_scaled_array, feature_names=X_test_non_scaled.columns)
      st.pyplot(plt)

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

      comparison_df = pd.DataFrame({'Valeurs Réelles': y_test, 'Valeurs Prédites': y_pred,'Residuals': residuals})
      comparison_df.sort_values(by='Valeurs Réelles',ascending=True, inplace=True)  

      col1, col2 = st.columns(2)
      with col1:
        st.write("10 plus petites valeurs")
        st.dataframe(comparison_df.head(10))
      with col2:
        st.write("10 plus grandes valeurs")
        st.dataframe(comparison_df.tail(10))
      
      st.dataframe(comparison_df.describe())
      
      fig = px.scatter(comparison_df, y="Residuals", x="Valeurs Réelles")
      
      fig.update_layout(width=800,height=400)
      
      st.plotly_chart(fig)

      st.write('# Valeurs réelles - Valeurs prédites:\n\n')
      fig = px.scatter(comparison_df, x="Valeurs Réelles", y="Valeurs Prédites")
      
      fig.update_layout(width=800,height=400)
      
      st.plotly_chart(fig)

      

      st.write('# SHAP values:\n\n')
      shap_values_test = shap.TreeExplainer(model).shap_values(X_test_non_scaled)

      X_test_non_scaled_array = X_test_non_scaled
      X_test_non_scaled_array = X_test_non_scaled.values
      plt.figure()
      shap.summary_plot(shap_values_test, X_test_non_scaled_array, feature_names=X_test_non_scaled.columns)
      st.pyplot(plt)

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


  


