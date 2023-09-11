
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

def automate_preprocessing(df,categorical_cols,numerical_cols):
     
       for col in categorical_cols:
        if df[col].isnull().sum() > 0:

            option = st.selectbox(f"Handling missing values in '{col}'", ['Most Frequent', 'Additional Class'])
            if option == 'Most Frequent':
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif option == 'Additional Class':
                df[col].fillna('Missing', inplace=True) 

       for col in numerical_cols:
          if df[col].isnull().sum() > 0:

            option = st.selectbox(f"Handling missing values in '{col}'", ['Mean', 'Median', 'Mode'])
            if option == 'Mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif option == 'Median':
                df[col].fillna(df[col].median(), inplace=True)
            elif option == 'Mode':
                df[col].fillna(df[col].mode()[0], inplace=True)        

       return st.dataframe(df)
with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML")
    choice = st.radio("Navigation", ["Upload","EDA","Modelling","Evaluation" ,"Download"])
    st.info("This project application helps you build and explore your data.")


if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == 'EDA':
     st.title("Exploratory Data Analysis")
     for coln in df.columns:
         st.write(f"name column :'{coln}'")
         st.write(df[coln].dtype)
     st.info("number null for columns ")
     st.write(df.isna().sum())
     st.info("number duplicated for Dataset,then it will be deleted automatically ")
     st.write(df.duplicated().sum())
     df.drop_duplicates()
     st.info('when you want to drop column')

     btn = st.button('determine column to drop ') 
     if btn:
         col_drop = st.text_input('name column')
         df.drop(col_drop,axis = 1, inplace = True)



     categorical_cols = df.select_dtypes(include=['object', 'category','bool']).columns
     numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns  

     st.info('click to handling missing value ')
     automate_preprocessing(df,categorical_cols,numerical_cols) 
     st.info("number null for columns ")
     st.write(df.isna().sum())

     option = st.radio(f" Choose type visualization to Data  ", ['Plot','Hieatmap', 'Scatter', 'Bar','Histogram','Violin plot','Box plot'])
     if option== 'Heatmap':
         fig = plt.figure(figsize=(12,6))
         sns.heatmap(df.corr(),annot = True)
         st.pyplot(fig)
     if option == 'Plot':
         fig = plt.figure(figsize=(12,6))
         option_x = st.selectbox('Choose column axes x',numerical_cols)
         option_y = st.selectbox('Choose column axes y',numerical_cols)
         plt.plot(df[option_x],df[option_y],c = 'r')
         plt.title(f"'{option_x}' VS '{option_y}'")
         plt.xlabel(f"'{option_x}'")
         plt.ylabel(f"'{option_y}'")
         st.pyplot(fig)
         
           
     if option== 'Histogram':
         option_1 = st.selectbox('Choose column',df.columns)
         fig = px.histogram(df[option_1])
         st.plotly_chart(fig)

     if option== 'Scatter':
         option_x1 = st.selectbox('Choose column axes x',numerical_cols)
         option_y1 = st.selectbox('Choose column axes y',numerical_cols)
         fig = px.scatter(data_frame=df,x = option_x1,y = option_y1)
         st.plotly_chart(fig) 

     if option== 'Bar':
         option_2 = st.selectbox('Choose column',df.columns)
         fig = px.bar(df[option_2])
         st.plotly_chart(fig)    
     if option== 'Violin plot':
         option_3 = st.selectbox('Choose column',numerical_cols)
         fig = px.violin(df[option_3])
         st.plotly_chart(fig) 
     if option== 'Box plot':
        option_4 = st.selectbox('Choose column',numerical_cols)
        fig = px.box(df[option_4])
        st.plotly_chart(fig)         


if choice == "Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    df.dropna(subset=[chosen_target], inplace=True)
    if df[chosen_target].dtype == 'int64' or 'float64':
       from pycaret.regression import *
    elif df[chosen_target].dtype == 'object' or 'bool':
       from pycaret.classification import *
    train_size = st.slider('determine train set size',1,100) 
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target,train_size = train_size*0.01)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        st.info('choose the best model for you')
        btn = st.button('create model')
        if btn:
            chosen_model = st.text_input('Enter name model from column abbreviations ')
            cremod = create_model(chosen_model)
            cremod_2 = pull()
            st.dataframe(cremod_2)

if choice == "Evaluation":
    tunmod = tune_model(cremod,choose_better = True )
    evaluate_model(tunmod)
    save_model(tunmod,'best_model')
    

if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="trainee_model.pkl")