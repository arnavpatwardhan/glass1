import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

X = glass_df.iloc[:, :-1]

y = glass_df['GlassType']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

@st.cache()
def prediction(model,ri,na,mg,al,si,k,ca,ba,fe):
    glass_type=model.predict([[ri,na,mg,al,si,k,ca,ba,fe]])
    glass_type=glass_type[0]
    if glass_type==1:
        return "building windows float processed"
    elif glass_type==2:
        return "building windows non float processed"
    elif glass_type==3:
        return "vehicle windows float processed"
    elif glass_type==4:
        return "vehicle windows non float processed"
    elif glass_type==5:
        return "containers"
    elif glass_type ==6:
        return "tableware"
    else:
        return "headlapse"
st.title("glass type predictor")
st.sidebar.title("explorary data analysis")
if st.sidebar.checkbox('show raw data'):
    st.subheader("full data set")
    st.dataframe(glass_df)

st.sidebar.subheader('scatterplot')

features_list =st.sidebar.multiselect("select x axis values",("RI","Na","Mg","Al","Si","K","Ca","Ba","Fe"))
st.set_option("deprecation.showPyplotGlobalUse",False)
for feature in features_list:
    st.subheader(f"scatterplot between {feature} and glass_type")
    plt.figure(figsize=(12,6)) 
    sns.scatterplot(x=feature,y="GlassType",data=glass_df)
    st.pyplot()

st.sidebar.subheader('histogram')

features_hist =st.sidebar.multiselect("select features",("RI","Na","Mg","Al","Si","K","Ca","Ba","Fe"))

for feature in features_hist:
    st.subheader(f"histogram between {feature} and glass_type")
    plt.figure(figsize=(12,6)) 
    plt.hist(glass_df[feature],bins='sturges',edgecolor='black')
    st.pyplot()
plot_types = st.sidebar.multiselect("select the charts or plots",('boxplot','countplot','piechart','heatmap','pairplot'))
if "boxplot" in plot_types:
    st.subheader('boxplot')
    columns= st.sidebar.selectbox('select the column to create the boxplot',("RI","Na","Mg","Al","Si","K","Ca","Ba","Fe",'glass_type'))
    plt.figure(figsize=(12,6))
    plt.title(f"boxplot for {columns}")
    sns.boxplot(glass_df[columns])
    st.pyplot()

if "countplot" in plot_types:
    st.subheader('countplot')
    columns= st.sidebar.selectbox('select the column to create the boxplot',("RI","Na","Mg","Al","Si","K","Ca","Ba","Fe",'glass_type'))
    plt.figure(figsize=(12,6))
    plt.title(f"countplot for {columns}")
    sns.countplot(glass_df['GlassType'])
    st.pyplot()

if "piechart" in plot_types:
    st.subheader('piechart')
    pie_data = glass_df['GlassType'].value_counts()
    columns= st.sidebar.selectbox('select the column to create the boxplot',("RI","Na","Mg","Al","Si","K","Ca","Ba","Fe",'glass_type'))
    plt.figure(figsize=(12,6))
    plt.title(f"countplot for {columns}")
    plt.pie(pie_data,labels=pie_data.index,autopct='%1.2f%%',startangle=30,explode=np.linspace(0.06,0.16,6))
    st.pyplot()

if "heatmap" in plot_types:
    st.subheader('heatmap')
    columns= st.sidebar.selectbox('select the column to create the boxplot',("RI","Na","Mg","Al","Si","K","Ca","Ba","Fe",'glass_type'))
    plt.figure(figsize=(12,6))
    plt.title(f"heatmap for {columns}")
    sns.heatmap(glass_df.corr(),annot=True)
    st.pyplot()
if "pairplot" in plot_types:
    st.subheader('pairplot')
    columns= st.sidebar.selectbox('select the column to create the boxplot',("RI","Na","Mg","Al","Si","K","Ca","Ba","Fe",'glass_type'))
    plt.figure(figsize=(12,6))
    plt.title(f"pairplot for {columns}")
    sns.pairplot(glass_df)
    st.pyplot()