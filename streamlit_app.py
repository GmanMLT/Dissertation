import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from performancedataframe import dfperformancemetrics
import shap
import joblib
import os
import plotly.graph_objects as go
from yfiles_graphs_for_streamlit import StreamlitGraphWidget, Node, Edge
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget


dfperformancemetrics = pd.read_csv('dfperformancemetrics.csv', index_col =0)
dfperformanceacc = pd.read_csv('dfperformanceacc.csv', index_col =0)

print(dfperformanceacc)

st.set_page_config(
    page_title="HR Attrition Tool",  # the page title shown in the browser tab
    layout="wide",  # page layout : use the entire screen
)


#Navigation
page = st.sidebar.radio(
    "Navigate",
    ["Disclaimer", "Key Correlation Plots", "Organisational Level", "Departamental Level", "Employee Level", "Model Performance", "Employee Graph"]
)

if page == "Disclaimer":


    # add page title
    st.title("HR Attrition Strategy")



    #About the ML model

    with st.expander("Disclaimer"):
          st.subheader("Intended Use:")
          st.write("The Machine Learning model is solely intended to support HR practitioners develop/modify Attrition Strategies.")
          st.write("The Machine Learning model is a support tool and thus the final decision shall always rest with human HR practitioners.")
          st.subheader("Model Performance:")
          st.write("No Machine Learning model is capable of predicting outcomes perfectly.")
          st.write("The prediction ability of a Machine Learning Model will vary in accordance with its training/testing dataset.")
          st.write("Prior to using any Machine Learning Model it is important to understand the Model Performance Metrics.")

elif page == "Model Performance":

          st.title("Model Performance")
          stayclasspre = (dfperformancemetrics['Stay Class'].values[0].round(2))
          stayclassrec = (dfperformancemetrics['Stay Class'].values[1].round(2))
          stayclassF1 = (dfperformancemetrics['Stay Class'].values[2].round(2))
          leaveclasspre = (dfperformancemetrics['Leave Class'].values[0].round(2))
          leaveclassrec = (dfperformancemetrics['Leave Class'].values[1].round(2))
          leaveclassF1 = (dfperformancemetrics['Leave Class'].values[2].round(2))




          col1, col2, col3 = st.columns(3)
          col4, col5, col6 = st.columns(3)


          col1.metric(
          label="Stay Class Precision",
          value=f"{stayclasspre}")

          col2.metric(
          label="Stay Class Recall",
          value=f"{stayclassrec}")

          col3.metric(
          label="Stay Class F1",
          value=f"{stayclassF1}")

          col4.metric(
          label="Leave Class Precision",
          value=f"{leaveclasspre}")

          col5.metric(
          label="Leave Class Recall",
          value=f"{leaveclassrec}")

          col6.metric(
          label="Leave Class F1",
          value=f"{leaveclassF1}")


          #st.dataframe(
          #dfperformancemetrics,
          #width="content",
          #height="content"
          #)


          st.metric(
          label="Overall Accuracy",
          value=f"{dfperformanceacc.iloc[0, 0]:.1%}"
          )



       
   #       st.header("About the Model")

    #      st.write("Accuracy is a global measure of how often the model correctly predicts the employee's status (leave / stay) ")
    #      st.write("Recall provides an indication of correctly identified instances i.e. when an employee at risk of attrition is identified as not being at risk of attrition")
    #      st.write("Precision provides an indication of predicted positive instances that are actually correct. It is important to know when the model may misidentify employees who plan on staying as being an attrition risk")





