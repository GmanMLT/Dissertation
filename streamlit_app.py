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
          value=f"{dfperformanceacc.iloc[0, 0]}"
          )

elif page == "Key Correlation Plots":
        st.title("Key Correlation Plots")

        st.write("Kindly find the top 4 most highly correleated features with respect to attrition label")

        # (Overtime, JobRole, JobLevel, StockOptionLevel) with respect to the attrition label")
        st.write("1. Overtime  \n2. JobRole  \n3. JobLevel  \n4. StockOptionLevel")


        ##Retrieve data
        dfcorreleations = pd.read_csv('dfcorreleations.csv', index_col =0)


        st.subheader("Overtime versus Attrition")
        st.write("Overtime is associated with the Strategic Category **'Work Enviroment'** ")
        #Diagram 1
        res = pd.crosstab(dfcorreleations.OverTime, dfcorreleations.Attrition)
        st.dataframe(res)


   #     st.write("###Heatmap")

        fig, ax = plt.subplots(figsize=(6, 4))

        sns.heatmap(
            pd.crosstab(
                dfcorreleations["OverTime"],
                dfcorreleations["Attrition"],
                normalize="index"
            ),
            annot=True,
            cmap="Blues",
            ax=ax
        )

        ax.set_xlabel("Attrition")
        ax.set_ylabel("OverTime")

        st.pyplot(fig, width = "stretch")



      #'Diagram2 '
        st.subheader("JobRole versus Attrition")
        st.write("JobRole is associated with the Strategic Category **'Work Enviroment'** ")

        res2 = pd.crosstab(dfcorreleations.JobRole, dfcorreleations.Attrition)
        st.dataframe(res2)



        fig, ax = plt.subplots(figsize=(6, 4))

        sns.heatmap(
            pd.crosstab(
                dfcorreleations["JobRole"],
                dfcorreleations["Attrition"],
                normalize="index"
            ),
            annot=True,
            cmap="Blues",
            ax=ax
        )

        ax.set_xlabel("Attrition")
        ax.set_ylabel("JobRole")

        st.pyplot(fig, width = "stretch")

        ## Job Level
        st.subheader("JobLevel versus Attrition")
        st.write("JobLevel is associated with the Strategic Category **'Management Culture'** ")
        res3 = pd.crosstab(dfcorreleations.JobLevel, dfcorreleations.Attrition)
        st.dataframe(res3)

        #st.write("###Heatmap")

        fig, ax = plt.subplots(figsize=(6, 4))

        sns.heatmap(
            pd.crosstab(
                dfcorreleations["JobLevel"],
                dfcorreleations["Attrition"],
                normalize="index"
            ),
            annot=True,
            cmap="Blues",
            ax=ax
        )

        ax.set_xlabel("Attrition")
        ax.set_ylabel("JobLevel")

        st.pyplot(fig, width = "stretch")


        ## StockOptionLevel
        st.subheader("StockOptionLevel versus Attrition")
        st.write("StockOptionLevel is associated with the Strategic Category **'Management Culture'** ")
        res4 = pd.crosstab(dfcorreleations.StockOptionLevel, dfcorreleations.Attrition)
        st.dataframe(res4)

        #st.write("###Heatmap")

        fig, ax = plt.subplots(figsize=(6, 4))

        sns.heatmap(
            pd.crosstab(
                dfcorreleations["StockOptionLevel"],
                dfcorreleations["Attrition"],
                normalize="index"
            ),
            annot=True,
            cmap="Blues",
            ax=ax
        )

        ax.set_xlabel("Attrition")
        ax.set_ylabel("StockOptionLevel")

        st.pyplot(fig, width = "stretch")


elif page == "Organisational Level":
  st.title("Attrition Causes Organisational Level")

  st.image(
    "Organisation_attrition.jpg",
    caption="SHAP violin plot showing top features influencing attrition (Class 1)",
    use_container_width=True
)
       
   #       st.header("About the Model")

    #      st.write("Accuracy is a global measure of how often the model correctly predicts the employee's status (leave / stay) ")
    #      st.write("Recall provides an indication of correctly identified instances i.e. when an employee at risk of attrition is identified as not being at risk of attrition")
    #      st.write("Precision provides an indication of predicted positive instances that are actually correct. It is important to know when the model may misidentify employees who plan on staying as being an attrition risk")


elif page == "Employee Graph":


  from_csv_leavers_graph = pd.read_csv('dfleaversgraph.csv', index_col =0)

  # Step 1: Create unique nodes
  nodes = []
  node_id_map = {}  # maps value -> node id
  current_id = 0



    # Employees
  for emp in from_csv_leavers_graph['EmployeeNumber'].unique():
      nodes.append(Node(id=current_id, properties={"label": str(emp), "type": "employee"}))
      node_id_map[emp] = current_id
      current_id += 1


  # JobRoles
  for role in from_csv_leavers_graph['JobRole'].unique():
      nodes.append(Node(id=current_id, properties={"label": str(role), "type": "jobrole"}))
      node_id_map[role] = current_id
      current_id += 1

  # Step 2: Create edges
  edges = []
  for _, row in from_csv_leavers_graph.iterrows():
      emp_id = node_id_map[row['EmployeeNumber']]
      role_id = node_id_map[row['JobRole']]

      
      edges.append(Edge(start=emp_id, end=role_id, properties={"label": "has_role"}))

  # Step 3: Render
  StreamlitGraphWidget(nodes=nodes, edges=edges).show()

elif page == "Departamental Level":
   st.title("Attrition Causes Departamental Level")





# 1️⃣ Create a selection widget

   option = st.radio(
        "Choose a Departments Attrition Causes to Display:",
        ("Human Resources", "Research and Development", "Sales"),
        horizontal = 1,
        width = "content"

   )

   if option == "Human Resources":
     st.image(
     "HR_DEPARTMENT.jpg",
     caption="HR attrition Causes")
   elif option == "Research and Development":
     st.image(
     "RD_DEPARTMENT.jpg",
     caption="RD attrition Causes")
   elif option == "Sales":
     st.image(
     "SALES_DEPARTMENT.jpg",
     caption="Sales attrition Causes")


elif page == "Employee Level":
   st.title("Leavers")


   option = st.radio(
        "View all potential Leavers or Focus on a Single Leaver:",
        ("All Leavers", "Single Leaver"),
        horizontal = 1,
        width = "content"

   )

   if option == "All Leavers":

      st.subheader("List of Employees expected to leave:")

      leavers_sorted_df = pd.read_csv('Leavers_sorted.csv', index_col = 0)
      leavers_sorted_df = leavers_sorted_df[['EmployeeNumber', 'LeaveProbability']]
      leavers_sorted_df = leavers_sorted_df.sort_values(by = 'LeaveProbability', ascending=False)




      st.dataframe(leavers_sorted_df, hide_index=True)






   elif option == "Single Leaver":
      st.title("Single Leaver")

      leavers_sorted_df = pd.read_csv('leavers_sorted.csv', index_col = 0)
      employeenumber = st.selectbox("EmployeeNumbers", leavers_sorted_df['EmployeeNumber'])
      #st.title(f"{employeenumber}")

      image_path = f"{employeenumber}_Employee_attrition.jpg"

      if os.path.exists(image_path):
        st.image(
          image_path,
          caption=f"Attrition Causes – Employee {employeenumber}"
      )












