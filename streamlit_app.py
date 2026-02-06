import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import joblib
import os
import plotly.graph_objects as go
from yfiles_graphs_for_streamlit import StreamlitGraphWidget, Node, Edge
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget





#Page title = the title shown in the browser tab. 
#layout = wide, page elements use the entire screen width. 
st.set_page_config(
    page_title="HR Attrition Tool",  # the page title shown in the browser tab
    layout="wide",  # page layout : use the entire screen
)


#Navigation
#Various elements passed to sidebar through object notation, are pinned to the left 
# in the form of a radio widget. The parameter Navigation is simply a label to guide users. 
page = st.sidebar.radio(
    "Navigation",
    ["Disclaimer", "Key Correlation Plots", "Organisational Level", "Departamental Level", "Employee Level", "Model Performance", "Employee Graph"]
)


#Simple if and elif conditional statements are used to navigate acoss different pages


if page == "Disclaimer":


    # Page Title 
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




elif page == "Key Correlation Plots":

       
       


        #Page Title 
        st.title("Key Correlation Plots")

        #Page Text
        st.write("Kindly find the top 4 most highly correleated features with respect to attrition label")

        # (Overtime, JobRole, JobLevel, StockOptionLevel) with respect to the attrition label")
        st.write("1. Overtime  \n2. JobRole  \n3. JobLevel  \n4. StockOptionLevel")


        ##Retrieve data
        dfcorreleations = pd.read_csv('data/dfcorreleations.csv', index_col =0)

        #Overtime versus Attrition 
        st.subheader("Overtime versus Attrition")
        st.write("Overtime is associated with the Strategic Category **'Work Enviroment'** ")
        
        res = pd.crosstab(dfcorreleations.OverTime, dfcorreleations.Attrition)
        
        #Display Dataframe res as a table 
        st.dataframe(res)



        #plot cross tabulation 
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



      # Job Role Versus Attrition 
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

        # JobLevel Versus Attrition 
        st.subheader("JobLevel versus Attrition")
        st.write("JobLevel is associated with the Strategic Category **'Management Culture'** ")
        res3 = pd.crosstab(dfcorreleations.JobLevel, dfcorreleations.Attrition)
        st.dataframe(res3)



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


        # StockOptionLevel Versis Attrotopm 
        st.subheader("StockOptionLevel versus Attrition")
        st.write("StockOptionLevel is associated with the Strategic Category **'Management Culture'** ")
        res4 = pd.crosstab(dfcorreleations.StockOptionLevel, dfcorreleations.Attrition)
        st.dataframe(res4)

        

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







elif page == "Model Performance":

          #Retrieve relevant data and place in dataframe
          dfperformancemetrics = pd.read_csv('dfperformancemetrics.csv', index_col =0)
          dfperformanceacc = pd.read_csv('dfperformanceacc.csv', index_col =0)     


          st.title("Model Performance")
          
          #Put relevant datapoints into variables 
          stayclasspre = (dfperformancemetrics['Stay Class'].values[0].round(2))
          stayclassrec = (dfperformancemetrics['Stay Class'].values[1].round(2))
          stayclassF1 = (dfperformancemetrics['Stay Class'].values[2].round(2))
          leaveclasspre = (dfperformancemetrics['Leave Class'].values[0].round(2))
          leaveclassrec = (dfperformancemetrics['Leave Class'].values[1].round(2))
          leaveclassF1 = (dfperformancemetrics['Leave Class'].values[2].round(2))



          # Creating a grid of 3 columns per row (across 2 rows) 
          # of equal width to display results across stay and leave classes 
          col1, col2, col3 = st.columns(3)
          col4, col5, col6 = st.columns(3)

          #Placing metric in reserved grid space
          #label = heaer / title of metric 
          #Value of metric 
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


       
          st.metric(
          label="Overall Accuracy",
          value=f"{dfperformanceacc.iloc[0, 0]}"
          )




elif page == "Organisational Level":
  st.title("Attrition Causes Organisational Level")

  #Retrieve SHAP for organisational level  
  st.image(
    "graph/organisation/Organisation_attrition.jpg",
    caption="Organisational SHAP plot showing top features influencing attrition (Class 1)",
    use_container_width=True
)
      
   


elif page == "Employee Graph":

  #Retrieve data containing employee and their job role  
  from_csv_leavers_graph = pd.read_csv('dfleaversgraph.csv', index_col =0)

  # Store unqique employees and job roles 
  nodes = []
  
  # Stores id's assigned to nodes to support edge creation 
  node_id_map = {}  
  
  #Represents node IDs 
  current_id = 0


    #Step 1: Creation of Nodes 
    # Create a node for each unique employee incrementing the id which each new unique employee 
    #The node should store the employee number 
    #The node is of type employee 
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
  #Iterate through each row in CSV i.e. all employee 
  #Get row data, Employee Number and Job Role and use this to retrieve previous mappings
  for _, row in from_csv_leavers_graph.iterrows():
      emp_id = node_id_map[row['EmployeeNumber']]
      role_id = node_id_map[row['JobRole']]

      #Create a link between EMPid and roleid 
      edges.append(Edge(start=emp_id, end=role_id, properties={"label": "has_role"}))

  # Step 3: Render
  StreamlitGraphWidget(nodes=nodes, edges=edges).show()

elif page == "Departamental Level":
   st.title("Attrition Causes Departamental Level")





# Options between 3 departments 

   option = st.radio(
        "Choose a Departments Attrition Causes to Display:",
        ("Human Resources", "Research and Development", "Sales"),
        horizontal = 1,
        width = "content"

   )
   
   #Display relevant graphs related to selected departments. 

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

#Let user view either all leavers or dive into a single leaver in detail 
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



        #Display all leavers sorted by chance of leaving                        
      st.dataframe(leavers_sorted_df, hide_index=True)






   elif option == "Single Leaver":
      st.title("Single Leaver")
      
        #list of leavers, to select by employee number 
      leavers_sorted_df = pd.read_csv('leavers_sorted.csv', index_col = 0)
      employeenumber = st.selectbox("EmployeeNumbers", leavers_sorted_df['EmployeeNumber'])
  
      #Dynamic image path based on employee number   
      image_path = f"{employeenumber}_Employee_attrition.jpg"

      
      st.image(
          image_path,
          caption=f"Attrition Causes – Employee {employeenumber}"
      )
















