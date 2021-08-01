import streamlit as st
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


st.set_option('deprecation.showfileUploaderEncoding', False)

upload_file1, upload_file2 = st.beta_columns(2)
btn1, btn2 = st.beta_columns(2)
button1, button2 = st.beta_columns(2)

ch = ["None", "replace with Mean", "replace with Median", "replace with 0", "remove entire column",
      "remove entire row"]
uploaded_file1 = st.sidebar.file_uploader(label='', type=['csv'])
uploaded_file2 = st.sidebar.file_uploader(label=' ', type=['csv'])

if uploaded_file1 is not None:
    df1 = pd.read_csv(uploaded_file1, encoding='unicode_escape')
    upload_file1.success("Dataset 1 had been uploaded")
    col = btn1.checkbox("Descriptive statistics")
    if col:
        st.subheader("DESCRIPTIVE STATISTICS FOR DATASET 1")
        st.table(df1.describe())
    else:
        upload_file1.info("you didn't select any option")
    data_frame = df1
    miss_val = btn1.checkbox("Missing Values")
    if miss_val:
        select = st.sidebar.selectbox("Select any one for dataset 1", ch)
        if select == "replace with Mean":
            st.subheader("Missing values for dataset 1")
            mv_1 = df1.fillna(df1.mean())
            data_frame = mv_1
            st.write(mv_1)
        elif select == "replace with Median":
            st.subheader("Missing values for dataset 1")
            mv_1 = df1.fillna(df1.median())
            data_frame = mv_1
            st.write(mv_1)
        elif select == "replace with 0":
            st.subheader("Missing values for dataset 1")
            mv_1 = df1.fillna(0)
            data_frame = mv_1
            st.write(mv_1)
        elif select == "remove entire row":
            st.subheader("Missing values for dataset 1")
            mv_1 = df1.dropna()
            data_frame = mv_1
            st.write(mv_1)
        elif select == "remove entire column":
            st.subheader("Missing values for dataset 1")
            mv_1 = df1.dropna(axis=1)
            data_frame = mv_1
            st.write(mv_1)
        else:
            upload_file1.info("Please select any one")

    math_functions = button1.checkbox("Apply math operation")
    if math_functions:
        math_function = ["None", "basic", "compare","round"]
        oper = st.sidebar.selectbox("select a function", math_function)
        if oper == "basic":
            st.subheader("basic operations for dataset 1")
            all_columns = data_frame.columns.tolist()
            col1 = st.sidebar.selectbox("choose the variable ", all_columns)
            col2 = st.sidebar.selectbox('choose the variable  ', all_columns)
            option = ['none', 'Add', 'Sub', 'Mul', 'Div']
            arithmetic = st.sidebar.selectbox("select the arithmetic operator ", option)

            if arithmetic == "Add":


                data_frame['add']=data_frame[col1]+data_frame[col2]

                st.write(data_frame)


            elif arithmetic == "Sub":
                st.header("subtract")

                data_frame['sub'] = data_frame[col1] - data_frame[col2]

                st.write(data_frame)

            elif arithmetic == "Mul":
                st.header("Multiply")


                data_frame['Mul'] = data_frame[col1] * data_frame[col2]

                st.write(data_frame)

            elif arithmetic == "Div":
                st.header("Divivide")


                data_frame['Div'] = data_frame[col1] / data_frame[col2]
                st.write(data_frame)
        elif oper == "compare":
            st.subheader("comparison operations for dataset 1")
            all_columns = data_frame.columns.tolist()
            col3 = st.sidebar.selectbox("choose the variable for A ", all_columns)
            col4 = st.sidebar.selectbox("choose the variable for B ", all_columns)
            conditions = [data_frame[col3] > data_frame[col4],
                          data_frame[col3] < data_frame[col4],
                          data_frame[col3] == data_frame[col4]]

            # define choices
            choices = ['TRUE', 'FALSE', 'EQUAL']

            # create new column in DataFrame that displays results of comparisons
            data_frame['comparison'] = np.select(conditions, choices)
            data_frame['Diff'] = np.where(data_frame[col3] == data_frame[col4], 0, data_frame[col3] - data_frame[col4])
            st.write(data_frame)



if uploaded_file2 is not None:
    df2 = pd.read_csv(uploaded_file2, encoding='unicode_escape')
    upload_file2.success("Dataset 2 had been uploaded")
    col_2 = btn2.checkbox("Descriptive  statistics")
    if col_2:
        st.subheader("DESCRIPTIVE STATISTICS FOR DATASET 2")
        st.table(df2.describe())
    data_frame_2 = df2
    miss_val_1 = btn2.checkbox("Missing  Values")
    if miss_val_1:
        select_1 = st.sidebar.selectbox("Select any one for dataset 2", ch)
        if select_1 == "replace with Mean":
            st.subheader("Missing values for dataset 2")
            mv_2 = df2.fillna(df2.mean())
            data_frame_2 = mv_2
            st.write(mv_2)
        elif select_1 == "replace with Median":
            st.subheader("Missing values for dataset 2")
            mv_2 = df2.fillna(df2.median())
            data_frame_2 = mv_2
            st.write(mv_2)
        elif select_1 == "replace with 0":
            st.subheader("Missing values for dataset 2")
            mv_2 = df2.fillna(0)
            data_frame_2 = mv_2
            st.write(mv_2)
        elif select_1 == "remove entire row":
            st.subheader("Missing values for dataset 2")
            mv_2 = df2.dropna()
            data_frame_2 = mv_2
            st.write(mv_2)
        elif select_1 == "remove entire column":
            st.subheader("Missing values for dataset 2")
            mv_2 = df2.dropna(axis=1)
            data_frame_2 = mv_2
            st.write(mv_2)
        else:
            upload_file2.info("Please select any one")

    math_functions = button2.checkbox("Apply math operation ")
    if math_functions:
        math_function=["None", "basic", "compare"]
        oper = st.sidebar.selectbox("select a function ", math_function)
        if oper == "basic":
            st.subheader("basic operations for dataset 2")
            all_columns = data_frame_2.columns.tolist()
            col1 = st.sidebar.selectbox("choose the variable", all_columns)
            col2 = st.sidebar.selectbox('choose the variable ', all_columns)
            option = ['none', 'Add', 'Sub', 'Mul', 'Div']
            arithmetic = st.sidebar.selectbox("select the arithmetic operator", option)

            if arithmetic == "Add":

                st.subheader('Add')
                data_frame_2['add'] = data_frame_2[col1] + data_frame_2[col2]

                st.write(data_frame_2)


            elif arithmetic == "Sub":
                st.header("subtract")

                data_frame_2['sub'] = data_frame_2[col1] - data_frame_2[col2]

                st.write(data_frame_2)

            elif arithmetic == "Mul":
                st.header("Multiply")

                data_frame_2['Mul'] = data_frame_2[col1] * data_frame_2[col2]

                st.write(data_frame_2)

            elif arithmetic == "Div":
                st.header("Divide")

                data_frame_2['Div'] = data_frame_2[col1] / data_frame_2[col2]
                st.write(data_frame_2)
        elif oper == "compare":
            st.subheader("comparison operations for dataset 2")
            all_columns = data_frame_2.columns.tolist()
            col3 = st.sidebar.selectbox("choose the variable for A", all_columns)
            col4 = st.sidebar.selectbox("choose the variable for B", all_columns)
            conditions = [data_frame_2[col3] > data_frame_2[col4],
                          data_frame_2[col3] < data_frame_2[col4],
                          data_frame_2[col3] == data_frame_2[col4]]

            # define choices
            choices = ['TRUE', 'FALSE', 'EQUAL']

            # create new column in DataFrame that displays results of comparisons
            data_frame_2['comparison'] = np.select(conditions, choices)
            data_frame_2['Diff'] = np.where(data_frame_2[col3] == data_frame_2[col4], 0, data_frame_2[col3] - data_frame_2[col4])
            st.write(data_frame_2)

    # join two datasets
    i=st.checkbox ("join two dataset")
    df_index = pd.merge(data_frame, data_frame_2, right_index=True, left_index=True)
    if i:
        st.subheader("JOIN TWO DATASETS")
        st.write(df_index)
        st.subheader("machine learning")
        st.write("K-Means")
        all_columns12 = df_index.columns.tolist()
        Data = {'x': df_index[st.selectbox("Select X             ", all_columns12)],
            'y': df_index[st.selectbox("Select Y             ", all_columns12)]
            }
        rf = DataFrame(Data, columns=['x', 'y'])
        st.write(rf)
        cluster = st.selectbox("values", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        kmeans = KMeans(n_clusters=cluster).fit(rf)
        centroids = kmeans.cluster_centers_
        st.write(centroids)
        plt.scatter(rf['x'], rf['y'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
        st.pyplot()
