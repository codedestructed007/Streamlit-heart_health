# -*- coding: utf-8 -*-
import time
# import pandas_profiling
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from streamlit_option_menu import option_menu
from streamlit_pandas_profiling import st_profile_report
from pydantic_settings import BaseSettings

# Importing ends here --------------------------------------

# -----------------------------------------page layout----------------------------


st.set_page_config(
    page_title="Heart Health Predictor",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

with st.sidebar:
    selected_option = option_menu("Main Menu", ["Home", "Visualizaiton", "Report"],
                                  icons=["house", "book", "file-text"], menu_icon="cast", default_index=0)


# page layout ends------------------------------------------------------------------


# backend starts
df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv('data/test.csv')
df = pd.concat([df_train, df_test], ignore_index=True)
removed_duplicates_df = df.drop_duplicates()
# backend ends


#!Visualization starts from here


def corr_heatmap(palette):
    corr_mat = df.corr().stack().reset_index(name="correlation")
    g = sns.relplot(
        data=corr_mat, x='level_0', y='level_1', hue='correlation', sizes=(200, 400), palette=palette, size='correlation',
        size_norm=(0.4, 0.7)
    )
    # remove tick lines
    plt.tick_params(left=False, bottom=False)
    # remove the spines
    g.despine(left=True, bottom=True)
    plt.xticks(rotation=90)
    plt.xlabel("Columns (X-axis)")
    plt.ylabel("Columns (Y-axis)")
    plt.title('Correlation among all the columns')
    return g


def histplot(x, hue, palette):
    fig, ax = plt.subplots()
    hist = sns.histplot(data=df, x=x, hue=hue, palette=palette, legend='brief')
    legend = hist.get_legend()
    legend.set_bbox_to_anchor((1.05, 1))
    plt.xlabel(x)

    plt.title('Distribution of {} in sample dataset'.format(x))
    return fig


def boxplot(y, x, hue, notch,):
    fig, axes = plt.subplots()

    sns.boxplot(data=df, y=y, x=x, ax=axes, hue=hue, notch=notch)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('Distribution of Health Attributes in Heart Disease')
    return fig


def scatterplot(x, y, hue, style, palette):
    fig, axes = plt.subplots()
    sns.scatterplot(data=df, x=x, y=y, hue=hue, style=style, palette=palette)
    plt.ylabel(y)
    plt.xlabel(x)

    # Get the current figure object
    fig = plt.gcf()
    plt.title('Correlation between {x} and {y}')
    return fig


def catplot(x, y, col, rows, kind, palette, hue):
    plt.figure(figsize=(12, 8))
    sns.set_theme(style='darkgrid')
    cat = sns.catplot(
        data=df, x=x, y=y, col=col, row=rows, kind=kind,
        palette=palette, hue=hue, legend_out=False

    )

    cat.set_xticklabels(df[x].value_counts().index,)
    cat.set_axis_labels(x, y)
    plt.tight_layout()
    plt.xlabel(x)
    plt.ylabel(y)
    # attaching link as a reference
    text = "Learn more about interpreting this plot: [Click here](https://seaborn.pydata.org/generated/seaborn.catplot.html)"
    return text, cat


if selected_option == 'Home':
    total_length = len(removed_duplicates_df)
    header_style = """
        <style>
        .header-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            background-color: black;
            border-bottom: 1px solid #ccc;
        }

        .title {
            font-size: 24px;
            font-weight: bold;
            font-family: "Helvetica Neue", sans-serif;
            color: white;
        }
        </style>
"""
    st.markdown(header_style, unsafe_allow_html=True)
    # title container
    st.markdown(
        """
<div class = "header-continer">
    <div class = "title"> \U0001F493 Let's Predict Your Heart Health</div>
    </div>
""",
        unsafe_allow_html=True
    )
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            <div style='background-color: black; padding: 20px; border-radius: 10px;'>
                <h4>About the Model &#128641; </h4>
                <p>
                    This model predicts whether a patient has heart disease or not.
                    The model is built using the Logistic Regression algorithm.
                    It has shown excellent performance with high accuracy scores on both the training and test datasets.
                </p>
                <h5>Model Performance &#127919;</h5>
                <ul>
                    <li>Accuracy score on training dataset: 0.8761061946902655</li>
                    <li>Accuracy score on test dataset: 0.8157894736842105</li>
                </ul>
                
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
 <div style='background-color: black; padding: 20px; border-radius: 10px;'>
                <h4>How Model works &#10067 </h4>
                <p>
                    To make a prediction, please fill in the required fields on the right and click on the <b>Predict</b> button.                
                </p>           
            </div>
            """,
            unsafe_allow_html=True
        )

        # Create a box to display the content
        st.markdown(
            f"""
            <div style='background-color: black; padding: 20px; border-radius: 10px;'>
                <h4>About the Dataset &#128202;</h4>
                <p>
                    This model, built using the Decision Tree classifier algorithm, predicts whether a patient has heart disease.
                    It has demonstrated high accuracy scores on both the training and test datasets.
                    The dataset used for training and evaluation was separated into two files: train.csv and test.csv.
                    total observations on which our model is trained is {total_length}
                </p>
                <h5>Key Columns:</h5>
                <ul>
                    <li><b>sex:</b> Gender of the patient (0: female, 1: male)</li>
                    <li><b>cp:</b> Type of chest pain experienced by the patient</li>
                    <li><b>trestbps:</b> Resting blood pressure in mm Hg</li>
                    <li><b>chol:</b> Serum cholesterol level in mg/dl</li>
                    <li><b>fbs:</b> Fasting blood sugar level (1: > 120 mg/dl, 0: <= 120 mg/dl)</li>
                    <li><b>restecg:</b> Resting electrocardiographic results</li>
                    <li><b>thalach:</b> Maximum heart rate achieved</li>
                    <li><b>exang:</b> Exercise-induced angina (1: yes, 0: no)</li>
                    <li><b>oldpeak:</b> ST depression induced by exercise relative to rest</li>
                    <li><b>slope:</b> Slope of the peak exercise ST segment</li>
                    <li><b>ca:</b> Number of major vessels colored by fluoroscopy</li>
                    <li><b>thal:</b> Thalassemia type</li>
                    <li><b>target:</b> Presence of heart disease (1: disease present, 0: disease not present)</li>
                </ul>
                
            </div>
            """,
            unsafe_allow_html=True
        )

        with right:
            input_data = {
                'age': st.number_input('**Enter your age**', min_value=13, max_value=95),
                'sex': bool(st.radio('**Your Gender**', options=['Male', 'Female'])),
                'cp': st.radio('**Constrictive pericarditis (CP)**', options=[0, 1, 2, 3]),
                'trestbps': st.slider('**trestbps**', min_value=50, max_value=400),
                'chol': st.slider('**cholesterol**', min_value=100, max_value=600),
                'fbs': bool(st.radio('**fbs**', options=[True, False])),
                'restecg': st.radio('**resting electrocardiographic (Restecg)**', options=[True, False]),
                'thalach': st.slider('**Maximum Heart Rate Achieved (THALACH)**', min_value=50, max_value=300),
                'exang': bool(st.radio('**resting electrocardiographic**', options=[True, False])),
                'oldpeak': st.number_input('**ST depression (oldpeak)**', help='Enter the value between 0 to 20'),
                'slope': st.radio('**ST/heart rate slope(slope)**', options=[0, 1, 2]),
                'ca': st.radio('**Coronary angiography (CA)**', options=[0, 1, 2, 3, 4]),
                'thal': st.radio('**thalassemia (thal)**', options=[0, 1, 2, 3])
            }

            input_df = pd.DataFrame.from_dict(input_data, orient='index').T

            # prediciton function
            prediction_button = st.button(
                'Predict', help='Click to perform prediction')

            # Check if the prediction button is clicked
            if prediction_button:
                # Start the timer
                start_time = time.time()

                # Display the "Performing prediction..." message
                loading_message = st.empty()
                loading_message.text('Performing prediction...')

                # Simulate the algorithm series
                X_train, X_test, y_train, y_test = train_test_split(
                    removed_duplicates_df.iloc[:, 0:13], removed_duplicates_df.loc[:, ['target']], random_state=42)
                lr = LogisticRegression().fit(X_train, y_train)
                prediction = lr.predict(input_df)
                # Replace this with your actual algorithm logic
                time.sleep(3)  # Simulating a 3-second algorithm series

                # Stop the timer and calculate the elapsed time
                elapsed_time = time.time() - start_time

                # Display the prediction completion message
                loading_message.text(
                    'Prediction completed in {:.2f} seconds!'.format(elapsed_time))

                # converting target value(1/0) into yes or No
                if prediction == 1:
                    st.success(
                        "We regret to inform you that the model predicts a presence of heart disease.")
                if prediction == 0:
                    st.success(
                        "There is no indication of heart disease. However, we recommend consulting with your doctor for further confirmation.")


if selected_option == 'Visualizaiton':
    header_style = """
        <style>
        .header-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            background-color: black;
            border-bottom: 1px solid #ccc;
        }

        .title {
            font-size: 24px;
            font-weight: bold;
            font-family: "Helvetica Neue", sans-serif;
            color: white;
        }
        .visualization-logo {
            width: 40px;
            height: auto;
        }
        </style>
        """
    st.markdown(header_style, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="header-container">
            <div class="title">Visualization\U0001F4C8..</div>
        
        </div>
            """,
        unsafe_allow_html=True
    )

    def continuous_columns(df):
        num_col = []
        for i in df.columns:
            if len(df[i].value_counts().index) > 10:
                num_col.append(i)
        return num_col

    def categorical_columns(df):
        cat_columns = []
        for i in df.columns:
            if len(df[i].value_counts().index) < 10:
                cat_columns.append(i)
        return cat_columns

    options = ['Correlation Heatmap', 'Histogram Plot',
               'Boxplot', 'Scatter Plot', 'Catplot']

    selected_graph = st.sidebar.selectbox('Select a Visualization', options,)
    if selected_graph == 'Correlation Heatmap':
        selected_palette = st.sidebar.selectbox(
            'Color Palette', ['YlGnBu', 'BuGn', 'PuBu', 'Blues', 'Greens'])
        st.pyplot(corr_heatmap(selected_palette))
        text = "Learn more about interpreting this plot: [Click here](https://seaborn.pydata.org/examples/heat_scatter.html)"
        st.markdown(text, unsafe_allow_html=True)
    if selected_graph == options[1]:
        axis = st.sidebar.selectbox('Select x axis', continuous_columns(df))
        hue = st.sidebar.selectbox('hue', options=categorical_columns(df))
        palette_colour = st.sidebar.selectbox(
            'Palette color', ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind'])
        st.pyplot(histplot(axis, hue, palette_colour))
        text = "Learn more about interpreting this plot: [Click here](https://seaborn.pydata.org/generated/seaborn.histplot.html)"
        st.markdown(text, unsafe_allow_html=True)

    if selected_graph == options[2]:
        yaxis = st.sidebar.selectbox('Y axis', continuous_columns(df))
        xaxis = st.sidebar.selectbox('X axis', categorical_columns(df))
        hue = st.sidebar.selectbox('Hue', categorical_columns(df))
        notch = st.sidebar.radio('Notch', [True, False])
        st.pyplot(boxplot(yaxis, xaxis, hue, notch))
        text = "Learn more about interpreting this plot: [Click here](https://seaborn.pydata.org/generated/seaborn.boxplot.html)"
        st.markdown(text, unsafe_allow_html=True)
    if selected_graph == options[3]:
        xaxis = st.sidebar.selectbox('X axis', continuous_columns(df))
        yaxis = st.sidebar.selectbox('Y axis', continuous_columns(df))
        hue = st.sidebar.selectbox('hue', categorical_columns(df))
        style = st.sidebar.selectbox('style', categorical_columns(df))
        palette_colour = st.sidebar.selectbox(
            'Palette color', ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind'])
        st.pyplot(scatterplot(xaxis, yaxis, hue, style, palette_colour))
        text = "Learn more about interpreting this plot: [Click here](https://seaborn.pydata.org/generated/seaborn.scatterplot.html)"
        st.markdown(text, unsafe_allow_html=True)

    if selected_graph == options[4]:
        xaxis = st.sidebar.selectbox('X-axis', categorical_columns(df))
        yaxis = st.sidebar.selectbox('Y-axis', continuous_columns(df))
        columns = st.sidebar.selectbox(
            'Columns', [None] + categorical_columns(df))
        rows = st.sidebar.selectbox('Rows', [
                                    None] + categorical_columns(df), help='We recommend you to keep the graph simple')
        kind = st.sidebar.selectbox(
            'kind', ['box', 'violin', 'boxen', 'point', 'swarm', 'bar', 'strip'])
        palette_colour = st.sidebar.selectbox(
            'Palette color', ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind'])
        hue = st.sidebar.selectbox('hue', categorical_columns(df))
        text, graph = catplot(xaxis, yaxis, columns, rows,
                              kind, palette_colour, hue)
        st.pyplot(graph)
        st.markdown(text, unsafe_allow_html=True)
if selected_option == 'Report':
    header_style = """
        <style>
        .header-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            background-color: black;
            border-bottom: 1px solid #ccc;
        }

        .title {
            font-size: 24px;
            font-weight: bold;
            font-family: "Helvetica Neue", sans-serif;
            color: white;
        }
        .visualization-logo {
            width: 40px;
            height: auto;
        }
        </style>
        """
    st.markdown(header_style, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="header-container">
            <div class="title">Report\U0001F4CB..</div>
        
        </div>
            """,
        unsafe_allow_html=True
    )

    pr = df.profile_report()
    st_profile_report(pr)
