import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Load and prepare data
model_df = pd.read_csv("dataset.csv")

le_gate = LabelEncoder()
model_df["GATE_Score"] = le_gate.fit_transform(model_df["GATE_Score"])

le_master = LabelEncoder()
model_df["Should_Do_Masters"] = le_master.fit_transform(model_df["Should_Do_Masters"])

if le_master.transform(['Yes'])[0] != 1:
    model_df["Should_Do_Masters"] = 1 - model_df["Should_Do_Masters"]
    le_master = LabelEncoder()
    le_master.fit(["No", "Yes"])

gate_label_map = dict(zip(le_gate.transform(le_gate.classes_), le_gate.classes_))
master_label_map = {0: "No", 1: "Yes"}

X = model_df[["GATE_Score", "Salary"]]
y = model_df["Should_Do_Masters"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(x_train, y_train)

# UI
st.title("🎓 Masters Decision Support System")

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "About Model", "Dataset", "Prediction Model", "Graph"],
        icons=["house", "info-circle", "table", "cpu", "bar-chart"],
        orientation="vertical"
    )

# Pages
if selected == "Home":
    st.subheader("👋 Welcome!")
    st.write("Use the sidebar to explore the application.")

elif selected == "About Model":
    st.title("📘 About the Model")
    st.markdown("""
    ### 🎯 Goal
    Predict whether a student should pursue a Master's degree based on:
    - GATE Score
    - Salary

    ### 🧠 Model
    - Random Forest Classifier
    - Trained on balanced_master_decision.csv


    """)

elif selected == "Dataset":
    st.title("📁 Uploaded Dataset (master_decison.csv)")
    try:
        dataset_df = pd.read_csv("dataset.csv")
        st.dataframe(dataset_df)
    except FileNotFoundError:
        st.error("data.csv file not found. Please upload or check path.")

elif selected == "Prediction Model":
    st.title("🤖 Predict: Should You Do a Master's?")
    gate_input = st.selectbox("Select GATE Score", le_gate.classes_)
    salary_input = st.number_input("Enter current salary (INR)", value=500000)

    if st.button("Predict"):
        gate_encoded = le_gate.transform([gate_input])[0]
        prediction = model.predict([[gate_encoded, salary_input]])
        result = le_master.inverse_transform(prediction)[0]
        st.success(f"Prediction: You should {'🧑‍🎓 do' if result == 'Yes' else '🚫 not do'} a Master's.")

elif selected == "Graph":
    st.title("📊 Visualize Data from master_decison.csv")
    feature = st.selectbox("Choose a feature", ["GATE_Score", "Salary", "Should_Do_Masters"])

    if feature == "GATE_Score":
        st.subheader("GATE Score Distribution")
        counts = model_df[feature].value_counts().sort_index()
        labels = [gate_label_map[i] for i in counts.index]
        plt.figure()
        plt.bar(labels, counts.values, color="purple")
        plt.xlabel("GATE Score")
        plt.ylabel("Count")
        st.pyplot(plt.gcf())

    elif feature == "Salary":
        st.subheader("Salary Histogram")
        plt.figure()
        plt.hist(model_df[feature], bins=15, color="orange", edgecolor="black")
        plt.xlabel("Salary")
        plt.ylabel("Frequency")
        st.pyplot(plt.gcf())

    elif feature == "Should_Do_Masters":
        st.subheader("Master's Decision Distribution")
        counts = model_df[feature].value_counts().sort_index()
        labels = [master_label_map[i] for i in counts.index]
        plt.figure()
        plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=["#ff9999", "#66b3ff"])
        st.pyplot(plt.gcf())
