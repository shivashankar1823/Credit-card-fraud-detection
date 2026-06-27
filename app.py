import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

model = joblib.load("credit_card_fraud.pkl")

st.title("💳 Credit Card Fraud Detection")
st.write("Predict whether a credit card transaction is Fraud or Legitimate.")

@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")

df = load_data()

feature_names = df.drop(["Class","Time",'V2','V16','V18','V21'], axis=1).columns
st.sidebar.header("Choose Input Method")

option = st.sidebar.radio(
    "Input Method",
    ["Manual Input", "Upload CSV"]
)

if option == "Manual Input":

    st.subheader("Enter Transaction Details")

    input_data = {}

    cols = st.columns(3)

    for i, feature in enumerate(feature_names):

        with cols[i % 3]:

            default = float(df[feature].median())

            input_data[feature] = st.number_input(
                feature,
                value=default,
                format="%.6f"
            )

    if st.button("Predict"):

        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0]

        st.markdown("---")

        if prediction == 1:
            st.error("⚠ Fraudulent Transaction Detected!")
        else:
            st.success("✅ Legitimate Transaction")

        st.subheader("Prediction Probability")

        st.write(f"Legitimate : **{probability[0]*100:.2f}%**")
        st.write(f"Fraud : **{probability[1]*100:.2f}%**")
else:

    st.subheader("Upload CSV File")

    uploaded_file = st.file_uploader(
        "Upload transaction data",
        type=["csv"]
    )

    if uploaded_file is not None:

        test = pd.read_csv(uploaded_file)

        test = test.drop(['Time'],axis=1,inplace=True)

        st.write("Uploaded Data")

        st.dataframe(test.head())

        prediction = model.predict(test)

        probability = model.predict_proba(test)

        test["Prediction"] = prediction

        test["Result"] = test["Prediction"].map({
            0: "Legitimate",
            1: "Fraud"
        })

        test["Fraud Probability"] = probability[:,1]

        st.subheader("Prediction Results")

        st.dataframe(test)

        csv = test.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Results",
            csv,
            "fraud_predictions.csv",
            "text/csv"
        )