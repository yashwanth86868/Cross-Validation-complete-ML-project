
import streamlit as st
import requests
import os


# Page configuration
st.set_page_config(
    page_title="Customer Purchase Prediction",
    page_icon="🤖",
    layout="centered"
)

# Application title
st.title("🤖 Customer Purchase Prediction")

st.write(
    "Predict whether a customer will purchase a product "
    "using a Machine Learning model."
)

st.divider()

# Customer inputs
st.subheader("Enter Customer Details")

age = st.number_input(
    "Customer Age",
    min_value=1,
    max_value=120,
    value=35
)

salary = st.number_input(
    "Estimated Salary",
    min_value=0,
    max_value=100000000,
    value=60000
)

# Prediction button
if st.button("Predict", type="primary"):
    
    import os



API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)

url = f"{API_URL}/predict"

data = {
        "age": age,
        "estimated_salary": salary
    }

try:

        response = requests.post(
            url,
            json=data,
            timeout=10
        )

        response.raise_for_status()

        result = response.json()

        st.divider()

        st.subheader("Prediction Result")

        if result["prediction"] == 1:

            st.success(
                "🎉 Customer may purchase the product!"
            )

        else:

            st.warning(
                "Customer may not purchase the product."
            )

        st.write("Prediction:", result["prediction"])

        st.write("Age:", result["age"])

        st.write("Estimated Salary:", result["estimated_salary"])

except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to FastAPI. "
            "Please start the backend server."
        )

except requests.exceptions.RequestException as error:

        st.error(f"API request failed: {error}")

st.divider()

st.caption("Powered by Python, FastAPI, Streamlit and Kernel SVM")