import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Revenue Status Prediction Streamlit App
# Developed by Dollen Kyotungire
# --------------------------------------------------
# Expected model file:
# final_revenue_status_model.pkl
#
# Expected input columns:
# Client, Manager, Ad section, salesRepName, salesRepCode,
# Product, Payment method, Section Category, Col/Bw,
# Day_of_Week, Month, Quarter
# --------------------------------------------------

st.set_page_config(
    page_title="Revenue Status Prediction App",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load saved model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("final_revenue_status_model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error("Model file not found or could not be loaded.")
    st.write("Actual error:")
    st.exception(e)
    st.info("Make sure `final_revenue_status_model.pkl` is in the same folder as this `app.py` file.")
    st.stop()


# -----------------------------
# App title and description
# -----------------------------
st.title("📊 Revenue Status Prediction App")

st.caption("Developed by Dollen Kyotungire")

st.write(
    """
    This app predicts whether an advertising booking is likely to generate revenue 
    or is at risk of **No Revenue**.
    """
)

st.markdown("---")


# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1.2, 1])


# -----------------------------
# Input section
# -----------------------------
with left:
    st.subheader("Enter Booking Details")

    client = st.text_input("Client", "NEW CLIENT")

    manager = st.text_input("Manager", "Ad Centre")

    ad_section = st.text_input("Ad Section", "ROP")

    sales_rep_name = st.text_input("Sales Representative Name", "Jane Mwangi")

    sales_rep_code = st.text_input("Sales Representative Code", "JM001")

    product = st.selectbox(
        "Product",
        ["DN", "BD", "TF", "EA", "Other"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        ["BL", "CG", "MGS", "SC", "MG", "CO", "EC", "BS", "CA", "CC", "CQ", "MP", "RH", "Other"]
    )

    section_category = st.text_input("Section Category", "Classified")

    col_bw = st.selectbox(
        "Colour / Black & White",
        ["BW", "COL", "Other"]
    )

    day_of_week = st.selectbox(
        "Day of Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )

    month = st.selectbox(
        "Month",
        list(range(1, 13))
    )

    quarter = st.selectbox(
        "Quarter",
        [1, 2, 3, 4]
    )


# -----------------------------
# Create input dataframe
# -----------------------------
input_data = pd.DataFrame([{
    "Client": client,
    "Manager": manager,
    "Ad section": ad_section,
    "salesRepName": sales_rep_name,
    "salesRepCode": sales_rep_code,
    "Product": product,
    "Payment method": payment_method,
    "Section Category": section_category,
    "Col/Bw": col_bw,
    "Day_of_Week": day_of_week,
    "Month": month,
    "Quarter": quarter
}])


# -----------------------------
# Prediction section
# -----------------------------
with right:
    st.subheader("Prediction Panel")

    st.write("Click the button below to predict the revenue status of the booking record.")

    if st.button("Predict Revenue Status", use_container_width=True):

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        prob_has_revenue = probability[0]
        prob_no_revenue = probability[1]

        if prediction == 1:
            st.error("Prediction: No Revenue Risk")

            st.metric(
                label="Probability of No Revenue",
                value=f"{prob_no_revenue:.2%}"
            )

            st.metric(
                label="Probability of Has Revenue",
                value=f"{prob_has_revenue:.2%}"
            )

            st.warning(
                "Recommendation: This booking should be prioritised for early follow-up "
                "by the sales or finance team."
            )

        else:
            st.success("Prediction: Likely Has Revenue")

            st.metric(
                label="Probability of Has Revenue",
                value=f"{prob_has_revenue:.2%}"
            )

            st.metric(
                label="Probability of No Revenue",
                value=f"{prob_no_revenue:.2%}"
            )

            st.info(
                "Recommendation: This booking appears less risky, but normal follow-up should continue."
            )


# -----------------------------
# Display input record
# -----------------------------
st.markdown("---")

st.subheader("Input Record Used for Prediction")

st.dataframe(input_data, use_container_width=True)


# -----------------------------
# About section
# -----------------------------
st.markdown("---")

with st.expander("About this demo"):
    st.write(
        """
        This demo loads a saved machine learning pipeline and applies the same preprocessing 
        used during training.

        The model predicts whether a booking record is likely to generate revenue or is at risk 
        of No Revenue using information such as client, product, sales representative, payment 
        method, section category and timing.

        The output should be used as a decision-support signal and not as a replacement for 
        human judgement.
        """
    )


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.markdown(
    """
    <div style='text-align: center; font-size: 14px; color: gray;'>
        Developed by <b>Dollen Kyotungire</b>
    </div>
    """,
    unsafe_allow_html=True
)
