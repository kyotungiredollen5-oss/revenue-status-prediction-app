import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Revenue Status Prediction Streamlit App
# Developed by Dollen Kyotungire
# --------------------------------------------------

st.set_page_config(
    page_title="Revenue Status Prediction App",
    page_icon="📊",
    layout="wide"
)

st.title("Revenue Status Prediction App")
st.caption("Developed by Dollen Kyotungire")

st.write(
    """
    This app trains a machine learning model using the advertising revenue dataset
    stored inside the project folder. It predicts whether a booking record is likely
    to result in Has Revenue or No Revenue Risk.
    """
)

st.markdown("---")


# --------------------------------------------------
# Load dataset from repository
# --------------------------------------------------

@st.cache_data
def load_data():
    file_path = "Data.xlsx"
    xls = pd.ExcelFile(file_path)
    sheet_name = xls.sheet_names[0]
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return data, sheet_name


try:
    df, sheet_used = load_data()
except Exception as e:
    st.error("Dataset could not be loaded.")
    st.write("Make sure the file is named exactly `Data.xlsx` and is in the same folder as `app.py`.")
    st.write("Actual error:")
    st.exception(e)
    st.stop()


st.success(f"Dataset loaded successfully from sheet: {sheet_used}")
st.write("Dataset shape:", df.shape)

with st.expander("Preview dataset"):
    st.dataframe(df.head(), use_container_width=True)


# --------------------------------------------------
# Data cleaning and preparation
# --------------------------------------------------

st.subheader("1. Data Cleaning and Preparation")

df.columns = df.columns.astype(str).str.strip()

# Remove unnecessary unnamed columns
unnamed_cols = [col for col in df.columns if col.lower().startswith("unnamed")]
if unnamed_cols:
    df = df.drop(columns=unnamed_cols)

if "Revenue" not in df.columns:
    st.error("The dataset must contain a column called `Revenue`.")
    st.stop()

df["Revenue_Clean"] = pd.to_numeric(df["Revenue"], errors="coerce").fillna(0)

df["Revenue_Status"] = df["Revenue_Clean"].apply(
    lambda x: "Has Revenue" if x > 0 else "No Revenue"
)

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Day_of_Week"] = df["Date"].dt.day_name()
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
else:
    df["Day_of_Week"] = "Unknown"
    df["Month"] = np.nan
    df["Quarter"] = np.nan

target_distribution = df["Revenue_Status"].value_counts().reset_index()
target_distribution.columns = ["Revenue Status", "Count"]
target_distribution["Percentage"] = target_distribution["Count"] / len(df) * 100

col1, col2 = st.columns(2)

with col1:
    st.write("Target distribution")
    st.dataframe(target_distribution, use_container_width=True)

with col2:
    st.write("Columns used after cleaning")
    st.write(df.columns.tolist())


# --------------------------------------------------
# Feature selection
# --------------------------------------------------

st.subheader("2. Feature Selection")

safe_features = [
    "Client",
    "Manager",
    "Ad section",
    "salesRepName",
    "salesRepCode",
    "Product",
    "Payment method",
    "Section Category",
    "Col/Bw",
    "Day_of_Week",
    "Month",
    "Quarter"
]

available_features = [col for col in safe_features if col in df.columns]
missing_features = [col for col in safe_features if col not in df.columns]

if not available_features:
    st.error("None of the required modelling features were found in the dataset.")
    st.stop()

st.write("Features used in the model:")
st.write(available_features)

if missing_features:
    st.write("Features missing from the dataset and excluded:")
    st.write(missing_features)


X = df[available_features]

y = df["Revenue_Status"].map({
    "Has Revenue": 0,
    "No Revenue": 1
})

valid_rows = y.notna()
X = X.loc[valid_rows]
y = y.loc[valid_rows]


# --------------------------------------------------
# Model training
# --------------------------------------------------

st.subheader("3. Model Training")

if y.nunique() < 2:
    st.error("The target variable has only one class. The model requires both Has Revenue and No Revenue records.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", GradientBoostingClassifier(random_state=42))
])

with st.spinner("Training model..."):
    model.fit(X_train, y_train)

st.success("Model trained successfully.")


# --------------------------------------------------
# Model evaluation
# --------------------------------------------------

st.subheader("4. Model Evaluation")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision_no_revenue = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
recall_no_revenue = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
f1_no_revenue = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

m1, m2, m3 = st.columns(3)
m4, m5, m6 = st.columns(3)

m1.metric("Accuracy", f"{accuracy:.3f}")
m2.metric("Precision - No Revenue", f"{precision_no_revenue:.3f}")
m3.metric("Recall - No Revenue", f"{recall_no_revenue:.3f}")
m4.metric("F1 - No Revenue", f"{f1_no_revenue:.3f}")
m5.metric("ROC-AUC", f"{roc_auc:.3f}")
m6.metric("PR-AUC", f"{pr_auc:.3f}")

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Has Revenue", "Actual No Revenue"],
    columns=["Predicted Has Revenue", "Predicted No Revenue"]
)

st.write("Confusion Matrix")
st.dataframe(cm_df, use_container_width=True)

with st.expander("Classification report"):
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Has Revenue", "No Revenue"],
        zero_division=0
    )
    st.text(report)


# --------------------------------------------------
# Prediction interface
# --------------------------------------------------

st.subheader("5. Prediction Interface")

st.write(
    """
    Enter booking details below. The model will predict whether the record is likely
    to result in revenue or no revenue.
    """
)

left, right = st.columns([1.2, 1])

with left:
    client = st.text_input("Client", "NEW CLIENT")

    manager_options = sorted(df["Manager"].dropna().astype(str).unique().tolist()) if "Manager" in df.columns else ["Ad Centre"]
    manager = st.selectbox("Manager", manager_options)

    ad_section_options = sorted(df["Ad section"].dropna().astype(str).unique().tolist()) if "Ad section" in df.columns else ["ROP"]
    ad_section = st.selectbox("Ad Section", ad_section_options)

    sales_rep_options = sorted(df["salesRepName"].dropna().astype(str).unique().tolist()) if "salesRepName" in df.columns else ["Jane Mwangi"]
    sales_rep_name = st.selectbox("Sales Representative Name", sales_rep_options)

    sales_rep_code_options = sorted(df["salesRepCode"].dropna().astype(str).unique().tolist()) if "salesRepCode" in df.columns else ["JM001"]
    sales_rep_code = st.selectbox("Sales Representative Code", sales_rep_code_options)

    product_options = sorted(df["Product"].dropna().astype(str).unique().tolist()) if "Product" in df.columns else ["DN", "BD", "TF", "EA"]
    product = st.selectbox("Product", product_options)

    payment_options = sorted(df["Payment method"].dropna().astype(str).unique().tolist()) if "Payment method" in df.columns else ["BL"]
    payment_method = st.selectbox("Payment Method", payment_options)

    section_options = sorted(df["Section Category"].dropna().astype(str).unique().tolist()) if "Section Category" in df.columns else ["Classified"]
    section_category = st.selectbox("Section Category", section_options)

    col_bw_options = sorted(df["Col/Bw"].dropna().astype(str).unique().tolist()) if "Col/Bw" in df.columns else ["BW"]
    col_bw = st.selectbox("Colour / Black and White", col_bw_options)

    day_of_week = st.selectbox(
        "Day of Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )

    month = st.selectbox("Month", list(range(1, 13)))
    quarter = st.selectbox("Quarter", [1, 2, 3, 4])


input_record = pd.DataFrame([{
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

input_record = input_record[available_features]

with right:
    st.write("Prediction result")

    if st.button("Predict Revenue Status", use_container_width=True):
        prediction = model.predict(input_record)[0]
        probability = model.predict_proba(input_record)[0]

        prob_has_revenue = probability[0]
        prob_no_revenue = probability[1]

        if prediction == 1:
            st.error("Prediction: No Revenue Risk")
            st.metric("Probability of No Revenue", f"{prob_no_revenue:.2%}")
            st.metric("Probability of Has Revenue", f"{prob_has_revenue:.2%}")
            st.warning("Recommendation: Prioritise this booking for early follow-up.")
        else:
            st.success("Prediction: Likely Has Revenue")
            st.metric("Probability of Has Revenue", f"{prob_has_revenue:.2%}")
            st.metric("Probability of No Revenue", f"{prob_no_revenue:.2%}")
            st.info("Recommendation: Continue normal follow-up.")


st.markdown("---")

st.subheader("Input Record Used")
st.dataframe(input_record, use_container_width=True)

st.markdown("---")

with st.expander("About this app"):
    st.write(
        """
        This app trains the model directly from the dataset stored in the project folder.
        This avoids saved-model version problems between Google Colab and Streamlit Cloud.

        The model uses Gradient Boosting Classifier and focuses on identifying records
        that may result in No Revenue.

        The output should be used as a decision-support signal, not as a replacement for human judgement.
        """
    )

st.markdown(
    """
    <div style='text-align: center; font-size: 14px; color: gray;'>
        Developed by <b>Dollen Kyotungire</b>
    </div>
    """,
    unsafe_allow_html=True
)
