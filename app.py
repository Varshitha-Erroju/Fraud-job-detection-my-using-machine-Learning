import os
import re
import string
import joblib
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Fake Job Posting Detection",
    page_icon="🛡️",
    layout="wide"
)

DATASET_PATH = "fake_job_postings.csv"
MODEL_PATH = "fraud_job_model.pkl"

TEXT_COLUMNS = [
    "title", "company_profile", "description",
    "requirements", "benefits"
]

CATEGORICAL_COLUMNS = [
    "location", "department", "salary_range", "employment_type",
    "required_experience", "required_education", "industry", "function"
]

NUMERIC_COLUMNS = [
    "telecommuting", "has_company_logo", "has_questions"
]

REQUIRED_COLUMNS = (
    ["job_id"] + TEXT_COLUMNS + CATEGORICAL_COLUMNS +
    NUMERIC_COLUMNS + ["fraudulent"]
)


def clean_text(text):
    """Basic NLP text cleaning."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_dataframe(df):
    """Create combined text and retain model features."""
    df = df.copy()

    for column in TEXT_COLUMNS + CATEGORICAL_COLUMNS:
        df[column] = df[column].fillna("").astype(str)

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    df["combined_text"] = (
        df["title"] + " " +
        df["company_profile"] + " " +
        df["description"] + " " +
        df["requirements"] + " " +
        df["benefits"]
    ).apply(clean_text)

    return df


def build_pipeline():
    text_transformer = TfidfVectorizer(
        max_features=15000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2
                )
            )
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_transformer, "combined_text"),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
            ("num", numeric_transformer, NUMERIC_COLUMNS)
        ]
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )


@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH), None

    if not os.path.exists(DATASET_PATH):
        return None, None

    df = pd.read_csv(DATASET_PATH)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    df = prepare_dataframe(df)
    X = df[["combined_text"] + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS]
    y = df["fraudulent"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    joblib.dump(pipeline, MODEL_PATH)
    return pipeline, report


st.title("🛡️ Detecting Fraud Jobs Using NLP")
st.write(
    "Enter job posting details to predict whether the job is "
    "**Legitimate** or **Fraudulent**."
)

try:
    model, report = load_or_train_model()
except Exception as error:
    st.error(f"Model initialization error: {error}")
    st.stop()

if model is None:
    st.error(
        f"Dataset not found. Place `{DATASET_PATH}` in the same folder "
        "as `app.py`, then restart the app."
    )
    st.stop()

with st.sidebar:
    st.header("Project Information")
    st.write("**Project:** Detecting Fraud Jobs Using NLP")
    st.write("**Model:** TF-IDF + Logistic Regression")
    st.write("**Framework:** Streamlit")

    if report:
        st.subheader("Test Performance")
        st.metric("Accuracy", f"{report['accuracy'] * 100:.2f}%")
        if "1" in report:
            st.metric(
                "Fraud Recall",
                f"{report['1']['recall'] * 100:.2f}%"
            )
            st.metric(
                "Fraud F1-score",
                f"{report['1']['f1-score'] * 100:.2f}%"
            )

with st.form("job_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        title = st.text_input("Job Title")
        location = st.text_input("Location")
        department = st.text_input("Department")
        salary_range = st.text_input("Salary Range")
        employment_type = st.selectbox(
            "Employment Type",
            ["", "Full-time", "Part-time", "Contract", "Temporary", "Other"]
        )
        required_experience = st.text_input("Required Experience")
        required_education = st.text_input("Required Education")
        industry = st.text_input("Industry")
        function = st.text_input("Job Function")

    with col2:
        company_profile = st.text_area("Company Profile", height=100)
        description = st.text_area("Job Description", height=150)
        requirements = st.text_area("Requirements", height=120)
        benefits = st.text_area("Benefits", height=100)

        telecommuting = st.selectbox("Telecommuting", [0, 1])
        has_company_logo = st.selectbox("Has Company Logo", [1, 0])
        has_questions = st.selectbox("Has Screening Questions", [1, 0])

    submitted = st.form_submit_button(
        "🔍 Check Job Posting",
        use_container_width=True
    )

if submitted:
    if not title.strip() or not description.strip():
        st.warning("Please enter at least the Job Title and Job Description.")
    else:
        input_df = pd.DataFrame([{
            "title": title,
            "location": location,
            "department": department,
            "salary_range": salary_range,
            "company_profile": company_profile,
            "description": description,
            "requirements": requirements,
            "benefits": benefits,
            "telecommuting": telecommuting,
            "has_company_logo": has_company_logo,
            "has_questions": has_questions,
            "employment_type": employment_type,
            "required_experience": required_experience,
            "required_education": required_education,
            "industry": industry,
            "function": function
        }])

        input_df = prepare_dataframe(input_df)
        model_input = input_df[
            ["combined_text"] + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
        ]

        prediction = int(model.predict(model_input)[0])
        probability = model.predict_proba(model_input)[0]
        fraud_probability = float(probability[1])

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Fraudulent Job Posting Detected")
        else:
            st.success("✅ Legitimate Job Posting")

        st.metric(
            "Fraud Probability",
            f"{fraud_probability * 100:.2f}%"
        )
        st.progress(fraud_probability)

        if fraud_probability >= 0.70:
            st.warning(
                "High fraud risk. Verify the company, official website, "
                "recruiter identity, and payment requests."
            )
        elif fraud_probability >= 0.40:
            st.info(
                "Moderate risk. Perform additional verification before applying."
            )
        else:
            st.info(
                "Low predicted fraud risk. The result is an ML prediction "
                "and should not replace independent verification."
            )
