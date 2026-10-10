# Detecting Fraud Jobs Using NLP

A Streamlit machine learning application that predicts whether a job posting is **legitimate** or **fraudulent**.

## Project Features

- NLP text cleaning
- TF-IDF vectorization with unigrams and bigrams
- Categorical feature encoding
- Numerical feature preprocessing
- Logistic Regression with class balancing
- Fraud probability prediction
- Streamlit web interface
- Automatic model training and model saving

## Dataset

Use the Kaggle `fake_job_postings.csv` dataset.

The application expects these columns:

`job_id`, `title`, `location`, `department`, `salary_range`,
`company_profile`, `description`, `requirements`, `benefits`,
`telecommuting`, `has_company_logo`, `has_questions`,
`employment_type`, `required_experience`, `required_education`,
`industry`, `function`, `fraudulent`

## Project Folder Structure

```text
fake_job_detection_app/
├── app.py
├── requirements.txt
├── README.md
└── fake_job_postings.csv
```

After the first successful training run, the application creates:

```text
fraud_job_model.pkl
```

## Installation

Open Anaconda Prompt, Command Prompt, or the Jupyter terminal and move to the project folder.

```bash
cd fake_job_detection_app
```

Create a virtual environment if required:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

## Run the Application

Make sure `fake_job_postings.csv` is in the same folder as `app.py`.

Run:

```bash
streamlit run app.py
```

Streamlit will open the application in your browser.

## How the Model Works

The text columns `title`, `company_profile`, `description`,
`requirements`, and `benefits` are combined and cleaned.

TF-IDF converts the cleaned text into numerical NLP features.
Categorical columns are converted using One-Hot Encoding.
The binary columns are processed as numerical features.

A class-balanced Logistic Regression model predicts:

- `0` = Legitimate Job
- `1` = Fraudulent Job

## Important Note

This application is an educational machine learning project.
A prediction should not be treated as final proof that a company or
job posting is fraudulent. Always independently verify the company,
recruiter, official website, and job details.

## Project Title

**Detecting Fraud Jobs: Shielding Job Seekers Using NLP**
