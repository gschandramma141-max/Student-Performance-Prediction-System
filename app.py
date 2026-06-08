import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ---------------- UI ----------------
st.set_page_config(page_title="Student Performance App")
st.title("🎓 Student Performance Prediction System")

# ---------------- Load dataset ----------------
url = "https://raw.githubusercontent.com/iNeuron-Pvt-Ltd/statistics-documentation/main/StudentsPerformance.csv"
df = pd.read_csv(url)

df.columns = df.columns.str.strip()

# ---------------- Create target ----------------
df["Average"] = (df["math score"] + df["reading score"] + df["writing score"]) / 3
df["Result"] = df["Average"].apply(lambda x: 1 if x >= 40 else 0)

# ---------------- Encoding ----------------
encoders = {}
categorical_cols = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course"
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ---------------- Features & Target ----------------
X = df.drop(["Result", "Average"], axis=1)
y = df["Result"]

# ---------------- Train model ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------- UI Inputs ----------------
st.subheader("Enter Student Details")

gender = st.selectbox("Gender", list(encoders["gender"].classes_))
race = st.selectbox("Race/Ethnicity", list(encoders["race/ethnicity"].classes_))
education = st.selectbox("Parental Education", list(encoders["parental level of education"].classes_))
lunch = st.selectbox("Lunch", list(encoders["lunch"].classes_))
test_prep = st.selectbox("Test Preparation", list(encoders["test preparation course"].classes_))

math = st.number_input("Math Score", 0, 100, 50)
reading = st.number_input("Reading Score", 0, 100, 50)
writing = st.number_input("Writing Score", 0, 100, 50)

# convert input using trained encoders
input_data = [
    encoders["gender"].transform([gender])[0],
    encoders["race/ethnicity"].transform([race])[0],
    encoders["parental level of education"].transform([education])[0],
    encoders["lunch"].transform([lunch])[0],
    encoders["test preparation course"].transform([test_prep])[0],
    math,
    reading,
    writing
]

# ---------------- Prediction ----------------
if st.button("Predict Result"):
    prediction = model.predict([input_data])

    if prediction[0] == 1:
        st.success("🎉 Student will PASS")
    else:
        st.error("❌ Student will FAIL")

# ---------------- Graph ----------------
st.subheader("📊 Pass vs Fail Distribution")

fig, ax = plt.subplots()
df["Result"].value_counts().plot(kind="bar", ax=ax)

ax.set_xlabel("Result (0 = Fail, 1 = Pass)")
ax.set_ylabel("Count")
ax.set_title("Student Performance Distribution")

st.pyplot(fig)

# ---------------- Footer ----------------
st.write("---")
st.write("✔ Logistic Regression Model")
st.write("✔ Streamlit App Running Successfully")