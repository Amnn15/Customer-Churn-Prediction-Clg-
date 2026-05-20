import base64
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Bank Churn Predictor", layout="wide")

# ---------------- LOGIN ----------------

def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}

    /* dark overlay */
    div[data-testid="stVerticalBlock"] {{
        background: rgba(0,0,0,0.6);
        padding: 30px;
        border-radius: 15px;
    }}
    </style>
    """, unsafe_allow_html=True)


def login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # ✅ APPLY BACKGROUND ONLY BEFORE LOGIN
    if not st.session_state.logged_in:
        set_bg("bg.jpg")   # make sure image is in same folder

    if st.session_state.logged_in:
        return True

    st.markdown("<h2 style='text-align:center;color:white'>🔐 Login</h2>", unsafe_allow_html=True)

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login", use_container_width=True):
        if u == "aman" and p == "1234":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")

    return False

if not login():
    st.stop()

# ---------------- CSS ----------------
st.markdown("""
<style>
.block-container { padding-top: 80px; }
.card {
    border-radius: 16px;
    padding: 20px;
    background: linear-gradient(135deg, #1f2937, #111827);
    color: white;
}
.title { font-size:14px; opacity:.7; }
.value { font-size:24px; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ---------------- NAVBAR ----------------
selected = option_menu(
    menu_title=None,
    options=["Home","About","Contact","FAQ","Logout"],
    icons=["house","person","envelope","question","box-arrow-right"],
    orientation="horizontal"
)

if selected == "Logout":
    st.session_state.logged_in = False
    st.rerun()

# ---------------- DATA + MODEL ----------------
@st.cache_data
def load_data():
    return pd.read_csv("Churn_Modelling.csv")

@st.cache_resource
def train(df):
    df = df.drop(["RowNumber","CustomerId","Surname"], axis=1)
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled,y_train)

    rf = RandomForestClassifier()
    rf.fit(X_train_scaled,y_train)

    return model, rf, scaler, X, X_test_scaled, y_test

# ---------------- HOME ----------------
if selected == "Home":

    st.title("🏦 Customer Churn Dashboard")

    df = load_data()
    model, rf, scaler, X, X_test, y_test = train(df)

    # ---------------- Sidebar input ----------------
    st.sidebar.header("Customer Input")

    cs = st.sidebar.slider("Credit Score",300,900,600)
    age = st.sidebar.slider("Age",18,80,35)
    bal = st.sidebar.number_input("Balance",50000)
    sal = st.sidebar.number_input("Salary",50000)
    ten = st.sidebar.slider("Tenure",0,10,3)
    prod = st.sidebar.slider("Products",1,4,1)
    geo = st.sidebar.selectbox("Geo",["India","Germany","Spain"])
    gen = st.sidebar.selectbox("Gender",["Male","Female"])
    act = st.sidebar.selectbox("Active",[0,1])
    card = st.sidebar.selectbox("Card",[0,1])

    inp = pd.DataFrame({
        'CreditScore':[cs],'Age':[age],'Tenure':[ten],'Balance':[bal],
        'NumOfProducts':[prod],'HasCrCard':[card],'IsActiveMember':[act],
        'EstimatedSalary':[sal],'Geography':[geo],'Gender':[gen]
    })

    for col in X.columns:
        if col not in inp:
            inp[col] = 0

    inp = inp[X.columns]
    scaled = scaler.transform(inp)

    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]


    # ---------------- BATCH  ----------------

st.subheader("📂 Batch Prediction (Upload CSV)")

uploaded = st.file_uploader(
    "📤 Drag & Drop or Click to Upload CSV",
    type=["csv"]
)

if uploaded is not None:

    try:
        batch_df = pd.read_csv(uploaded)

        st.success("✅ File uploaded successfully")



        # ---------------- PREVIEW ----------------
        st.subheader("👀 Data Preview")
        st.dataframe(batch_df.head())

        # ---------------- VALIDATION ----------------
        required_cols = [
            'CreditScore','Age','Tenure','Balance','NumOfProducts',
            'HasCrCard','IsActiveMember','EstimatedSalary',
            'Geography','Gender'
        ]

        missing = [col for col in required_cols if col not in batch_df.columns]

        if missing:
            st.error(f"❌ Missing columns: {missing}")
            st.stop()

        st.success("✅ All required columns present")

        # ---------------- PROCESS ----------------
        for col in X.columns:
            if col not in batch_df:
                batch_df[col] = 0

        batch_df = batch_df[X.columns]

        batch_scaled = scaler.transform(batch_df)

        batch_df["Prediction"] = model.predict(batch_scaled)
        batch_df["Probability"] = model.predict_proba(batch_scaled)[:,1]

        # ---------------- OUTPUT ----------------
        st.subheader("📊 Prediction Results")
        st.dataframe(batch_df)

        # ---------------- SUMMARY ----------------
        st.subheader("📈 Summary")

        churn_count = batch_df["Prediction"].sum()
        total = len(batch_df)

        st.write(f"Total Customers: {total}")
        st.write(f"Churn Risk Customers: {churn_count}")
        st.write(f"Churn Rate: {(churn_count/total)*100:.2f}%")

        # ---------------- DOWNLOAD ----------------
        st.download_button(
            "⬇ Download Results",
            batch_df.to_csv(index=False),
            "batch_results.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")

# ----------------^^^^^ BATCH ^^^^^----------------

    # ---------------- KPI ----------------
    c1,c2,c3 = st.columns(3)

    with c1:
        st.markdown(f"<div class='card'><div class='title'>Prediction</div><div class='value'>{'⚠️ Churn' if pred else '✅ Stay'}</div></div>", unsafe_allow_html=True)

    with c2:
        risk = "High" if prob>0.7 else "Medium" if prob>0.4 else "Low"
        st.markdown(f"<div class='card'><div class='title'>Risk</div><div class='value'>{risk}</div></div>", unsafe_allow_html=True)

    with c3:
        st.markdown(f"<div class='card'><div class='title'>Probability</div><div class='value'>{prob:.2f}</div></div>", unsafe_allow_html=True)

    # ---------------- GAUGE ----------------
    st.subheader("📊 Probability Gauge")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        gauge={'axis': {'range': [0,1]}}
    ))
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- MODEL PERFORMANCE ----------------
    st.subheader("📉 Model Performance")

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm,annot=True,fmt="d",ax=ax)
    st.pyplot(fig)

    # ---------------- MODEL COMPARISON ----------------
    st.subheader("📊 Model Comparison")

    lr_acc = accuracy_score(y_test, model.predict(X_test))
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    st.write(f"Logistic Regression Accuracy: {lr_acc:.2f}")
    st.write(f"Random Forest Accuracy: {rf_acc:.2f}")

    # ---------------- EXPLANATION ----------------
    st.subheader("🧠 Why this prediction?")

    if age > 50:
        st.write("• Higher age increases churn risk")
    if bal > 100000:
        st.write("• High balance customers tend to leave")
    if act == 0:
        st.write("• Inactive users are more likely to churn")
    if prod == 1:
        st.write("• Fewer products increase churn risk")

    # ---------------- BUSINESS IMPACT ----------------
    st.subheader("💰 Business Impact")

    if pred == 1:
        loss = int(50000 * prob)
        st.error(f"Estimated Loss: ₹{loss}")
    else:
        st.success("Customer likely to stay → No loss")

    # ---------------- SEGMENTATION ----------------
    st.subheader("👥 Customer Segmentation")

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)

    df["Segment"] = clusters
    st.dataframe(df[["Age","Balance","Segment"]].head())

    # ---------------- FEATURE IMPORTANCE ----------------
    st.subheader("📊 Feature Importance")

    importance = model.coef_[0]
    imp_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(imp_df.set_index("Feature"))

    # ---------------- WHAT-IF ----------------
    st.subheader("🔄 What-if Analysis")

    new_age = st.slider("Simulate Age Change", 18, 80, age)
    st.write(f"If Age changes to {new_age}, churn risk may change.")

    # ---------------- SHAP ----------------
    if st.checkbox("Show SHAP Explanation"):
        explainer = shap.Explainer(model, X)
        shap_values = explainer(inp)
        fig2, ax2 = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig2)

# ---------------- OTHER PAGES ----------------
elif selected == "About":
    st.title("About")
    st.write(""" This project predicts customer churn using Machine Learning. 
    - Logistic Regression Model 
    - SHAP Explainability 
    - Real-time Predictions """)

elif selected == "Contact":
    st.write("📞 Phone: +91-6398769254")
    st.write("Email: aman@gmail.com")

elif selected == "FAQ":
    st.title("FAQ")
    st.write(""" **Q: What is churn?**\n
    Answer: Customer leaving the bank.\n 
    **Q: Accuracy?** \n 
    Answer: Around 80%. """)