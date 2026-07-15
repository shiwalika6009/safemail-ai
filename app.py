import sys
import asyncio

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="SafeMail AI", page_icon="🛡️", layout="wide")

# --- CLEAN PREMIUM LIGHT UI (COMBINED TITLE CARD) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #E6EDF5 !important;
        color: #111827 !important;
    }
    
    /* Top Header Icons */
    header[data-testid="stHeader"] svg {
        fill: #0F172A !important;
    }
    header[data-testid="stHeader"] button {
        color: #0F172A !important;
    }
    
    /* Combined Header Card Design */
    .header-card {
        background-color: #FFFFFF !important;
        border: 2px solid #CBD5E1 !important;
        border-radius: 14px !important;
        padding: 25px !important;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
        margin-bottom: 25px !important;
        margin-top: -20px !important;
    }
    
    /* Main Title Styling */
    .main-title {
        font-size: 42px !important;
        font-weight: 800 !important;
        color: #1E3A8A !important;
        margin-bottom: 8px !important;
    }
    
    /* Subtitle Styling */
    .sub-title {
        font-size: 17px !important;
        color: #475569 !important;
        font-weight: 500;
        margin: 0px !important;
    }

    /* Target ONLY real visible containers, remove empty wrapper boxes */
    div[data-testid="stBlock"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0px !important;
    }

    /* Content Area white rounded background layout */
    div.element-container:has(textarea), 
    div.element-container:has(div.stButton), 
    div[data-testid="stDataFrameContainer"],
    div[data-testid="element-container"]:has(.stAlert),
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 2px solid #CBD5E1 !important;
        border-radius: 14px !important;
        padding: 20px !important;
        margin-bottom: 15px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
    }

    /* Radio button panel wrapper styling */
    div[data-testid="stWidgetLabel"] + div {
        background-color: #FFFFFF !important;
        border: 2px solid #CBD5E1 !important;
        border-radius: 14px !important;
        padding: 15px 20px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Blue Button Design */
    div.stButton > button:first-child {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 700 !important;
        padding: 12px 24px !important;
        border-radius: 10px !important;
        width: 100%;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.15) !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #1D4ED8 !important;
    }

    /* Darker input fonts */
    textarea {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 2px solid #94A3B8 !important;
        border-radius: 10px !important;
    }
    
    h3 {
        color: #0F172A !important;
        font-weight: 800 !important;
        margin-top: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- COMBINED TITLE & SUBTITLE IN ONE BOX ---
st.markdown("""
    <div class="header-card">
        <div class="main-title">🛡️ SafeMail AI Dashboard</div>
        <div class="sub-title">Advanced Email Threat Detection & Text Analytics Powered by Machine Learning</div>
    </div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_full_dataset_and_train():
    comprehensive_data = {
        'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham'],
        'text': [
            "Get free lottery worth 10000 dollars now! Click link", "Hey, are we still meeting for lunch today at the cafeteria?",
            "Congratulations! You won a free iPhone. Claim your cash prize instantly", "Please review the attached project documentation and report by EOD.",
            "URGENT! Your bank account has been suspended. Verify your login credentials now.", "Hi team, the project meeting is postponed to 4 PM tomorrow.",
            "Win cash prize instantly click this link to secure your bonus", "Can you send me the quarterly financial report before the weekend?",
            "Dear customer, your credit card security is compromised. Reset password here.", "Thanks for the update. I will check the details and get back to you.",
            "Double your income in one week! Work from home slots available now.", "Are you free for a quick call regarding the software deployment?",
            "EXCLUSIVE DEAL! Click to get 90% discount on premium electronics.", "Let's schedule the interview for the new Python developer position.",
            "Your subscription has expired. Renew now to avoid auto-debit penalties.", "Can we review the marketing strategy slides during the morning standup?",
            "Free entry token for casino. Play now and win millions tonight!", "Kindly submit your expense receipts to the HR department by Friday.",
            "You have been selected for a luxury holiday package. Call this number now.", "The code has been pushed to the main branch. Please review the PR.",
            "INVEST NOW! High return bitcoin investment opportunity guaranteed.", "Just wanted to wish you a very happy birthday! Have a great day ahead.",
            "Suspicious login attempt detected on your profile. Secure your account.", "Could you please forward me the tracking number for the shipment?"
        ]
    }
    df = pd.DataFrame(comprehensive_data)
    pipeline = Pipeline([('vectorizer', CountVectorizer(stop_words='english')), ('nb', MultinomialNB())])
    pipeline.fit(df['text'], df['label'])
    return pipeline

model = load_full_dataset_and_train()
st.sidebar.success("🛡️ SafeMail Core: Active")

def check_urls(text):
    urls = re.findall(r'(https?://\S+|www\.\S+|\\b\\w+\\.(?:com|org|net|in|xyz)\\b)', text)
    if not urls:
        return "✅ No links detected in this email."
    suspicious_words = ['free', 'win', 'lottery', 'gift', 'login', 'verify', 'bank', 'cashback', 'secure']
    for url in urls:
        if any(word in url.lower() for word in suspicious_words):
            return f"🚨 Warning: Suspicious link identified: `{url}`"
    return f"ℹ️ Links found (`{urls}`), but they appear to be safe."

# Choose layout mode
choice = st.radio("Select Analysis Mode:", ["📧 Single Email Analyzer", "📊 Bulk CSV Classifier"], horizontal=True)

if choice == "📧 Single Email Analyzer":
    st.markdown("### 🖋️ Paste Your Email Content")
    user_input = st.text_area("Enter email text to analyze:", height=150, key="single_input", placeholder="Type or paste your email content here...")
    analyze_btn = st.button("🔍 Run AI Diagnostics")
    
    if analyze_btn:
        if user_input.strip() != "":
            prediction = model.predict([user_input])[0]
            probabilities = model.predict_proba([user_input])
            
            class_labels = list(model.classes_)
            pred_index = class_labels.index(prediction)
            confidence = probabilities[0][pred_index] * 100
            
            st.markdown("### 📊 AI Assessment Output")
            if prediction == 'spam':
                st.error("🚨 ASSESSMENT: SPAM DETECTED")
            else:
                st.success("✅ ASSESSMENT: SAFE / LEGITIMATE")
            
            st.metric(label="🤖 AI Confidence Score", value=f"{confidence:.2f}%")
            st.info(f"🔗 **Link Safety Evaluation:** {check_urls(user_input)}")
        else:
            st.warning("⚠️ Please enter some email text before running diagnostics.")

else:
    st.markdown("### 📁 Upload Emails in Bulk")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df_bulk = pd.read_csv(uploaded_file)
        text_col = None
        for col in ['text', 'email_text', 'email', 'body']:
            if col in df_bulk.columns:
                text_col = col
                break
        
        if text_col:
            st.success(f"Found data column: `{text_col}`")
            df_bulk['AI_Prediction'] = model.predict(df_bulk[text_col].astype(str))
            counts = df_bulk['AI_Prediction'].value_counts()
            
            st.dataframe(df_bulk, use_container_width=True)
            
            fig_pie = px.pie(names=counts.index, values=counts.values, title="Bulk Scan Results Distribution", template="plotly_white", color=counts.index, color_discrete_map={'spam': '#EF4444', 'ham': '#10B981'})
            st.plotly_chart(fig_pie, use_container_width=True)
            
            csv_data = df_bulk.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Classified CSV", data=csv_data, file_name="safemail_ai_predictions.csv", mime="text/csv", use_container_width=True)
        else:
            st.error("Error: CSV must contain a column named 'text' or 'email_text'.")
