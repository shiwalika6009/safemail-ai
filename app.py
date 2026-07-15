import sys
import asyncio

# 1. Windows Python 3.12 asyncio bug fix (Taaki WinError 10054 na aaye)
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

# 2. Page Configuration (Modern Layout)
st.set_page_config(page_title="SafeMail AI", page_icon="🛡️", layout="wide")

# --- CUSTOM CSS FOR REFRESHED BACKGROUND & THEME ---
st.markdown("""
    <style>
    /* Cool Soft Gray-Blue Background for Premium Tech Product Vibe */
    .stApp {
        background-color: #f1f5f9;
    }
    /* Dark Premium Slate Blue for Main Title */
    .main-title {
        font-size: 42px !important;
        font-weight: 800 !important;
        color: #0F172A;
        text-align: center;
        margin-bottom: 5px;
        letter-spacing: -0.5px;
    }
    /* Subtitle in sleek gray */
    .sub-title {
        font-size: 18px !important;
        color: #64748B;
        text-align: center;
        margin-bottom: 35px;
    }
    /* Button Custom Color (Modern Teal/Sky Blue) */
    div.stButton > button:first-child {
        background-color: #0EA5E9 !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        border-radius: 8px !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #0284C7 !important;
        color: white !important;
    }
    /* Tab active indicator decoration */
    .stTabs [data-baseweb="tab"] {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #0EA5E9 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Main Titles (Proper English)
st.markdown('<div class="main-title">🛡️ SafeMail AI Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced Email Threat Detection & Text Analytics Powered by Machine Learning</div>', unsafe_allow_html=True)

# 3. Complete Real-World Local Dataset (No Internet Required)
@st.cache_resource
def load_full_dataset_and_train():
    comprehensive_data = {
        'label': [
            'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham',
            'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham',
            'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham'
        ],
        'text': [
            "Get free lottery worth 10000 dollars now! Click link", 
            "Hey, are we still meeting for lunch today at the cafeteria?",
            "Congratulations! You won a free iPhone. Claim your cash prize instantly", 
            "Please review the attached project documentation and report by EOD.",
            "URGENT! Your bank account has been suspended. Verify your login credentials now.", 
            "Hi team, the project meeting is postponed to 4 PM tomorrow.",
            "Win cash prize instantly click this link to secure your bonus", 
            "Can you send me the quarterly financial report before the weekend?",
            "Dear customer, your credit card security is compromised. Reset password here.",
            "Thanks for the update. I will check the details and get back to you.",
            "Double your income in one week! Work from home slots available now.",
            "Are you free for a quick call regarding the software deployment?",
            "EXCLUSIVE DEAL! Click to get 90% discount on premium electronics.",
            "Let's schedule the interview for the new Python developer position.",
            "Your subscription has expired. Renew now to avoid auto-debit penalties.",
            "Can we review the marketing strategy slides during the morning standup?",
            "Free entry token for casino. Play now and win millions tonight!",
            "Kindly submit your expense receipts to the HR department by Friday.",
            "You have been selected for a luxury holiday package. Call this number now.",
            "The code has been pushed to the main branch. Please review the PR.",
            "INVEST NOW! High return bitcoin investment opportunity guaranteed.",
            "Just wanted to wish you a very happy birthday! Have a great day ahead.",
            "Suspicious login attempt detected on your profile. Secure your account.",
            "Could you please forward me the tracking number for the shipment?"
        ]
    }
    
    df = pd.DataFrame(comprehensive_data)
    
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(stop_words='english')), 
        ('nb', MultinomialNB())
    ])
    pipeline.fit(df['text'], df['label'])
    return pipeline, len(df)

model, total_rows = load_full_dataset_and_train()
st.sidebar.success(f"🛡️ SafeMail AI: Active & Fully Loaded!")

# Helper function: URL Safety Checker
def check_urls(text):
    urls = re.findall(r'(https?://\S+|www\.\S+|\b\w+\.(?:com|org|net|in|xyz)\b)', text)
    if not urls:
        return "No Links Found", "✅ No links detected in this email."
    
    suspicious_words = ['free', 'win', 'lottery', 'gift', 'login', 'verify', 'bank', 'cashback', 'secure']
    for url in urls:
        if any(word in url.lower() for word in suspicious_words):
            return "Unsafe Links Detected", f"🚨 Warning: Suspicious link identified: `{url}`"
    return "Safe Links", f"ℹ️ Links found (`{urls}`), but they appear to be safe."

# --- NAVIGATION TABS ---
tab1, tab2 = st.tabs(["📧 Single Email Analyzer", "📊 Bulk CSV Classifier"])

# --- TAB 1: SINGLE EMAIL ANALYZER ---
with tab1:
    left_col, right_col = st.columns([1.1, 0.9], gap="large")
    
    with left_col:
        st.markdown("### 🖋️ Paste Your Email Content")
        user_input = st.text_area("Enter email text to analyze:", height=180, key="single_input", placeholder="Type or paste your email content here...")
        analyze_btn = st.button("🔍 Run AI Diagnostics", use_container_width=True)
    
    with right_col:
        st.markdown("### 📊 AI Assessment & Analytics")
        if analyze_btn:
            if user_input.strip() != "":
                # Prediction & Confidence (Fixed Logic)
                prediction = model.predict([user_input])[0]
                probabilities = model.predict_proba([user_input])[0]
                
                class_labels = list(model.classes_)
                pred_index = class_labels.index(prediction)
                confidence = probabilities[pred_index] * 100
                
                # Results Display
                if prediction == 'spam':
                    st.error("🚨 ASSESSMENT: SPAM DETECTED")
                else:
                    st.success("✅ ASSESSMENT: SAFE / LEGITIMATE")
                
                # Metrics
                c1, c2 = st.columns(2)
                with c1:
                    st.metric(label="🤖 AI Confidence", value=f"{confidence:.2f}%")
                with c2:
                    word_count = len(user_input.split())
                    st.metric("📝 Word Count", word_count)
                
                # URL Status Box
                url_status, url_msg = check_urls(user_input)
                st.info(f"🔗 **Link Safety:** {url_msg}")
                
                # Clean Minimalist Chart
                chart_data = pd.DataFrame({
                    'Metrics': ['Your Email', 'Avg Spam', 'Avg Safe'],
                    'Words': [word_count, 15, 8]
                })
                fig = px.bar(chart_data, x='Metrics', y='Words', color='Metrics', 
                             title="Length Benchmark Analysis", 
                             template="plotly_white",
                             color_discrete_sequence=['#0EA5E9', '#64748B', '#94A3B8'])
                fig.update_layout(showlegend=False, height=220, margin=dict(t=30, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please enter some text before analyzing.")
        else:
            st.info("👈 Enter email content on the left and click the button to see AI results.")

# --- TAB 2: BULK CSV CLASSIFIER ---
with tab2:
    st.markdown("### 🗂️ Batch Processing System")
    st.write("Upload a CSV file containing multiple emails to process them all at once and download the report.")
    
    uploaded_file = st.file_uploader("Choose your .csv file (Column header must be 'text')", type=["csv"])
    
    if uploaded_file is not None:
        df_user = pd.read_csv(uploaded_file)
        
        if 'text' in df_user.columns:
            df_user['AI Prediction'] = model.predict(df_user['text'])
            
            b_col1, b_col2 = st.columns([1.2, 0.8], gap="medium")
            
            with b_col1:
                st.markdown("#### 📋 Processed Emails Preview")
                st.dataframe(df_user, use_container_width=True, height=300)
                
                csv = df_user.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Download Complete Report (.CSV)", data=csv, file_name="ai_classified_report.csv", mime="text/csv", use_container_width=True)
            
            with b_col2:
                st.markdown("#### 📈 Threat Distribution")
                summary = df_user['AI Prediction'].value_counts().reset_index()
                summary.columns = ['Result', 'Count']
                
                # Fixed color map for chart consistency
                fig_pie = px.pie(summary, values='Count', names='Result', hole=0.4,
                                 color='Result',
                                 color_discrete_map={'ham': '#10B981', 'spam': '#EF4444'})
                fig_pie.update_layout(height=280, margin=dict(t=20, b=0, l=0, r=0))
