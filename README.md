# 🛡️ SafeMail AI Dashboard

Advanced Email Threat Detection & Text Analytics Powered by Machine Learning. This project helps users detect whether an incoming email is **Spam** or **Safe (Ham)** using a Naive Bayes Classifier model.

## 🚀 Live Demo
👉 [Click here to view the Live App](https://streamlit.app) *(Tip: Yahan apni actual streamlit app ki link paste kar dena)*

## ✨ Features
- **📧 Single Email Analyzer:** Paste any email text to get instant AI classification with confidence scores.
- **📊 Bulk CSV Classifier:** Upload a `.csv` file to analyze multiple emails at once and visualize the results via interactive charts.
- **🔗 Link Safety Checker:** Automatically detects suspicious links containing danger keywords (like lottery, win, free).
- **🎨 Modern UI:** Clean layout styled with custom CSS and Plotly data visualizations.

## 🛠️ Tech Stack
- **Language:** Python
- **Framework:** Streamlit
- **Machine Learning:** Scikit-Learn (CountVectorizer & MultinomialNB)
- **Data Visualization:** Plotly Express

## 📂 Project Structure
- `app.py`: The core application code and logic.
- `file.csv`: Local dataset used to train the machine learning pipeline.
- `requirements.txt`: List of dependencies required to run the project.

## 💻 How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
