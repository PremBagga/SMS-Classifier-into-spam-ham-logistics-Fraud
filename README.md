# SMS-Classifier-into-spam-ham-logistics-Fraud

# SMS Classification System

## 📌 Project Overview
This project is an **AI-powered SMS Classification System** that categorizes messages into different types such as Spam, Ham (Not Spam), Fraud, OTP, and Logistics. It leverages **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques to analyze and classify messages effectively.

## 🚀 Features
- **Multi-category Classification**: Classifies SMS into Spam, Not Spam, Fraud, OTP, and Logistics.
- **Text Preprocessing**: Tokenization, stemming, and stopword removal.
- **Machine Learning Model**: Trained using a TF-IDF vectorizer and a predictive model.
- **Interactive UI**: Built with **Streamlit** for user-friendly message classification.
- **Order Tracking Feature**: Redirects to order tracking for logistics-related messages.

## 📂 Project Structure
```
📁 SMS_Classifier
│-- app.py                   # Streamlit App for classification
│-- sms-spam-detection.ipynb # Model training and evaluation
│-- vectorizer.pkl           # TF-IDF vectorizer
│-- model.pkl                # Trained ML model
│-- dataset.csv              # SMS dataset (if applicable)
│-- README.md                # Project documentation
```

## 🔧 Installation & Setup
1. **Clone the repository**
   ```sh
   git clone https://github.com/yourusername/sms-classifier.git
   ```
2. **Navigate to the project directory**
   ```sh
   cd sms-classifier
   ```
3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
4. **Run the Streamlit App**
   ```sh
   streamlit run app.py
   ```

## 📊 How It Works
1. Enter an SMS message in the text area.
2. Click on the **Predict** button.
3. The model classifies the SMS into one of the categories:
   - **Not Spam** ✅
   - **Spam** 🚨
   - **Fraud** ⚠️
   - **OTP** 🔑
   - **Logistics** 📦 (with an option to track orders)

## ⚡ Model Training Pipeline
1. **Data Preprocessing**: Converts text to lowercase, removes punctuation & stopwords, tokenizes, and applies stemming.
2. **Feature Extraction**: Uses **TF-IDF vectorization** to convert text into numerical format.
3. **Model Training**: Trains a classification model using machine learning algorithms.
4. **Evaluation**: Assesses performance using accuracy, precision, recall, and F1-score.

## 🤖 Technologies Used
- **Python** (NLTK, Scikit-learn, Pandas, NumPy, Streamlit)
- **Machine Learning Algorithms**
- **TF-IDF Vectorization**
- **Streamlit for UI**

## 🛡️ Future Enhancements
- **Deep Learning Integration**
- **Multilingual SMS Classification**
- **Real-time SMS Filtering**
- **Enhanced Fraud Detection Mechanisms**

---
📲 **Stay Secure & Organized! Classify Messages with AI!** 🚀

