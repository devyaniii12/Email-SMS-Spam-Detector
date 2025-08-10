# Email-SMS Spam Detector

## Overview

The **Email-SMS Spam Detector** is a machine learning-based application designed to classify messages as "Spam" or "Not Spam" (Ham). This tool utilizes Natural Language Processing (NLP) techniques and a **Multinomial Naive Bayes** classifier to effectively filter unwanted messages in both emails and SMS.

## Features

- **User-Friendly Interface**: Input a message and receive a real-time classification.
- **Pre-trained Model**: Utilizes a Multinomial Naive Bayes model trained on a dataset of SMS messages vectorized using TF-IDF.
- **Scalability**: Easily extendable to incorporate email data and more sophisticated models.

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - scikit-learn (for Multinomial Naive Bayes classifier and TF-IDF vectorization)
  - nltk (for natural language processing tasks)
  - streamlit (for building the web application interface)

## Project Structure
plaintext
```
Email-SMS-Spam-Detector/
├── app.py               
├── model.pkl             
├── vectorizer.pkl        
├── requirements.txt      
├── setup.sh            
└── README.md
```
## Installation

### Clone the repository

bash
```
git clone https://github.com/devyaniii12/Email-SMS-Spam-Detector.git
cd Email-SMS-Spam-Detector
```

### Install required packages
bash
```
pip install -r requirements.txt
```
### Run the app locally
```
streamlit run app.py
```

### Usage
- Enter an email or SMS message into the input box.

- Click Predict to check if the message is Spam or Not Spam.

## Deployment

The app is deployed and accessible here:  
[streamlit-app-link](https://email-sms-spam-detector-y2tyunxa3ncfxfydv5zyzx.streamlit.app/) 

## Model Performance

The model achieves approximately **96.7% accuracy** on the test data.
