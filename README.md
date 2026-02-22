# 💳 Real-Time Fraud Detection System

🚀 **Live Demo:** https://fraud-detection-api-xage.onrender.com/docs

A production-ready Fraud Detection REST API built using **XGBoost** and **FastAPI**, designed to minimize **expected financial loss** instead of maximizing raw accuracy.

This project demonstrates financial risk modeling, cost-sensitive machine learning, and cloud deployment.

---

## 🚀 Project Overview

This system is built using the Kaggle Credit Card Fraud dataset:

- 284,807 transactions  
- 492 fraud cases  
- 0.17% fraud rate (extreme class imbalance)

The model detects fraudulent transactions and recommends actions based on **expected financial impact**, not just probability.

---

## 🧠 Key Features

- Extreme class imbalance handling  
- XGBoost with `scale_pos_weight`  
- ROC-AUC and PR-AUC evaluation  
- Cost-sensitive decision logic  
- Expected loss modeling  
- REST API with FastAPI  
- Interactive Swagger documentation  
- Cloud deployment ready  

---

## 💰 Business Logic (Core Strength)

Instead of using a fixed probability threshold, the system compares:

**Expected Loss if Allowed**  
= P(Fraud) × Fraud_Loss  

**Expected Loss if Blocked**  
= (1 − P(Fraud)) × False_Block_Cost  

### Assumptions:
- Fraud allowed → $1000 loss  
- Legit transaction blocked → $10 inconvenience cost  

The system selects the action that minimizes expected financial loss.

---

## 📊 Model Performance

- ROC-AUC: ~0.97+  
- PR-AUC: ~0.85+  
- Handles extreme 0.17% fraud imbalance  
- Optimized for financial risk reduction  

---

## 📂 Project Structure


fraud-detection-system/
│
├── data/ # Dataset (ignored in GitHub)
├── src/ # Preprocessing & training scripts
├── models/ # Saved model artifacts
├── app/ # FastAPI application
├── requirements.txt
└── README.md


---

## 🔌 API Usage

### Run Locally

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --reload

Open:

http://127.0.0.1:8000/docs
POST /predict

Example Response:

{
  "fraud_probability": 0.87,
  "expected_loss_if_allowed": 870.0,
  "expected_loss_if_blocked": 1.3,
  "recommended_action": "Block Transaction 🚨"
}
🌐 Deployment

The API is cloud-deployable using:

Render

Railway

AWS

Azure

Docker containers

Swagger documentation available at:

https://your-app-name.onrender.com/docs
🛠 Tech Stack

Python

XGBoost

Scikit-learn

FastAPI

Uvicorn

Pandas

🏆 Why This Project Is Strong

This project demonstrates:

Financial risk modeling

Cost-sensitive ML optimization

Handling extreme imbalance datasets

Production-grade API design

Cloud deployment capability

Business-driven decision systems

👩‍💻 Author

Soumyasree Mitra

Aspiring Machine Learning Engineer



