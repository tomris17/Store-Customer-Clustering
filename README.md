# Mall / Store Customer Clustering Project

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

This repository contains an unsupervised machine learning clustering project that groups real retail store customers based on age, total purchase amount, and product purchase quantity[cite: 12].

---

## Dataset Notice
*Note: The dataset (`Different_stores_data.csv`) used in this project[cite: 12] can be obtained from retail sales data sources or Kaggle.*

---

## Dataset Features & Engineering
* **age**: Customer age[cite: 12].
* **quantity**: Number of units purchased[cite: 12].
* **selling_price_per_unit**: Price per unit of the item.
* **Total_Purchase_Amount**: Engineered feature calculated by multiplying `quantity` and `selling_price_per_unit`[cite: 12].
* **customer_id**: Unique identifier aggregated to group store transactions by individual customers[cite: 12].

---

## Project Workflow
* **Feature Engineering**: Computing total purchase amounts per transaction and aggregating data per customer (`age`, `Total_Purchase_Amount`, `quantity`)[cite: 12].
* **Preprocessing**: Dropping missing values[cite: 12] and scaling features using `StandardScaler`[cite: 12].
* **Optimal Cluster Detection**: Applying the Elbow Method via Yellowbrick's `KElbowVisualizer` to determine the optimal number of clusters[cite: 12].
* **Model Training**: Fitting a `KMeans` clustering model on the processed store customer dataset[cite: 12].
* **Model Persistence**: Exporting the trained model using `joblib` into `real_data_clustering_model.pkl`[cite: 12].
* **Web Application**: Interactive deployment interface built with Streamlit.

---

## Getting Started & Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/store-customer-clustering.git](https://github.com/YOUR_USERNAME/store-customer-clustering.git)
   cd store-customer-clustering
