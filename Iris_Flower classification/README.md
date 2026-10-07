# 🌸 IrisAI — K-Means Clustering Dashboard

An interactive machine learning project that applies **K-Means Clustering** to the Iris dataset and presents the results through a **Streamlit analytics dashboard**.

The project includes both a Jupyter Notebook for the complete clustering workflow and a Streamlit application for interactive prediction, visualization, cluster profiling, and evaluation.

---

## 📌 Project Overview

This project demonstrates an end-to-end unsupervised learning workflow using the classic Iris dataset.

The notebook:

- Loads the Iris dataset, using `Iris.csv` when available and falling back to Scikit-Learn's built-in dataset.
- Explores the dataset and checks for missing values.
- Uses four numerical flower measurements as clustering features.
- Standardizes the features with `StandardScaler`.
- Evaluates K-Means for multiple values of **K (2–6)**.
- Uses the **Elbow Method** and **Silhouette Score** as clustering diagnostics.
- Uses known Iris species labels **only for post-hoc evaluation**, not for fitting K-Means.
- Selects **K = 3** as the final clustering configuration.
- Compares clusters with known species using majority-label mapping, accuracy, and a confusion matrix.
- Uses PCA to visualize the clusters in two dimensions.

The Streamlit application turns this workflow into an interactive dashboard where users can enter flower measurements and inspect the resulting cluster assignment.

---

## ✨ Features

### 🧪 Interactive Cluster Prediction
Enter:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The dashboard standardizes the input and predicts its K-Means cluster.

### 📈 PCA Cluster Visualization
A 2D PCA projection displays the Iris observations in cluster space and dynamically marks the user's input flower.

### 📊 Cluster Profiling
The dashboard shows average feature values for each cluster using an interactive Plotly bar chart.

### 📉 Clustering Evaluation
The application calculates and displays Silhouette Scores for K values from 2 through 6.

### 🤖 Automatic Model Handling
The Streamlit app can:

1. Load `iris_kmeans_model.pkl` when a saved model is available.
2. Otherwise train a scaled K-Means model automatically using `K = 3`.

### 🎨 Interactive UI
The dashboard uses Streamlit with a custom dark glassmorphism-style interface and interactive Plotly visualizations.

---

## 🧠 Machine Learning Workflow

```text
Iris Dataset
     │
     ▼
Data Loading
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Select 4 Numerical Features
     │
     ▼
StandardScaler
     │
     ▼
Evaluate K = 2, 3, 4, 5, 6
     │
     ├── Elbow Method
     └── Silhouette Score
     │
     ▼
Final K-Means Model (K = 3)
     │
     ├── Cluster Assignment
     ├── Majority-Species Mapping
     ├── Accuracy / Confusion Matrix
     └── PCA Visualization
     │
     ▼
Interactive Streamlit Dashboard
```

---

## 📊 Features Used

| Feature | Description |
|---|---|
| `SepalLengthCm` | Sepal length in centimeters |
| `SepalWidthCm` | Sepal width in centimeters |
| `PetalLengthCm` | Petal length in centimeters |
| `PetalWidthCm` | Petal width in centimeters |

The four numerical measurements are standardized before applying K-Means.

---

## 🗂️ Project Structure

```text
Iris-KMeans-Clustering/
│
├── app.py
├── Iris_KMeans_Clustering_Corrected.ipynb
├── Iris.csv                         # Optional: local Iris dataset
├── iris_kmeans_model.pkl            # Optional: saved model + scaler
├── README.md
└── requirements.txt                 # Recommended for deployment
```

> `Iris.csv` and `iris_kmeans_model.pkl` are optional for the Streamlit fallback logic. If `Iris.csv` is unavailable, the application uses Scikit-Learn's built-in Iris dataset. If the pickle model is unavailable, the application trains a K-Means model automatically.

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-Learn**
- **Streamlit**
- **Plotly**
- **Matplotlib**
- **Seaborn**
- **SciPy**
- **Jupyter Notebook**

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt` yet, install the main packages with:

```bash
pip install streamlit pandas numpy scikit-learn plotly matplotlib seaborn scipy jupyter
```

---

## ▶️ Run the Streamlit Dashboard

From the project directory:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, normally similar to:

```text
http://localhost:8501
```

Open the URL in your browser to use the dashboard.

---

## 📓 Run the Jupyter Notebook

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
Iris_KMeans_Clustering_Corrected.ipynb
```

Run the notebook cells from top to bottom to reproduce the analysis.

---

## 🔬 Model Details

The final model uses:

```text
Algorithm: K-Means
K: 3
Initialization: k-means++
n_init: 10
random_state: 42
Preprocessing: StandardScaler
```

The notebook evaluates K values from **2 to 6** before using K = 3 as the final configuration.

The Streamlit application also uses a scaled K-Means model with:

```python
KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)
```

---

## 📐 Evaluation Metrics

### Elbow Method

The Elbow Method evaluates the K-Means inertia for different values of K and helps inspect how clustering compactness changes as the number of clusters increases.

### Silhouette Score

The Silhouette Score is calculated for K values from 2 through 6 to evaluate how well-separated the resulting clusters are.

### Post-Hoc Accuracy

Because K-Means is unsupervised, the `Species` labels are **not used to train the model**.

For evaluation only, each cluster is mapped to its majority Iris species and the resulting labels are compared with the known species labels.

> Important: cluster IDs are arbitrary. A raw K-Means cluster number should not be directly compared with a species code without label mapping.

---

## 📊 Dashboard Sections

The Streamlit application contains three main tabs:

### 1. 🧪 Predict Cluster

Use sliders to enter flower measurements and view:

- Predicted cluster
- Dominant species associated with the cluster
- Input feature summary

### 2. 📈 PCA Cluster Space

Displays:

- 2D PCA projection
- K-Means cluster groups
- Iris species information when available
- Dynamic marker for the user's input flower

### 3. 📊 Cluster Profiling & Evaluation

Displays:

- Average feature values per cluster
- Silhouette Score evaluation across K values

---

## 🔄 Dataset Fallback

The application first looks for:

```text
Iris.csv
```

If it is not found, it loads the Iris dataset from:

```python
sklearn.datasets.load_iris()
```

This makes the dashboard usable even when a local `Iris.csv` file has not been added to the repository.

---

## 🚀 Deploy on Streamlit Community Cloud

1. Push the project to GitHub.
2. Make sure `app.py` and `requirements.txt` are in the repository.
3. Open Streamlit Community Cloud.
4. Connect your GitHub repository.
5. Select `app.py` as the main application file.
6. Deploy the application.

A minimal `requirements.txt` for the dashboard can contain:

```text
streamlit
pandas
numpy
scikit-learn
plotly
```

If you also want to run the complete notebook environment, include:

```text
matplotlib
seaborn
scipy
jupyter
```

---

## 📌 Example Use Case

A user can enter measurements such as:

```text
Sepal Length : 5.8 cm
Sepal Width  : 3.0 cm
Petal Length : 3.8 cm
Petal Width  : 1.2 cm
```

The application standardizes these measurements, passes them to the K-Means model, and displays the resulting cluster assignment and its majority-species mapping.

---

## 🎯 Learning Outcomes

This project demonstrates:

- Unsupervised machine learning
- K-Means clustering
- Feature standardization
- Cluster selection
- Elbow Method
- Silhouette Score
- PCA dimensionality reduction
- Post-hoc cluster evaluation
- Confusion matrix analysis
- Interactive data visualization
- Streamlit application development
- Model loading and fallback training

---

## ⚠️ Important Note

K-Means cluster labels are arbitrary identifiers. Therefore, cluster numbers do not inherently correspond to Iris species numbers.

The species mapping shown in the dashboard is based on the **majority species within each cluster** and is intended for interpretation/evaluation rather than supervised classification.

---

## 👤 Author

**Your Name**
Sherkar Manisha Minanath

