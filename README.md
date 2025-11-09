# BarkoTell

A Deep Learning Solution for Canine Vocalization Analysis

This project is a highly technical, end-to-end machine learning pipeline that transforms a challenging bio-acoustic problem into an innovative consumer application. It demonstrates the use of deep learning to classify and translate complex, non-human sounds, bridging the communication gap between humans and their canine companions.

## ⚙️ Technical Architecture & Machine Learning Pipeline

Our solution treats sound classification as an image recognition problem by leveraging the power of deep learning. The core of the system is a multi-stage pipeline designed for efficiency and accuracy.

### 1. Data Ingestion & Preprocessing

We sourced and meticulously labeled a diverse dataset of dog vocalizations, including barks, whimpers, and howls. The data was classified based on key acoustic features: sentiment, tempo, and repetitiveness. This rigorous data classification process was fundamental to the model's ability to generalize and make accurate predictions.

### 2. Audio-to-Spectrogram Conversion

Raw audio files were preprocessed and converted into spectrograms—visual representations of the sound's frequency over time. This crucial step transformed the time-series audio data into a format (an image) that our deep learning model could easily consume.

### 3. Model Training on Azure ML

![Neural Network Training Plan](images/NNTrainingPlan.jpg)
![Accuracy vs Validation Graph](images/accuracyValidationvsTrainingGraph.jpg)

The spectrograms were used to train a custom deep learning model. We utilized Azure Machine Learning for this process, which provided a robust, GPU-accelerated environment. Azure ML allowed us to build an efficient ML pipeline that streamlined experimentation and reduced training time significantly. The model's performance was closely monitored, with training and validation accuracy curves indicating effective learning and generalization.

### 4. Classification & Interpretation

![Grad-CAM Heatmap](images/gradmap.jpg)

The trained model analyzes the spectrogram and outputs a predicted class (e.g., "Excited," "Anxious," "Warning"). A key aspect of our approach is the ability to interpret these predictions. Grad-CAM (Gradient-weighted Class Activation Mapping) could be implemented in future iterations to generate a heatmap over the spectrogram, highlighting the specific frequency-time regions that most contributed to the model's decision, thereby providing an extra layer of explainability to our classifications.

The model was deployed and integrated into a responsive web application using Flask. This served as the bridge between our backend Python logic and the front end. The web interface, designed with Figma and built with HTML, CSS, JavaScript, and JQuery, provides a seamless user experience for live recording or uploading audio files.

## 🚧 Technical Challenges & Strategic Learnings

Building this project in a short period presented significant technical hurdles:

- **Dataset Acquisition:** Finding a diverse and well-labeled dataset of dog vocalizations was a primary challenge, as data quality is paramount to a model's performance.

- **Environment Discrepancies:** We faced challenges migrating frontend code developed in Repl to a Flask environment, which highlighted the importance of consistent development pipelines.

## 🩺 Health Scoring Model

A lightweight logistic regression model that estimates a pet’s probability of being healthy based on its vital signs and behavioral indicators.
This model is currently trained on synthetic data for demonstration and UI testing inside the Pet Health Timeseries App.

## 📘 Overview

The model produces:

health_proba → Probability (0–1) that the pet is healthy

health_label → "Healthy" or "Not healthy" depending on a 0.5 threshold

The output integrates with the Streamlit dashboard (app_health_timeseries_plotly.py) for visualization and monitoring.

This is to give a demo of an output that would eventually be part of the app

## ⚙️ Model Architecture
Component	Description
Scaler	StandardScaler (feature normalization)
Estimator	Logistic Regression (max_iter=1000)
Output	predict_proba(X)[:,1] → Health probability

## 🧠 Training Setup

Data: Synthetic, generated via synth(n=4000)

Split: 80% train / 20% test (stratified)

Evaluation Metric: ROC AUC on holdout set

Saved Artifact: models/health_lr.joblib

Threshold: 0.5 for binary classification

## 📊 Features Used
Category	Features
Demographics	age, weight_kg
Vitals	hr, rr, temp_c, sbp, dbp, spo2
Behavioral	activity_level, appetite_score
Symptoms	has_vomiting, has_diarrhea

All inputs must be numeric. Symptom indicators are binary (0 or 1).

## 🧩 Synthetic Label Generation

The training label (healthy ∈ {0,1}) is created from a rule-based score:
score =
  + 0.9*(SpO2 - 92)
  - 0.5*max(0, |HR - 95| - 20)
  - 0.6*max(0, |RR - 22| - 8)
  - 4.5*max(0, |TempC - 38.6| - 0.7)
  - 0.03*max(0, SBP - 160)
  - 0.04*max(0, 90 - DBP)
  + 1.5*ActivityLevel
  + 1.2*AppetiteScore
  - 3.0*has_vomiting
  - 2.5*has_diarrhea

After adding Gaussian noise N(0,2), any record with score > 2.5 is labeled Healthy (1), else Not healthy (0).

## 🧮 Inference Pipeline

1. Load or Train

If models/health_lr.joblib exists → load it.

Otherwise → synthesize data → train → save artifact.

2. Ingest CSV

Validate schema and convert feature columns to numeric types.

3. Predict

health_proba = model.predict_proba(X)[:,1]
health_label = np.where(health_proba >= 0.5, "Healthy", "Not healthy")

4. Write-back

Append predictions to the CSV and create a timestamped backup.

5. Visualize

Streamlit dashboard plots:

Combined vitals (HR, RR, Temp, SBP, DBP)

Activity Level

Appetite Score

Overall Health Score trend

## 📤 Outputs

| Column             | Description                               |
| ------------------ | ----------------------------------------- |
| **`health_proba`** | Probability (0–1) that the pet is healthy |
| **`health_label`** | `"Healthy"` or `"Not healthy"`            |


## ⚠️ Caveats

The model is not medically validated — labels are synthetic and approximate.

We have used it for testing and visualization only until trained on real data.

We will replace synth() with genuine clinical or sensor-based datasets before deployment.

## 🚀 Next Steps

Train on real labeled veterinary data.

Add temporal features (e.g., short-term HR or Temp deltas).

Calibrate probabilities

Add drift monitoring and model card documentation.

## 📁 Files

models/health_lr.joblib       # saved logistic regression pipeline
app/app_health_timeseries_plotly.py  # Streamlit UI that uses this model
data/vitals/pet_vitals_test.csv      # example dataset for predictions
docs/model_card.md                   # model documentation

