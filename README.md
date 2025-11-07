# 🩺 Diabetic Retinopathy Detection through Machine Learning

This project uses **Deep Learning (DenseNet121 CNN model)** to detect the stages of **Diabetic Retinopathy (DR)** from retinal fundus images. It also includes a **Streamlit web application** for easy image upload, visualization, and prediction — complete with image segmentation and audio feedback of predictions using **gTTS**.

---

## 📘 Project Overview

**Diabetic Retinopathy (DR)** is a diabetes complication that affects the eyes. Detecting DR early can prevent vision loss.
This project automates the detection of DR severity levels from retinal images using a pre-trained **DenseNet121** model fine-tuned on a custom dataset.

The app:

* Accepts one or more retinal images.
* Preprocesses and resizes images (65x65).
* Predicts the DR severity category.
* Displays original and segmented images.
* Converts the prediction result into speech using **Google Text-to-Speech (gTTS)**.

---

## 🧠 Model Details

* **Architecture:** DenseNet121 (Transfer Learning)
* **Input Size:** 65 × 65 × 3
* **Output Classes:**

  * `No_DR` (No Diabetic Retinopathy)
  * `Mild`
  * `Moderate`
  * `Severe`
  * `Proliferate_DR`
* **Optimizer:** Adam
* **Loss Function:** Categorical Cross-Entropy
* **Metrics:** Accuracy
* **Dataset Split:** 80% training / 20% testing

The trained model is saved as `model.h5` and loaded in the Streamlit app for predictions.

---

## 📂 Project Structure

```
📁 Diabetic-Retinopathy-Detection/
│
├── 📁 Dataset/
│   ├── Mild/
│   ├── Moderate/
│   ├── No_DR/
│   ├── Proliferate_DR/
│   └── Severe/
│
├── 📁 background/
│   └── 2.jpg
│
├── model.h5
├── app.py                   # Streamlit web app
├── train_model.py           # Model training script
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Diabetic-Retinopathy-Detection.git
cd Diabetic-Retinopathy-Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🧾 Requirements

Add this to your **requirements.txt**:

```
streamlit
opencv-python
numpy
matplotlib
tensorflow
keras
scikit-learn
mlxtend
Pillow
gTTS
```

---

## 🖼️ Usage Guide

1. Launch the app with:

   ```bash
   streamlit run app.py
   ```
2. Upload one or multiple retinal images.
3. The app will:

   * Display the uploaded image.
   * Predict the DR stage.
   * Show the segmented image.
   * Speak out the prediction result.
4. The list of predictions will appear at the end with an audio playback option.

---

## 📊 Model Training

Use `train_model.py` to:

* Load and preprocess dataset images.
* Build and train the DenseNet121 model.
* Evaluate the model’s performance.
* Save the trained model as `model.h5`.

Example training command:

```bash
python train_model.py
```

---

## 🔍 Evaluation Metrics

* **Accuracy**
* **Loss**
* **Confusion Matrix**
* **Classification Report (Precision, Recall, F1-score)**

---

## 🎨 Features

✅ Deep learning model with DenseNet121
✅ Streamlit-based interactive UI
✅ Image segmentation using OpenCV
✅ Voice output using gTTS
✅ Multi-image upload and prediction
✅ Easy to train and deploy

---


<img width="339" height="295" alt="Screenshot 2025-04-12 163628" src="https://github.com/user-attachments/assets/e21b5935-9e43-4a64-b6a0-d103544bad53" />


---

## 📚 References

* [DenseNet121 Paper](https://arxiv.org/abs/1608.06993)
* [Kaggle: Diabetic Retinopathy Dataset](https://www.kaggle.com/c/diabetic-retinopathy-detection)
* [Streamlit Documentation](https://docs.streamlit.io/)
* [TensorFlow Keras Docs](https://www.tensorflow.org/guide/keras)

---

## 👩‍💻 Author

**Saikumar Vallipanni**
📧 saikumarvallipanni@gmail.com
💼 www.linkedin.com/in/saikumar-vallipanni-34ba5a298

---

## 🏁 Future Enhancements

* Support for Grad-CAM visualizations
* Web-based data collection
* Integration with cloud storage (AWS / GCP)
* Deployment on Streamlit Cloud or Heroku


