# Model Training Project

This project contains a Jupyter Notebook for training a machine learning model. The goal is to preprocess data, train a predictive model,
and evaluate its performance using standard metrics.

## 📂 Files

- `Model_training.ipynb`: The main notebook that contains data loading, preprocessing, model training, and evaluation steps.
- `app.py`: A streamlit based web app for creating posts based on the fine tuned distilgpt-2 model.
- `final.csv`: The dataset used for fine-tuning the model. The dataset contains posts from LinkedIn accounts.
- `README.md`: Project overview and usage instructions.

## 📊 Project Structure

The notebook includes the following steps:

1. **Import Libraries**
2. **Load Dataset**
3. **Data Cleaning & Preprocessing**
4. **Feature Engineering**
5. **Model Selection & Training**
6. **Model Evaluation**
7. **Results & Conclusions**

## 🤖 Model Description

> The model used is distilgpt-2 with 82M paramenters. Using the training dataset the model was fine tuned for LinkedIn post generation.

## 🧪 Requirements
> The necessary libraries are mentioned in requirements.txt file.

## 📊 Running the model
> To use the model:
> 1. Run the Model_training.ipynb file using GPU and save the fined tuned model in the local directory.
> 2. Place the app.py file in the same directory as the fine tuned model folder.
> 3. Run the app.py file using 'streamlit run app.py'
