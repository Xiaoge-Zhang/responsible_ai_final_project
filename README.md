# Responsible AI Final Project: COVID-19 Mortality Prediction and Fairness Evaluation

This project aims to evaluate various machine learning (ML) models and their fairness in predicting COVID-19 mortality. The dataset used is from the CDC's COVID-19 Case Surveillance Public Use Data, which contains detailed information on COVID-19 cases in the U.S.

## Dataset
The dataset is available at:  
[CDC COVID-19 Case Surveillance Public Use Data](https://data.cdc.gov/Case-Surveillance/COVID-19-Case-Surveillance-Public-Use-Data-with-Ge/n8mc-b4w4/about_data)

This dataset contains 19 fields with data on COVID-19 cases, including demographics, geography (county and state of residence), exposure history, disease severity indicators, outcomes, and the presence of underlying medical conditions and risk behaviors.

## Project Structure
- **Data Folder**: The processed dataset is stored in the `data` folder.
- **Python Scripts**:
  - `data_preprocessing.py`: This script processes the raw data and creates datasets by attributes and labels. It stores the processed data in the `data` folder.
  - `SVM.py`: A basic machine learning model that predicts whether a patient will die from COVID-19 based on various features.
  - `model_name.py`: these scripts run the corresponding ML model (e.g., SVM.py) and evaluates its fairness by considering sex, race, or both as protected attributes. It also evaluates the model after applying three different fairness mitigation methods. The results are printed out and saved in the `output` folder.
  - `result_visualization.py`: This script generates plots for the results presented in the report. The figures are saved as PDF files in the `visualization` folder.

## Tools and Libraries Used
This project uses the following libraries:
- **Scikit-learn**: For ready-to-use machine learning models and evaluation tools.
- **PyTorch**: For potential future use of multi-layer perceptron (MLP) models.
- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation and analysis.
- **SciPy**: For statistical tests (to be used when race data is available in the dataset).

## Environment Setup
To set up the environment and install the necessary packages, run the following command:

```bash
pip install numpy pandas scikit-learn sodapy aif360
