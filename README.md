# how to set up the envrionment
to be able to run the programs, you need to use the following command to install the packages used:
pip install numpy, pandas, scikit-learn, sodapy, aif360

# responsible_ai_final_project
For this project, I am going to test different ML models and their fairness on the fatality prediction for the COVID-19 virus infection
the data is from hhttps://data.cdc.gov/Case-Surveillance/COVID-19-Case-Surveillance-Public-Use-Data-with-Ge/n8mc-b4w4/about_data 
This case surveillance public use dataset has 19 elements for all COVID-19 cases shared with CDC and includes demographics, geography (county and state of residence), any exposure history, disease severity indicators and outcomes, and presence of any underlying medical conditions and risk behaviors.
the processed data is stored in the data folder, and data_preprocessing in the python_src folder processes the raw data
the SVM.py is a basic predictor that predicts whether a patient will die from the COVID-19 pandemic based on features. 

# tools I used
The tools I used are mainly the Scikit-learn library for some ready-to-use ML models. I will also use Pytorch for the MLP models in the future. Other libraries I used are Numpy and Pandas. I will also use the Scipy library for p-tests once I acquire the CDC datasets containing race data. (it is very big (103M rows) and I am still figuring out a way to process them by batch through API)

# How to run the code
there are no parameters at this time
running data_gathering.py will gather the data from CDC dataset with tuned query
running data_preprocessing will create datasets by race of the patients
running Logi{{model name}.py will use the corresponding model to predict given the data, using sex and race as protected attributes. it will output the accuracy of the model on unmitigated data, output disparate impact ratio on sex and race, the average odds error on both attributes, and mitigate the bias if there's by feature reweighting. It will also output the mitigated performance as well as disparate impact ratio and average odds error.

# Future work
apparently right now the project isn't finished, I will have bigger datasets and test more ML models by the deadlines
