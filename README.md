# responsible_ai_final_project
For this project, I am going to test different ML models and its fairness on the fatality prediction for the COVID-19 virus infection
the data is from https://github.com/beoutbreakprepared/nCoV2019 which contains cases and their outcomes around the world since the beginning of the outbreak.
the processed data is stored in the data folder, and data_preprocessing in the python_src folder processes the raw data (raw data is too big to put in github).
the SVM.py is a basic predictor that predicts weather a patient will die from the covid-19 pandemic based on features. 

# how to run the code
simply run SVM.py will output the accuracy precision and recall of SVM on the preprocessed dataset. also it will save the details of testing samples, prediction results and true labels in the form of CSV in the output folder. 
