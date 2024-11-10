# responsible_ai_final_project
For this project, I am going to test different ML models and their fairness on the fatality prediction for the COVID-19 virus infection
the data is from https://github.com/beoutbreakprepared/nCoV2019 which contains cases and their outcomes around the world since the beginning of the outbreak.
the processed data is stored in the data folder, and data_preprocessing in the python_src folder processes the raw data (raw data is too big to put in Git Hub).
the SVM.py is a basic predictor that predicts whether a patient will die from the COVID-19 pandemic based on features. 

# tools I used
The tools I used are mainly the Scikit-learn library for some ready-to-use ML models. I will also use Pytorch for the MLP models in the future. Other libraries I used are Numpy and Pandas. I will also use the Scipy library for p-tests once I acquire the CDC datasets containing race data. (it is very big (103M rows) and I am still figuring out a way to process them by batch through API)

# How to run the code
simply run SVM.py will output the accuracy precision and recall of SVM on the preprocessed dataset. also, it will save the details of testing samples, prediction results and true labels in the form of CSV in the output folder. 
