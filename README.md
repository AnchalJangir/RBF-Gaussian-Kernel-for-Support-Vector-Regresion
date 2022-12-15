# RBF-Gaussian-Kernel-for-Support-Vector-Regresion
Creating a RBF kernel code for a Support Vector Regression Problem for a concrete data set.
Please refer the attached "concrete data set" file.
Importing all the libraries we need.
First read the data excel file using "pd.read_excel" and check if there are any missing values if any then fill the missing values by their column means respectively.
furthermore normalizing the data importing "MinMaxScaler".
Separating input data X and y.
Splitting the data into training and testing data set in the ratio of 8:2 respectively.
Defining a function for RBF kernel for a support vector regression and returning 4 types of error which help us to decide the hyper-parameter values. Errors are calculated using sklearn library.
For choosing the hyperparameter we use k-fold cross-validation and grid search(here I used 5-fold cross-validation). In polynomial kernel our hyper-parameter C, gamma are in the range (2^(-10), 2^(10)) and (2^(-10), 2^(10)) repectively. Getting the value of C and gamma for minimum MSE error.
Then apply the gamma kernel for testing dataset using optimum hyperparameter C, gamma, and calculating MSE, RMSE, MAE, R2 error.
