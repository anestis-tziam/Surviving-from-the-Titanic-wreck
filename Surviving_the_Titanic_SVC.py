###################################################################################################################
###################################################################################################################
######################################### Titanic: Dead or Alive? ################################################
######################################## Applying a SVM algorithm ################################################
###################################################################################################################


# First import the necessary libraries
import numpy as np
import pandas as pd
#from sklearn import preprocessing, cross_validation
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score


# Upload the dataset
df = pd.read_csv("titanic_clean.csv")


# You should perform the following tasks for having a detailed view on the dataset.


# Print some lines of the dataset. This will display the column names and the 
# corresponding data, thus it will help you understand the dataset. By choosing
# df.head(number), where number is an integer that will print the first "n" rows of 
# the dataset.
print(df.head())


# Check the datatypes of each column
print(df.dtypes)


###############################################################################################################
###############################################################################################################
################################### Remove columns you don't need #############################################
##############################################################################################################
##############################################################################################################


# In every dataset there are columns that are not important for a given task. For columns like 
# Unnamed 0 or name do not have an effect. Thus, such columns can be removed.

df.drop('Unnamed: 0', axis = 1, inplace = True)
df.drop('name', axis = 1, inplace = True)         
df.drop('cabin', axis = 1, inplace = True)
df.drop('embarked', axis = 1, inplace = True)
df.drop('body', axis = 1, inplace = True)
df.drop('home.dest', axis = 1, inplace = True)
df.drop('has_cabin_number', axis = 1, inplace = True)
df.drop('ticket', axis = 1, inplace = True)


# Check in which rows there are missing values and how many they are.
print(df.isnull().sum())

############################################################################################################
############################################################################################################
########################################### Edit the boat colum ############################################
############################################################################################################
############################################################################################################

# If you check the boat column you will see that there are a lot of values missing. This means that there 
# was no boat-space for these passengers (explaining the missing value) which was true in the case of 
# Titanic. The column is in string format and the feature boat has a number as value, or a letter. For 
# certain passengers more than one value appears. We can edit the boat column and set missings values to 
# zero for no boat, and to 1 for any other boat.


# Fill every missing value in the boat column with zero.
df['boat'] = df['boat'].fillna(str(0))


# Edit the boat column, and set every value non-equal to 0 to 1.
mask = df.boat != '0'
column_name = 'boat'
df.loc[mask, column_name] = '1'   

      
# Remove the missing values
df.dropna(axis = 0, inplace = True)


X = df
y = X[X.columns[1]].copy()
X.drop(X.columns[1], axis = 1,inplace = True)


# Encoding categorical data
# Encoding the Independent Variable
X = pd.concat([X, pd.get_dummies(X['pclass'],prefix = 'pclass',prefix_sep = ':')], axis = 1)
X.drop('pclass',axis = 1,inplace = True)

X = pd.concat([X, pd.get_dummies(X['sex'],prefix = 'sex',prefix_sep = ':')], axis = 1)
X.drop('sex',axis = 1,inplace = True)

X = pd.concat([X, pd.get_dummies(X['sibsp'],prefix = 'sibsp',prefix_sep = ':')], axis = 1)
X.drop('sibsp',axis = 1,inplace = True)

X = pd.concat([X, pd.get_dummies(X['parch'],prefix = 'parch',prefix_sep = ':')], axis = 1)
X.drop('parch',axis = 1,inplace = True)

X = pd.concat([X, pd.get_dummies(X['boat'],prefix = 'boat',prefix_sep = ':')], axis = 1)
X.drop('boat',axis = 1,inplace = True)


# Split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
y_pred_SVC = svc.predict(X_test)


# Results 

# Get the confusion matrix
cm_SVC = confusion_matrix(y_test, y_pred_SVC)


# Get the classification report
CL_report_SVC = classification_report(y_test, y_pred_SVC)
print(classification_report(y_test, y_pred_SVC))


# Get the accuracy of the SVM algorithm
accur_SVM = accuracy_score(y_test, y_pred_SVC)
print("The Accuracy for SVM is {}".format(accur_SVM))


# Below I present an example of how you can tune your code and calculate the optimal parameters


# Applying grid-search to find the best model and parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'kernel': ['linear'], 'C': [1,2,3] },
        {'C' : [1,2,3,4], 'kernel': ['rbf'], 'gamma' : [0.005,0.01,0.05,0.1]}]
grid_search = GridSearchCV(estimator = svc, 
                           param_grid = parameters, 
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)

# Get the best accuracy and the best parameters
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_