import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split

# Step 1
data = pd.read_csv('Project_1_Data.csv')
df = pd.DataFrame(data)
dfnew = df.to_numpy()
print(dfnew)

# Step 2
X, Y, Z, Step = dfnew.T
summary_stats = data.describe()
print(summary_stats)
X_train, X_test, Y_train, Y_test, Z_train, Z_test, Step_train, Step_test = train_test_split(X, Y, Z, Step, test_size=0.2, random_state=0)

#plt.plot(Step_train, X_train, label='X')
#plt.plot(Step_train, Y_train, label='Y')
#plt.plot(Step_train, Z_train, label='Z')
#plt.title('Coordinates vs. Steps')
#plt.ylabel('Coordinates')
#plt.xlabel('Steps')
#plt.legend()
#plt.show()

plt.scatter(Step_train, X_train, label='X')
plt.scatter(Step_train, Y_train, label='Y')
plt.scatter(Step_train, Z_train, label='Z')
plt.title('Coordinates vs. Steps')
plt.ylabel('Coordinates')
plt.xlabel('Steps')
plt.legend()
plt.show()

sns.displot(X_train)
plt.show()
sns.displot(Y_train)
plt.show()
sns.displot(Z_train)
plt.show()

# Step 3
corr = df.corr(method='spearman')
sns.heatmap(corr, vmin=-1, vmax=1, annot=True)

#Step 4
#Logistic Regression, Polynomial Regression, Random forest, Decision tree
#Decision tree
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
features = ['X', 'Y', 'Z']
X = df[features]
y = df['Step']
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
tree.plot_tree(dtree, feature_names=features)

#Logistic regression
from sklearn import linear_model
logr = linear_model.LogisticRegression(max_iter=10000, solver='lbfgs')
Xtrain = X_train.reshape(-1, 1)
Ytrain = Y_train.reshape(-1, 1)
Ztrain = Z_train.reshape(-1, 1)
features1 = np.column_stack((Xtrain,Ytrain,Ztrain))
gridLR = [{'multi_class':['multinomial', 'ovr'], 'C':[1, 10, 20 ,50]}]
clfLR = GridSearchCV(estimator = logr,
                     param_grid = gridLR,
                     scoring = 'accuracy',
                     cv = 5,)
clfLR.fit(features1, Step_train)
print(clfLR.best_estimator_)

#Random forest
# Define the Random Forest model
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
# Define the hyperparameter grid
param_grid = {
    'n_estimators': [5, 10, 15],  # Number of trees
    'max_depth': [None, 10, 20],     # Max depth of the tree
    'min_samples_split': [2, 5, 10], # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4]    # Minimum samples at a leaf node
}
clfRF = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)
clfRF.fit(features1, Step_train)
print(clfRF.best_estimator_)

#Random forest with RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
Randomrf = RandomizedSearchCV(estimator = rf_model, param_distributions = param_grid, cv = 5)
Randomrf.fit(features1, Step_train)
print(Randomrf.best_estimator_)