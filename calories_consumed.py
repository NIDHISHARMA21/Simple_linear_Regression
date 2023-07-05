# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

calories_consumed = pd.read_csv(r"H:\EDUCATION\DS_ML_AI\DATA SCIENCE\21. simple linear regression/calories_consumed.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

calories_consumed.describe()

#UNIVARIAT ANALYSIS

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

plt.bar(height = calories_consumed.X, x = np.arange(1, 15, 1))
plt.hist(calories_consumed.X) #histogram
plt.boxplot(calories_consumed.X) #boxplot

plt.bar(height = calories_consumed.Y, x = np.arange(1, 15, 1))
plt.hist(calories_consumed.Y) #histogram
plt.boxplot(calories_consumed.Y) #boxplot

#BIVARIATE ANALYSIS
# Scatter plot
plt.scatter(x = calories_consumed['X'], y = calories_consumed['Y'], color = 'green') 

# correlation
np.corrcoef(calories_consumed.X, calories_consumed.Y) 

# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

# Covariance
cov_output = np.cov(calories_consumed.X, calories_consumed.Y)[0, 1]
cov_output

# wcat.cov()

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('Y ~ X', data = calories_consumed).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(calories_consumed['X']))

# Regression Line
plt.scatter(calories_consumed.X, calories_consumed.Y)
plt.plot(calories_consumed.X, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = calories_consumed.Y - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(X); y = Y

plt.scatter(x = np.log(calories_consumed['X']), y = calories_consumed['Y'], color = 'brown')
np.corrcoef(np.log(calories_consumed.X), calories_consumed.Y) #correlation

model2 = smf.ols('Y ~ np.log(X)', data = calories_consumed).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(calories_consumed['X']))

# Regression Line
plt.scatter(np.log(calories_consumed.X), calories_consumed.Y)
plt.plot(np.log(calories_consumed.X), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = calories_consumed.Y - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = X; y = log(Y)

plt.scatter(x = calories_consumed['X'], y = np.log(calories_consumed['Y']), color = 'orange')
np.corrcoef(calories_consumed.X, np.log(calories_consumed.Y)) #correlation

model3 = smf.ols('np.log(Y) ~ X', data = calories_consumed).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(calories_consumed['X']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(calories_consumed.X, np.log(calories_consumed.Y))
plt.plot(calories_consumed.X, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = calories_consumed.Y - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = X; x^2 = X*X; y = log(Y)

model4 = smf.ols('np.log(Y) ~ X + I(X*X)', data = calories_consumed).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(calories_consumed))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = calories_consumed.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = wcat.iloc[:, 1].values


plt.scatter(calories_consumed.X, np.log(calories_consumed.Y))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = calories_consumed.Y - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(calories_consumed, test_size = 0.2)

finalmodel = smf.ols('np.log(Y) ~ X + I(X*X)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_AT = np.exp(test_pred)
pred_test_AT

# Model Evaluation on Test data
test_res = test.Y - pred_test_AT
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_AT = np.exp(train_pred)
pred_train_AT

# Model Evaluation on train data
train_res = train.Y - pred_train_AT
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse
