# Machine Learning with Python
Machine learning is changing the world and if you want to be a part of the ML revolution, this is a great place to start! This repository serves as an excellent introduction to implementing machine learning algorithms in depth such as linear and logistic regression, decision tree, random forest, SVM, Naive Bayes, KNN, K-Mean Cluster, PCA, Time Series Analysis and so on.

# Table of Content
1. Machine Learning - Introduction
2. Regression
3. Classfication
4. Clustering
5. SciKit Learn
6. Supervised Learning
7. Dimensionality Reduction
8. Unsupervised Learning
9. Association Rules Mining and Recommendation Systems
10. Reinforcement Learning
11. Time Series Analysis
12. Model Selection and Boosting

# 1. Machine Learning - Introduction
### What is Machine Learning (ML)?

Machine Learning (ML) is the science of getting machine to act without being explicitly programmed.

Machine learning is a type of artificial intelligence (AI) that allows software applications to learn from the data and become more accurate in predicting outcomes without human intervention.

Machine Learning is a subset of artificial intelligence (AI) which focuses mainly on machine learning from their experience and making predictions based on its experience.

Machine Learning is an application of artificial intelligence (AI) which provides systems the ability to automatically learn and improve from experience without being explicitly programmed.

### Top 10 real-life examples of Machine Learning

1. Face Detection or Image Recognition
2. Voice or Speech Recognition
3. Stock Trading or Statistical Arbitrage
4. Medical Diagnosis
5. Spam Detection
6. Learning Association
7. Classification
8. Customer Segmentation
9. Financial Services
10. Fraud Detection

### What does it do? 

It enables the computers or the machines to make data-driven decisions rather than being explicitly programmed for carrying out a certain task. These programs or algorithms are designed in a way that they learn and improve over time when are exposed to new data.

### How does Machine Learning Work?

Machine Learning algorithm is trained using a training data set to create a model. When new input data is introduced to the ML algorithm, it makes a prediction on the basis of the model.

The prediction is evaluated for accuracy and if the accuracy is acceptable, the Machine Learning algorithm is deployed. If the accuracy is not acceptable, the Machine Learning algorithm is trained again and again with an augmented training data set.

### Types of Machine Learning
1. Supervised Learning (Follow Task Driven Approach - Predict Next Value)
2. Unsupervised Learning (Follow Data Driven Approach - Identify Cluster)
3. Reinforcement Learning(Learn From Experience)



# 2. Regression
In statistical modeling, regression analysis is a set of statistical processes for estimating the relationships among variables. 

### What is Regression Analysis?

Regression analysis is a form of predictive modelling technique which investigates the relationship between a dependent (target) and independent variable (s) (predictor). This technique is used for forecasting, time series modelling and finding the causal effect relationship between the variables. For example, relationship between rash driving and number of road accidents by a driver is best studied through regression.

Regression analysis is an important tool for modelling and analyzing data.

### Why do we use Regression Analysis?

There are multiple benefits of using regression analysis. They are as follows:

It indicates the significant relationships between dependent variable and independent variable.
It indicates the strength of impact of multiple independent variables on a dependent variable.

### How many types of regression techniques do we have?

There are various kinds of regression techniques available to make predictions. These techniques are mostly driven by three metrics: 
1. Number of independent variables, 
2. Type of dependent variables and 
3. Shape of regression line 

### Types of Regression
•	Linear Regression
•	Logistic Regression
•	Polynomial Regression
•	Stepwise Regression


### 1. Linear Regression
A linear regression is one of the easiest statistical models in machine learning. It is used to show the linear relationship between a dependent variable and one or more independent variables.

In this technique, the dependent variable is continuous, independent variable(s) can be continuous or discrete, and nature of regression line is linear.

It is represented by an equation Y=a+b*X + e, where a is intercept, b is slope of the line and e is error term. This equation can be used to predict the value of target variable based on given predictor variable(s).

##### Important Points:
* There must be linear relationship between independent and dependent variables
* Multiple regression suffers from multicollinearity, autocorrelation, heteroskedasticity.
* Linear Regression is very sensitive to Outliers. It can terribly affect the regression line and eventually the forecasted values.
* Multicollinearity can increase the variance of the coefficient estimates and make the estimates very sensitive to minor changes in the model. The result is that the coefficient estimates are unstable
* In case of multiple independent variables, we can go with forward selection, backward elimination and step wise approach for selection of most significant independent variables.

### 2. Logistic Regression

Logistic regression is used to find the probability of event=Success and event=Failure. We should use logistic regression when the dependent variable is binary (0/ 1, True/ False, Yes/ No) in nature. Here the value of Y ranges from 0 to 1 and it can represented by following equation.

#### Important Points:
* It is widely used for classification problems
* Logistic regression doesn’t require linear relationship between dependent and independent variables.  It can handle various types of relationships because it applies a non-linear log transformation to the predicted odds ratio
* To avoid over fitting and under fitting, we should include all significant variables. A good approach to ensure this practice is to use a step wise method to estimate the logistic regression
* It requires large sample sizes because maximum likelihood estimates are less powerful at low sample sizes than ordinary least square
* The independent variables should not be correlated with each other i.e. no multi collinearity.  However, we have the options to include interaction effects of categorical variables in the analysis and in the model.
* If the values of dependent variable is ordinal, then it is called as Ordinal logistic regression
* If dependent variable is multi class then it is known as Multinomial Logistic regression.

### 3. Polynomial Regression
A regression equation is a polynomial regression equation if the power of independent variable is more than 1. The equation below represents a polynomial equation:
      
              y=a+b*x^2
              
In this regression technique, the best fit line is not a straight line. It is rather a curve that fits into the data points.


##### Important Points:
* While there might be a temptation to fit a higher degree polynomial to get lower error, this can result in over-fitting. Always plot the relationships to see the fit and focus on making sure that the curve fits the nature of the problem. Here is an example of how plotting can help:
*Especially look out for curve towards the ends and see whether those shapes and trends make sense. Higher polynomials can end up producing wierd results on extrapolation.


### 4. Stepwise Regression
This form of regression is used when we deal with multiple independent variables. In this technique, the selection of independent variables is done with the help of an automatic process, which involves no human intervention.

This feat is achieved by observing statistical values like R-square, t-stats and AIC metric to discern significant variables. Stepwise regression basically fits the regression model by adding/dropping co-variates one at a time based on a specified criterion. Some of the most commonly used Stepwise regression methods are listed below:

* Standard stepwise regression does two things. It adds and removes predictors as needed for each step.
* Forward selection starts with most significant predictor in the model and adds variable for each step.
* Backward elimination starts with all predictors in the model and removes the least significant variable for each step.

The aim of this modeling technique is to maximize the prediction power with minimum number of predictor variables. It is one of the method to handle higher dimensionality of data set.

# 3. Classfication


# 4. Clustering


# 5. SciKit Learn


# 6. Supervised Learning


# 7. Dimensionality Reduction


# 8. Unsupervised Learning


# 9. Association Rules Mining and Recommendation Systems


# 10. Reinforcement Learning


# 11. Time Series Analysis


# 12. Model Selection and Boosting
