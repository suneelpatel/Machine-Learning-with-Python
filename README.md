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
Classification is technique to categorize our data into a desired and distinct number of classes where we can assign label to each class. It can be performed on structured or unstructured data.

### Basic Terminology in Classification Algorithms
* **Classifier:** An algorithm that maps the input data to a specific category.
* **Classification model:** A classification model tries to draw some conclusion from the input values given for training. It will predict the class labels/categories for the new data.
* **Feature:** A feature is an individual measurable property of a phenomenon being observed.
* **Binary Classification:** Classification task with two possible outcomes. Eg: Gender classification (Male / Female)
* **Multi-class classification:** Classification with more than two classes. In multi-class classification, each sample is assigned to one and only one target label. Eg: An animal can be a cat or dog but not both at the same time. 
* **Multi-label classification:** Classification task where each sample is mapped to a set of target labels (more than one class). Eg: A news article can be about sports, a person, and location at the same time.

### Applications of Classification Algorithms
* Email spam classification
* Bank customers loan pay willingness prediction.
* Cancer tumor cells identification.
* Sentiment analysis 
* Drugs classification
* Facial key points detection
* Pedestrians detection in an automotive car driving.

### Types of Classification Algorithms
Classification Algorithms could be broadly classified as the following:

* Linear Classifiers
      - Logistic regression
      - Naive Bayes classifier
* Support vector machines
      - Least squares support vector machines
* Quadratic classifiers
* Kernel estimation
      - k-nearest neighbor 
* Decision trees
* Random forests
* Neural networks
* Learning vector quantization

# 4. Clustering
** Clustering** is a Machine Learning technique that involves the grouping of data points.

Clustering is the task of dividing the population or data points into a number of groups such that data points in the same groups are more similar to other data points in the same group than those in other groups. In simple words, the aim is to segregate groups with similar traits and assign them into clusters.

Clustering is an unsupervised machine learning approach, but can it be used to improve the accuracy of supervised machine learning algorithms as well by clustering the data points into similar groups and using these cluster labels as independent variables in the supervised machine learning algorithm.

### Types of Clustering
Broadly speaking, clustering can be divided into two subgroups :

* **Hard Clustering:** In hard clustering, each data point either belongs to a cluster completely or not. For example, in the above example each customer is put into one group out of the 10 groups.

* **Soft Clustering:** In soft clustering, instead of putting each data point into a separate cluster, a probability or likelihood of that data point to be in those clusters is assigned. For example, from the above scenario each costumer is assigned a probability to be in either of 10 clusters of the retail store.

### Classification of Clustering Algorithm:

1. Exclusive Clustering 
2. Overlapping Clustering
3. Hierarchical Clustering

### K-Means Clustering
K-Means algorithms are extremely easy to implement and very efficient computationally speaking. Those are the main reasons that explain why they are so popular. But they are not very good to identify classes when dealing with in groups that do not have a spherical distribution shape.

The K-Means algorithms aims to find and group in classes the data points that have high similarity between them. In the terms of the algorithm, this similiarity is understood as the opposite of the distance between datapoints. The closer the data points are, the more similar and more likely to belong to the same cluster they will be.

### Algorithm Steps
* First, we need to choose k, the number of clusters that we want to be finded.
* Then, the algorithm will select randomly the the centroids of each cluster.
* It will be assigned each datapoint to the closest centroid (using euclidean distance).
* It will be computed the cluster inertia.
* The new centroids will be calculated as the mean of the points that belong to the centroid of the previous step. In other words, by calculating the minimum quadratic error of the datapoints to the center of each cluster, moving the center towards that point
* Back to step 3.


# 5. SciKit Learn


# 6. Supervised Learning


# 7. Dimensionality Reduction


# 8. Unsupervised Learning


# 9. Association Rules Mining and Recommendation Systems


# 10. Reinforcement Learning


# 11. Time Series Analysis


# 12. Model Selection and Boosting
