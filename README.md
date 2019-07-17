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
### What is scikit-learn?
Scikit-learn provide a range of supervised and unsupervised learning algorithms via a consistent interface in Python.
It is licensed under a permissive simplified BSD license and is distributed under many Linux distributions, encouraging academic and commercial use.

The library is built upon the SciPy (Scientific Python) that must be installed before you can use scikit-learn. This stack that includes:
* **NumPy:** Base n-dimensional array package
* **SciPy:** Fundamental library for scientific computing
* **Matplotlib:** Comprehensive 2D/3D plotting
* **IPython:** Enhanced interactive console
* **Sympy:** Symbolic mathematics
* **Pandas:** Data structures and analysis

Extensions or modules for SciPy care conventionally named SciKits. As such, the module provides learning algorithms and is named scikit-learn.

The vision for the library is a level of robustness and support required for use in production systems. This means a deep focus on concerns such as easy of use, code quality, collaboration, documentation and performance.
 
### What are the features?
The library is focused on modelling data. It is not focused on loading, manipulating and summarizing data. For these features, refer to NumPy and Pandas.

Some popular groups of models provided by scikit-learn include:
* **Clustering:** for grouping unlabeled data such as KMeans.
* **Cross Validation:** for estimating the performance of supervised models on unseen data.
* **Datasets:** for test datasets and for generating datasets with specific properties for investigating model behavior.
* **Dimensionality Reduction:** for reducing the number of attributes in data for summarization, visualization and feature selection such as Principal component analysis.
* **Ensemble Methods:** for combining the predictions of multiple supervised models.
* **Feature Extraction:** for defining attributes in image and text data.
* **Feature Selection:** for identifying meaningful attributes from which to create supervised models.
* **Parameter Tuning:** for getting the most out of supervised models.
* **Manifold Learning:** For summarizing and depicting complex multi-dimensional data.
* **Supervised Models:** a vast array not limited to generalized linear models, discriminate analysis, naive bayes, lazy methods, neural networks, support vector machines and decision trees.

### Who is using it?
* The scikit-learn testimonials page lists Inria, Mendeley, wise.io , Evernote, Telecom ParisTech and AWeber as users of the library.
* If this is a small indication of companies that have presented on their use, then there are very likely tens to hundreds of larger organizations using the library.
* It has good test coverage and managed releases and is suitable for prototype and production projects alike.


# 6. Supervised Learning
### What is Supervised Learning?
Supervised Learning is the one, where you can consider the learning is guided by a teacher. We have a dataset which acts as a teacher and its role is to train the model or the machine. Once the model gets trained it can start making a prediction or decision when new data is given to it.



# 7. Dimensionality Reduction

Dimensionality reduction is the process of reducing the number of random variables under consideration, by obtaining a set of principal variables. It can be divided into feature selection and feature extraction.

#### Dimensionality reduction can be done in two different ways:
* By only keeping the most relevant variables from the original dataset (this technique is called feature selection)
* By finding a smaller set of new variables, each being a combination of the input variables, containing basically the same information as the input variables (this technique is called dimensionality reduction)

We will now look at various dimensionality reduction techniques and how to implement each of them in Python.

### Factor Analysis
Suppose we have two variables: Income and Education. These variables will potentially have a high correlation as people with a higher education level tend to have significantly higher income, and vice versa.

In the Factor Analysis technique, variables are grouped by their correlations, i.e., all variables in a particular group will have a high correlation among themselves, but a low correlation with variables of other group(s). Here, each group is known as a factor. These factors are small in number as compared to the original dimensions of the data. However, these factors are difficult to observe.


### Principal Component Analysis (PCA)
PCA is a technique which helps us in extracting a new set of variables from an existing large set of variables. These newly extracted variables are called Principal Components. You can refer to this article to learn more about PCA. For your quick reference, below are some of the key points you should know about PCA before proceeding further:
* A principal component is a linear combination of the original variables
* Principal components are extracted in such a way that the first principal component explains maximum variance in the dataset
* Second principal component tries to explain the remaining variance in the dataset and is uncorrelated to the first principal component
* Third principal component tries to explain the variance which is not explained by the first two principal components and so on

### What is principal component analysis?

Principal component analysis (PCA) is used to summarize the information in a data set described by multiple variables.
Note that, the information in a data is the total variation it contains.
PCA reduces the dimensionality of data containing a large set of variables. This is achieved by transforming the initial variables into a new small set of variables without loosing the most important information in the original data set.
These new variables corresponds to a linear combination of the originals and are called principal components.

#### Main purpose of PCA
The main goals of principal component analysis is :
* to identify hidden pattern in a data set
* to reduce the dimensionnality of the data by removing the noise and redundancy in the data
* to identify correlated variables
PCA method is particularly useful when the variables within the data set are highly correlated.
Correlation indicates that there is redundancy in the data. Due to this redundancy, PCA can be used to reduce the original variables into a smaller number of new variables ( = principal components) explaining most of the variance in the original variables.

#### Points to Remember
1. PCA is used to overcome features redundancy in a data set.
2. These features are low dimensional in nature.
3. These features a.k.a components are a resultant of normalized linear combination of original predictor variables.
4. These components aim to capture as much information as possible with high explained variance.
5. The first component has the highest variance followed by second, third and so on.
6. The components must be uncorrelated (remember orthogonal direction ? ). See above.
7. Normalizing data becomes extremely important when the predictors are measured in different units.
8. PCA works best on data set having 3 or higher dimensions. Because, with higher dimensions, it becomes increasingly difficult to make interpretations from the resultant cloud of data.
9. PCA is applied on a data set with numeric variables.
10. PCA is a tool which helps to produce better visualizations of high dimensional data.


# 8. Unsupervised Learning
Unsupervised Learning is a class of Machine learning techniques to find the patterns in data. The data given to unsupervised algorithm are not labelled, which means only the input variables(X) are given with no corresponding output variables. In unsupervised learning, the algorithms are left to themselves to discover interesting structures in the data.

### What is Unsupervised Learning?
The model learns through observation and finds structures in the data. Once the model is given a dataset, it automatically finds patterns and relationships in the dataset by creating clusters in it. What it cannot do is add labels to the cluster, like it cannot say this a group of apples or mangoes, but it will separate all the apples from mangoes.
Suppose we presented images of apples, bananas and mangoes to the model, so what it does, based on some patterns and relationships it creates clusters and divides the dataset into those clusters. Now if a new data is fed to the model, it adds it to one of the created clusters.

Well, this category of machine learning is known as unsupervised because unlike supervised learning there is no teacher. Algorithms are left on their own to discover and return the interesting structure in the data.
The goal for unsupervised learning is to model the underlying structure or distribution in the data in order to learn more about the data.

#### Let me rephrase it for you in simple terms:
In the unsupervised learning approach, the sample of a training dataset does not have an expected output associated with them. Using the unsupervised learning algorithms you can detect patterns based on the typical characteristics of the input data. Clustering can be considered as an example of a machine learning task that uses the unsupervised learning approach. The machine then groups similar data samples and identify different clusters within the data.

#### Example: 
Fraud Detection is probably the most popular use-case of Unsupervised Learning. Utilizing past historical data on fraudulent claims, it is possible to isolate new claims based on its proximity to clusters that indicate fraudulent patterns.


# 9. Association Rules Mining and Recommendation Systems
**Association rule learning** is a rule-based machine learning method for discovering interesting relations between variables in large databases.

### What is the Apriori Algorithm?

Apriori algorithm, a classic algorithm, is useful in mining frequent itemsets and relevant association rules. Usually, you operate this algorithm on a database containing a large number of transactions. One such example is the items customers buy at a supermarket.
It helps the customers buy their items with ease, and enhances the sales performance of the departmental store.
This algorithm has utility in the field of healthcare as it can help in detecting adverse drug reactions (ADR) by producing association rules to indicate the combination of medications and patient characteristics that could lead to ADRs.

#### Apriori Algorithm – An Odd Name
It has got this odd name because it uses ‘prior’ knowledge of frequent itemset properties. The credit for introducing this algorithm goes to Rakesh Agrawal and Ramakrishnan Srikant in 1994. We shall now explore the apriori algorithm implementation in detail.
Apriori algorithm – The Theory

#### Three significant components comprise the apriori algorithm. They are as follows.
* Support
* Confidence
* Lift

#### Support
Support is the default popularity of any item. You calculate the Support as a quotient of the division of the number of transactions containing that item by the total number of transactions. Hence, in our example,
Support (Jam) = (Transactions involving jam) / (Total Transactions)
                        = 200/2000 = 10%
#### Confidence
In our example, Confidence is the likelihood that customer bought both bread and jam. Dividing the number of transactions that include both bread and jam by the total number of transactions will give the Confidence figure.
Confidence = (Transactions involving both bread and jam) / (Total Transactions involving jam)
                    = 100/200 = 50%
It implies that 50% of customers who bought jam bought bread as well.

#### Lift
According to our example, Lift is the increase in the ratio of the sale of bread when you sell jam. The mathematical formula of Lift is as follows.
Lift = (Confidence (Jam͢͢ – Bread)) / (Support (Jam))
      = 50 / 10 = 5
It says that the likelihood of a customer buying both jam and bread together is 5 times more than the chance of purchasing jam alone. If the Lift value is less than 1, it entails that the customers are unlikely to buy both the items together. Greater the value, the better is the combination.

### What is Recommendation Engine?
A recommendation engine filters the data using different algorithms and recommends the most relevant items to users. It first captures the past behaviour of a customer and based on that, recommends products which the users might be likely to buy. The analysis uses the user details like.
* Gender,
* Age,
* Geographical location,
* Online searches,
* Previous purchase or items user interested.

Before we learn deeper aspects of recommendation engines. Let’s first understand about the real life and online recommendation engines.
What is the difference between Real life and online Recommendation Engine
Before summarizing the difference between Real life Recommendation engine and online Recommendation Engine lets qucikly look at the individual examples.

#### Real life Recommendation Engine:
* Your friend as movie recommendation engine.
      - We ask our friends recommend some good movies for the weekend. Most of the cases we enjoy the movies recommended by our friend.
* Your family members or friends as dress Recommendation Engine.
      - Selecting a dress from thousands of models is a little bit harder. That’s why, when we are going to buy a dress for our Birthday or any festival purpose we ask our family members or friends to select a good dress for us.

#### Why should we use recommendation engines?
There is one famous quote about customer’s relationship. The Summary of the quote will go like this ‘customers don’t know what they want until we show them’. If we succeed in showing something which customers may like business profit will sky rocket .
So recommendation engines will help customers find information, products, and services they might not have thought of. Recommendation applications are helpful in a wide variety of industries and Business.

##### Some of them we have seen before and some application listed below.
* Travel
* Financial service
* Music/Online radio
* TV and Videos
* Online publications
* Retail
* And countless others….

#### Different types of Recommendation Engines
Recommendation engines are mainly 2 types and one hybrid type:
1. User Based Filtering or Collaborative Filtering.
2. Content-Based Filtering
3. Hybrid Recommendation Systems


# 10. Reinforcement Learning


# 11. Time Series Analysis


# 12. Model Selection and Boosting
