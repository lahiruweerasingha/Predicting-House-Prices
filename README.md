# Predicting House Prices: A Comparative Analysis of Neural Networks and Traditional Machine Learning Models

This project focuses on predicting house prices using various machine learning techniques, including neural networks and traditional regression models. The dataset used for this analysis is the Boston housing dataset, obtained from Kaggle. The dataset contains various features related to housing attributes, such as crime rate, proportion of residential land, air pollution levels, average number of rooms, and more. The target variable is the median value of owner-occupied homes.

## Dataset obtained from:
https://www.kaggle.com/datasets/altavish/boston-housing-dataset

## Technical Stack

### Libraries Used:
- **Keras**: For building neural network models.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computing.
- **Matplotlib**: For data visualization.
- **Seaborn**: For statistical data visualization.
- **Scikit-learn**: For machine learning algorithms and evaluation metrics.
- **XGBoost**: For gradient boosting.
- **GridSearchCV**: For Hyper Parameter Optimization.

### Steps Involved:

1. **Data Import and Exploration**:
    - Importing the dataset.
    - Checking the basic information and summary statistics of the dataset.
    - Handling missing values and duplicated rows.

2. **Exploratory Data Analysis (EDA)**:
    - Visualizing the distribution of the target variable.
    - Calculating the correlation coefficients between features.
    - Exploring the relationships between features and the target variable.

3. **Data Preprocessing**:
    - Splitting the data into features and the target variable.
    - Standardizing the features using StandardScaler.

4. **Neural Network Model**:
    - Building a sequential neural network model using Keras.
    - Training the model on the training data and validating it.
    - Evaluating the model's performance using mean squared error and mean absolute error.
    - Visualizing the training and validation loss.

5. **Comparison with Traditional Machine Learning Models**:
    - Implementing various traditional regression models such as Linear Regression, Ridge Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost.
    - Training these models on the same training data.
    - Evaluating each model's performance using mean squared error and mean absolute error.
    - Comparing the performance of all models.

