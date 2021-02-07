<h2> References </h2>

This project attempts to create a forecasting model for cryptocurrency. The method and approach in building forecasting model was informed by the following academic paper.

>Laura Alessandretti, Abeer ElBahrawy, Luca Maria Aiello, Andrea Baronchelli, "Anticipating Cryptocurrency Prices Using Machine Learning", Complexity, vol. 2018, Article ID 8983590, 16 pages, 2018. https://doi.org/10.1155/2018/8983590


Additionally, the following blogpost was also referred to enhance the performance of the model and served as a supplementary material.

>Ng, Y. (2019, October 03). Machine learning techniques applied to stock price prediction. Retrieved February 03, 2021, from https://towardsdatascience.com/machine-learning-techniques-applied-to-stock-price-prediction-6c1994da8001

<br>
<h2> Getting Started </h2>

There are no additional data to be downloaded or any packages to install.

To go through the entire development process, you can follow the sequence below:
1. [Data Loading](./1.%20Data%20Loading.ipynb)
2. [Feature Engineering](./2.%20Feature%20Engineering.ipynb)
3. [Modelling Routines](./3.%20Modelling%20Routines.ipynb)
4. [Trading Strategy](./4.%20Trading%20Strategy.ipynb)
5. [Model Evaluations](./4.%20Model%20Evaluations.ipynb)

The first 4 notebooks were used as workspaces to develop necessary modules to run fit different models and evaluate them. Model training and evaluations themselves are fully contained within the Model Evaluations notebook. 

<br>
<h2> Important Note </h2>

Initially, the features used for this cryptocurrency forecasting model include taking the mean, median, standard deviation, trend and last value of the cryptocurrency properties. However, after running several iterations of the models using these features, it was clear that they were ineffective in predicting the price of a currency. Therefore, they needed to be changed.

The initial features of mean, median, standard deviation, trend and last value were replaced with lag features. Lag features are essentially values from previous data points that are used as features to predict the price for the given time period. 

Note that notebooks listed above only includes development works after the change of features. To check for the models before this change of features, you can explore [Preliminary Results](./0.%20Preliminary%20Results.ipynb) which showcases all the models' performance before the change of features.
