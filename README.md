Following the Udemy course, "Deep Learning A-Z™: Hands-On Artificial Neural Networks", by Kirill Eremenko, Hadelin de Ponteves, SuperDataScience Team

# Deep Learning
## Artificial Neural Networks (ANN)
Artificial Neural Networks to solve a Customer Churn problem. Solving a data analytics challenge for a bank given a dataset with a large sample of the bank's customers. To make this dataset, the bank gathered information such as customer id, credit score, gender, age, tenure, balance, if the customer is active, has a credit card, etc. During a period of 6 months, the bank observed if these customers left or stayed in the bank.

The goal is to make an Artificial Neural Network that can predict, based on geo-demographical and transactional information given above, if any individual customer will leave the bank or stay (customer churn).

    1/ Accuracy:              86%
    2/ Libraries:             numpy, pandas, sklearn, tensorflow, keras
    3/ Dataset:               'Churn_Modelling.csv'
    4/ Encoding:              LabelEncoder, OnHotEncoder
    5/ Feature Scaling:       StandardScaler
    6/ Activation Function:   relu, sigmoid
    7/ Loss Function:         binary_crossentropy
    8/ Optimizer:             adam, rmsprop
    9/ Evaluating:            k-Fold Cross Validation (cross_val_score)
    10/ Improving:            Dropout
    11/ Tuning:               GridSearchCV

## Convolutional Neural Networks (CNN)
Convolutional Neural Network for Image Recognition that is able to detect various objects in images. A CNN model is implemented to  recognize a cat or a dog in a set of pictures. However, this model can be reused to detect anything else.

    1/ Accuracy:              training (85%), test (82%)
    2/ Libraries:             numpy, tensorflow, keras
    3/ Dataset:               training (4000 dogs, 4000 cats), test (1000 dogs, 1000 cats)
    4/ Activation Function:   relu, sigmoid
    5/ Loss Function:         binary_crossentropy
    6/ Optimizer:             adam
    7/ Evaluating:            not needed
    8/ Improving:             Dropout
    9/ Tuning:                GridSearchCV
    
## Recurrent Neural Networks (RNN)
Recurrent Neural Networks to predict Stock Prices. An ultra-powerful RNN model with LSTMs (Long Short Term Memory RNNs) will take the challenge to predict the real Google stock price. The RNN model will learn the trend of the actual real Google stock price data from the past 5 years (2012-2016), then predict the trend of the stock price of January, 2017. The predicted stock price will be compared with the trend of the real stock price of January, 2017. As noted, we are interested in capturing the directions or the trends of the stock price rather than the accuracy in values of the stock price.

    1/ RMSE:                  17.2815 (Relative error = 0.0214)
    2/ Libraries:             numpy, matplotlib, pandas, math, sklearn, tensorflow, keras
    3/ Dataset:               'Google_Stock_Price.csv', training (1258 days in past 5 years 2012-2016), test (20 days in January, 2017)
    4/ Feature Scalling:      MinMaxScaler
    5/ Layers:                LSTMs and Dropout regularisation
    6/ Loss Function:         mean_squared_error
    7/ Optimizer:             adam
    8/ Evaluating:            rmse (not neccessary)
    9/ Improving:             get more training data, increase timesteps, add other indicators, add more LSTM layers, add more neurones
    10/ Tuning:               GridSearchCV
    
## Self-Organizing Maps (SOM)
Self-Organizing Maps to investigate Fraud (working)
    
