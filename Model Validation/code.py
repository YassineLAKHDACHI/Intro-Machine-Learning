# Set up your coding environment

# Code you have previously used to load data
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
iowa_file_path = 'C:/Users/Yqssine/OneDrive/Documents/Technocholabes_Internship/Dataset/dataset/train.csv'

home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

# Specify Model
iowa_model = DecisionTreeRegressor()
# Fit Model
iowa_model.fit(X, y)

print("First in-sample predictions:", iowa_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())


# Split Your Data

# Import the train_test_split function and uncomment
# from _ import _
from sklearn.model_selection import train_test_split
# fill in and uncomment
# train_X, val_X, train_y, val_y = ____
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)


# Specify and Fit the Model

# You imported DecisionTreeRegressor in your last exercise
# and that code has been copied to the setup code above. So, no need to
# import it again
from sklearn.tree import DecisionTreeRegressor
# Specify the model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit iowa_model with the training data.
iowa_model.fit(train_X,train_y)


# Make Predictions with Validation data

# Predict with all validation observations
val_predictions = iowa_model.predict(val_X)
val_predictions

# print the top few validation predictions
print(val_predictions)
# Calculate the Mean Absolute Error in Validation Data

from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_y, val_predictions)

# uncomment following line to see the validation_mae
print(val_mae)
