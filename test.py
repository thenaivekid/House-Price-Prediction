import joblib
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

categorical_cols = ['Address', 'City', 'Face', 'Road Type']
numeric_cols = ['Bedroom', 'Bathroom', 'Floors', 'Parking', 'Year', 'Area',
       'Road Width', 'Build Area', 'Amenities', 'Pricepersqft']
t = [('cat', OneHotEncoder(handle_unknown = 'ignore'), categorical_cols), ('num', MinMaxScaler(), numeric_cols)]
col_transform = ColumnTransformer(transformers=t)

# Load the saved model
xgb_tuned_pipe = joblib.load('xgb_tuned_pipe.pkl')
# create a dictionary with the new data point
new_data = {'Address': 'Budhanilkantha', 'City': 'Kathmandu', 'Bedroom': 6, 'Bathroom': 3, 'Floors': 2.0, 'Parking': 10, 'Face': 'West', 'Year': 2073.0, 'Area': 5476.0, 'Road Width': 20.0, 'Road Type': 'Blacktopped', 'Build Area': 98568.0, 'Amenities': 16, 'Pricepersqft':2333}
new_X = col_transform.transform(new_data)

# convert the dictionary to a dataframe
new_X = pd.DataFrame([new_X])


y_pred = xgb_tuned_pipe.predict(new_X)

y = 90000000