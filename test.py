import joblib
import pandas as pd

xgb_tuned_pipe = joblib.load('xgb_tuned_pipe.pkl')
new_data = {'Address': 'Budhanilkantha', 'City': 'Kathmandu', 'Bedroom': 6, 'Bathroom': 3, 'Floors': 2.0, 'Parking': 10, 'Face': 'West', 'Year': 2073.0, 'Area': 5476.0, 'Road Width': 20.0, 'Road Type': 'Blacktopped', 'Build Area': 98568.0, 'Amenities': 16, 'Pricepersqft':2333}
new_X = pd.DataFrame([new_data])
y_pred = xgb_tuned_pipe.predict(new_X)

y = 90000000
print(f'real price {y} and predicted {int(y_pred[0])}')