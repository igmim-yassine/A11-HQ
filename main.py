from utils import getting_external_data, completing_columns_and_order
# import libraries
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import VotingRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings("ignore")

# Loading initial data

print('Loading initial datasets...')
to_predict = pd.read_csv("flights_topredict.csv")
data = pd.read_csv("flights_train.csv")

# Loading external data

print('Loading External datasets...')
airports = pd.read_csv("Airports_US.csv")
pop = pd.read_excel('uscities.xlsx')
US_holiday = pd.read_csv("US _Holiday.csv")

# dropping unnecesssary informations:

airports.drop(columns={'type', 'id', 'name', 'iso_region',
              'gps_code', 'scheduled_service'}, inplace=True)

data = getting_external_data(data, airports, pop, US_holiday)
to_predict = getting_external_data(to_predict, airports, pop, US_holiday)
to_predict = completing_columns_and_order(data, to_predict)

# Create X, y
y = data[['target']]
X = data.drop(columns={'target'})

# Scaling!
trans_x = RobustScaler()
trans_y = RobustScaler()

# prepraration
X = pd.DataFrame(trans_x.fit_transform(X), columns=X.columns)
y = pd.DataFrame(trans_y.fit_transform(y), columns=['target'])

# split train, test
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.15,
                                                    random_state=42,
                                                    stratify=data['Holiday'])

# define the base models
models = list()
models.append(('xgb', XGBRegressor(objective='reg:squarederror',
              max_depth=4, n_estimators=660, learning_rate=0.3)))
models.append(('lgbm', LGBMRegressor(n_estimators=800, learning_rate=0.05)))

# define the voting ensemble
ensemble = VotingRegressor(estimators=models)

# fit the model on all available data
ensemble.fit(X, y)

# prediction of test set
to_predict_1 = pd.DataFrame(trans_x.transform(
    to_predict), columns=to_predict.columns)
y_sub_ensemble = ensemble.predict(to_predict_1)

# transform y_sub_ensemble
y_sub_ensemble = trans_y.inverse_transform(y_sub_ensemble.reshape(-1, 1))

y_sub_ensemble
# generate file of submission
y_sub_ensemble = pd.DataFrame(
    data=y_sub_ensemble, index=None, columns=['Target'])
y_sub_ensemble.to_csv("A11HQ_Sub.csv", index=None, header=None)
