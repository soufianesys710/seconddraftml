from seconddraftml.regression import MetaRandomForestRegressor
from sklearn.utils.estimator_checks import check_estimator

check_estimator(MetaRandomForestRegressor())
print("True")