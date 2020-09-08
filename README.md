# House_prediction

This is a Kaggle competition for Data science novice. 


Version_1:

	"house-prediction_v1.ipynb" is the first successful version for predicting the House Salesprice(https://www.kaggle.com/c/house-prices-advanced-regression-techniques).
	Stacking and blending models are implemented to give a public score of .12356.


	Libraries employed:

	import numpy as np # linear algebra
 	 import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
  	import matplotlib.pyplot as plt # data visualization
  	import seaborn as sns # data visualization
  	from scipy.special import boxcox1p # data normalization
  	from scipy.stats import boxcox_normmax # data normalization
  
  	# data preprocesing
 	from sklearn.model_selection import GridSearchCV
  	from sklearn.model_selection import KFold, cross_val_score, train_test_split
  	from sklearn.metrics import mean_squared_error
  	from sklearn.preprocessing import OneHotEncoder
  	from sklearn.preprocessing import LabelEncoder
  	from sklearn.pipeline import make_pipeline
  	from sklearn.preprocessing import scale
  	from sklearn.preprocessing import StandardScaler
  	from sklearn.preprocessing import RobustScaler
  	from sklearn.decomposition import PCA
  
  	# ML models 
  	from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, StackingRegressor
  	from sklearn.kernel_ridge import KernelRidge
  	from sklearn.linear_model import Lasso, Ridge, RidgeCV, LinearRegression, LassoCV
  	from sklearn.linear_model import ElasticNet, ElasticNetCV
  	from sklearn.svm import SVR
  	from mlxtend.regressor import StackingCVRegressor
  	import lightgbm as lgb
  	from lightgbm import LGBMRegressor
  	from xgboost import XGBRegressor
	
Version_2:

	Identify and remove outliers based on data visualization and correlation analysis between salesprice and other features. Public score is improved to .12073.
	
	
Credit: 
	https://www.kaggle.com/masumrumi/a-detailed-regression-guide-with-house-pricing
	
	https://www.kaggle.com/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition
	
