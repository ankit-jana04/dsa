import pandas as pd
import numpy as np
dataset = pd.read_csv("/content/car_data.csv")
dataset = pd.read_csv("/content/car_data.csv")
dataset = pd.DataFrame(dataset)
type(dataset)
dataset.shape
dataset_2 = dataset.drop(['Make'], axis=1)

dataset_2.shape
dataset_2 = pd.read_csv("/content/car_data.csv")
combined_data = pd.merge(dataset, dataset_2, on='Make')
combined_data.shape
dataset_3 = combined_data.sort_values(by=['Make'])
dataset_3.head()
final_data = pd.concat([combined_data, dataset_3])
final_data.shape
summary = final_data.describe()
print(summary)
numeric_data = final_data.select_dtypes(include=np.number)

skewness = numeric_data.skew()
print("skewness:")
print(skewness)
correlation_matrix = numeric_data.corr()
print("correlation matrix:")
print(correlation_matrix)