import pandas as pd
import numpy as np # Make sure pandas is imported
# Assuming your data is in a CSV file named 'your_data.csv'
dataset = pd.read_csv('/content/car_data.csv') # Load your data into the 'dataset' variable
# Now you can proceed with your operations:

#Dealing with Missing Values
data_without_missing_rows = dataset.dropna()
data_without_missing_columns = dataset.dropna(axis=1)
data_without_missing_columns.shape

dataset.replace(999, 0, inplace=True)

#Correcting Data format
dataset['Price'] = pd.to_numeric(dataset['Price'])
dataset.describe()


#Data Standardization, Normalization
dataset['Price_scaled'] = dataset['Price']/dataset['Price'].max()
print(dataset['Price_scaled'])


#Min-Max Scaling in Python:
dataset['Price_scaled'] = (dataset['Price']- dataset['Price'].min()) / (dataset['Price'].max()-dataset['Price'].min())
print(dataset['Price_scaled'])


#Z-Score in Python
dataset['Price_scaled'] = (dataset['Price']- dataset['Price'].mean()) / dataset['Price'].std()
print(dataset['Price_scaled'])


#Binning
bins = np.linspace(min(dataset['Price']), max(dataset['Price']), num=4)
group_names = ['low', 'medium', 'high']
# Change 'price' to 'Price' to match the actual column name
dataset['Price_binned'] = pd.cut(dataset['Price'], bins,labels=group_names, include_lowest= True)
import matplotlib.pyplot as plt
# Change 'price' to 'Price' here as well
dataset['Price'].hist(bins=3, color='blue', alpha=0.7)
plt.grid(True)


#Turning Categorical Variables into Quantitative Variables:
pd.get_dummies(dataset['Make'])