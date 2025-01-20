import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sb
df = pd.read_csv('/content/automobile.csv')
df.head()
df = pd.read_csv('/content/automobile.csv')
df.head()
#Value Counts
df['drive-wheels'].value_counts()

#Box Plots
sb.boxplot(x = df['price'])

#Scatter Plot
sb.scatterplot(x = df['horsepower'], y = df['price'])

sb.scatterplot(x = df['highway-mpg'], y = df['price'])

#Group BY
df_test = df[['drive-wheels', 'body-style', 'price']]
df_test.head()
df_grp = df_test.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
df_grp

#Pivot Table
df_pivot = df_grp.pivot(index = 'drive-wheels', columns = 'body-style')
df_pivot

#Heat Map
sb.heatmap(df_pivot, cmap= 'YlGn')

