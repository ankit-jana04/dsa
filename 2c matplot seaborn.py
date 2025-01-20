import seaborn as sns

# Load the Tips dataset
tips = sns.load_dataset("tips")

# Create a histogram of the total bill amounts
sns.histplot(data=tips, x="total_bill")

import seaborn as sns

# Load the exercise dataset
exercise = sns.load_dataset("exercise")

# check the head
exercise.head()

import seaborn as sns
tips = sns.load_dataset("tips")
sns.scatterplot(x="total_bill", y="tip", data=tips)

import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

# customize the scatter plot
sns.scatterplot(x="total_bill", y="tip", hue="sex", size="size",
sizes=(50, 200), data=tips)

# add labels and title
plt.xlabel("Total Bill")
plt.ylabel("Tip")
plt.title("Relationship between Total Bill and Tip")

# display the plot
plt.show()
import seaborn as sns
fmri = sns.load_dataset("fmri")
sns.lineplot(x="timepoint", y="signal", data=fmri)
import seaborn as sns

import matplotlib.pyplot as plt

fmri = sns.load_dataset("fmri")

# customize the line plot
sns.lineplot(x="timepoint", y="signal", hue="event", style="region",
markers=True, dashes=False, data=fmri)

# add labels and title
plt.xlabel("Timepoint")
plt.ylabel("Signal Intensity")
plt.title("Changes in Signal Intensity over Time")

# display the plot
plt.show()
import seaborn as sns
titanic = sns.load_dataset("titanic")
sns.barplot(x="class", y="fare", data=titanic)
import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset("titanic")

# customize the bar plot
sns.barplot(x="class", y="fare", hue="sex", ci=None, palette="muted",
data=titanic)

# add labels and title
plt.xlabel("Class")
plt.ylabel("Fare")
plt.title("Average Fare by Class and Gender on the Titanic")

# display the plot
plt.show()
import seaborn as sns
iris = sns.load_dataset("iris")
sns.histplot(x="petal_length", data=iris)
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris")

# customize the histogram
sns.histplot(data=iris, x="petal_length", bins=20, kde=True,
color="green")

# add labels and title
plt.xlabel("Petal Length (cm)")
plt.ylabel("Frequency")
plt.title("Distribution of Petal Lengths in Iris Flowers")

# display the plot
plt.show()
import seaborn as sns
tips = sns.load_dataset("tips")
sns.kdeplot(data=tips, x="total_bill")
import seaborn as sns
import matplotlib.pyplot as plt

# Load the "tips" dataset from Seaborn

tips = sns.load_dataset("tips")

# Create a density plot of the "total_bill" column from the "tips" dataset
# We use the "hue" parameter to differentiate between "lunch" and "dinner" meal times
# We use the "fill" parameter to fill the area under the curve
# We adjust the "alpha" and "linewidth" parameters to make the plotmore visually appealing
sns.kdeplot(data=tips, x="total_bill", hue="time", fill=True,
alpha=0.6, linewidth=1.5)

# Add a title and labels to the plot using Matplotlib
plt.title("Density Plot of Total Bill by Meal Time")
plt.xlabel("Total Bill ($)")
plt.ylabel("Density")

# Show the plot
plt.show()
import seaborn as sns
tips = sns.load_dataset("tips")
sns.boxplot(x="day", y="total_bill", data=tips)

import seaborn as sns
import matplotlib.pyplot as plt

# load the tips dataset from Seaborn
tips = sns.load_dataset("tips")

# create a box plot of total bill by day and meal time, using the "hue" parameter to differentiate between lunch and dinner
# customize the color scheme using the "palette" parameter
# adjust the linewidth and fliersize parameters to make the plot more visually appealing
sns.boxplot(x="day", y="total_bill", hue="time", data=tips,
palette="Set3", linewidth=1.5, fliersize=4)

# add a title, xlabel, and ylabel to the plot using Matplotlib functions
plt.title("Box Plot of Total Bill by Day and Meal Time")
plt.xlabel("Day of the Week")
plt.ylabel("Total Bill ($)")

# display the plot

plt.show()
import seaborn as sns

# load the iris dataset from Seaborn
iris = sns.load_dataset("iris")

# create a violin plot of petal length by species
sns.violinplot(x="species", y="petal_length", data=iris)

# display the plot
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
tips = sns.load_dataset('tips')
numerical_cols = tips.select_dtypes(include=['number']) # Filter numerical columns

# Create a heatmap of the correlation between variables

corr = numeric_tips.corr()

sns.heatmap(corr)

# Show the plot
plt.show()




#Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

np.random.seed(0)
data = np.random.randn(100)
categories = ['A', 'B', 'C', 'D']
category_counts = [10, 15, 7, 12]

plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.plot(data, marker='o', color='b', linestyle='-')
plt.title('Line Plot')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.scatter(np.arange(100), data, color='r', marker='x')
plt.title('Scatter Plot')

plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
# 3. Bar Plot
plt.subplot(2, 2, 3)
sns.barplot(x=categories, y=category_counts, palette='viridis')
plt.title('Bar Plot')
plt.ylabel('Counts')
# 4. Histogram
plt.subplot(2, 2, 4)
plt.hist(data, bins=15, color='g', alpha=0.7)
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
# 5. Pie Chart
plt.figure(figsize=(8, 6))
plt.pie(category_counts, labels=categories, autopct='%1.1f%%',
startangle=90, colors=sns.color_palette('pastel'))
plt.title('Pie Chart')
plt.axis('equal') # Equal aspect ratio ensures that pie chart is
circular
plt.show()
# 6. Count Plot (Using Seaborn)
# Creating a DataFrame for count plot
df = pd.DataFrame({'Category': np.random.choice(categories, size=100)})
plt.figure(figsize=(8, 6))
sns.countplot(x='Category', data=df, palette='husl')
plt.title('Count Plot')
plt.xlabel('Category')
plt.ylabel('Counts')
plt.show()