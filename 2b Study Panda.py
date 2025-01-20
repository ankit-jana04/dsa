import pandas as pd

#data = pd.Series([1, 2, 3, 4, 5])

#is_empty = data.empty
#print(f"Is the Series empty? {is_empty}")
#ndim = data.ndim
#print(f"Number of dimensions: {ndim}")

#size = data.size
#print(f"Size of the Series: {size}")
#dtype = data.dtype
#print(f"Data type of the Series: {dtype}")
#head = data.head()
#print("Head of the Series:")
#print(head)

#tail = data.tail()
#print("Tail of the Series:")
#print(tail)
import pandas as pd
import numpy as np

data = {
'A': [1, 2, 3, 4, 5],
'B': [10, 20, 30, 40, 50],
'Date': pd.date_range(start='2023-01-01', periods=5)
}
df = pd.DataFrame(data)

datatype = df.dtypes
print("Datatype of each column:")
print(datatype)

transpose = df.T
print("\nTransposed DataFrame:")
print(transpose)

is_empty = df.empty
print("\nIs DataFrame empty?")
print(is_empty)

ndim = df.ndim
print("\nNumber of dimensions:")
print(ndim)

shape = df.shape
print("\nShape of DataFrame:")
print(shape)

size = df.size
print("\nTotal number of elements:")
print(size)

values = df.values
print("\nValues in DataFrame:")
print(values)

head = df.head()
print("\nFirst 5 rows of DataFrame:")
print(head)

tail = df.tail()
print("\nLast 5 rows of DataFrame:")
print(tail)
is_datetime = pd.api.types.is_datetime64_any_dtype(df['Date'])
print("\nIs 'Date' column of datetime type?")

print(is_datetime)