# import numpy library
import numpy as np
a = [1,2,3,4,5]
a
b = np.array(a)
b
# create a numpy array
a = np.array([0,1,2,3,4])
a
# print each element
print("a[0]:", a[0])
print("a[1]:", a[1])
print("a[2]:", a[2])
print("a[3]:", a[3])
print("a[4]:", a[4])
# create numpy array
c=np.array([20,1,2,3,4])
c
# assign the first elememt to 100
c[0] = 100
c

# slicing the numpy array
d = c[1:4]
d
# set the fourth element and fiifth element to 300 ansd 400
c[3:5] = 300,400
c
# get the size of the numpy array
a.size

# get the number of dimensions of numpy array
a.ndim
# get the shape/size of the numpy array
a.shape
# numpy statistical functions
# create a numpy array
a = np.array([1,-1,1,-1])
# get the mean of numpy array
mean = a.mean()
mean
# get the standard deviation of numpy array
standard_deviation=a.std()
standard_deviation
# create a numpy array
b = np.array([-1,2,3,4,5])
b
# get the smallest value in numpy array
min_b = b.min()
min_b
# mathematical functions
# the value of pi
np.pi

# create numpy array in radians
x = np.array([0,np.pi/2,np.pi])
x
# linspace
# make a numpy array within [-2,2] and 9 elements
np.linspace(-2,2,num=5)
# make a numpy array within [-2,2] and 9 elements
np.linspace(-2,2, num=9)
# create a 2D numpy array

# import the libraries
import numpy as np
# create a list
a = [[11,12,13], [21,22,23], [31,32,33]]
a
# convert list to numpy array
# every element is the same type
a = np.array(a)
a
# show the numpy array dimensions
a.ndim
# show the numpy array shape
a.shape
# create a numpy array x
x = np.array([[1,0], [0,1]])
x
# create a numpy array y
y = np.array([[2,1], [1,2]])
y
# add x and y
z = x+y
z
z1 = np.add(x,y)
z1
# multiply y with 2
z = 2*y
z
# multiply x with y
z = x*y
z