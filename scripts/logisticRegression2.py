import matplotlib.pyplot as plt
import numpy as np


def draw(x1,x2):
    ln=plt.plot(x1,x2)
def sigmoid(score):
    return 1 / (1 + np.exp(-score))

def calculate_error(line_parameters,all_points,y):
    m=all_points.shape[0]
    linear_combination = all_points * line_parameters  # this has exactly the same meaning as of checking a point coord in the classification line,but in matrix form we multi the two items
    p=sigmoid(linear_combination)#probability of a point whether above or below with percentage
    cross_entropy = -(1 / m) * (np.log(p).T * y + np.log(1 - p).T * (1 - y))
    return cross_entropy


def gradient_descent(line_parameters, points, y, alpha):
    n = points.shape[0]
    for i in range(2000):
        p = sigmoid(points * line_parameters)
        gradient = points.T * (p - y) * (alpha / n)
        line_parameters = line_parameters - gradient

        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)

        x1 = np.array([points[:, 0].min(), points[:, 0].max()])
        x2 = -b / w2 + (x1 * (-w1 / w2))
    draw(x1, x2)
n_pts=100
np.random.seed(0)
bias=np.ones(n_pts)
#Top region
#random x1 values are the average points on X axis where diabetic people are plotted(red dots)
random_x1_values=np.random.normal(10,2,n_pts)#nornal distribution
#random x1 values are the average points on Y axis where diabetic people are plotted(red dots)
random_x2_values=np.random.normal(12,2,n_pts)
top_region=np.array([random_x1_values,random_x2_values,bias]).transpose()# transpose so that on every row we only one X and one Y coordinate
#print(top_region)

#Botton region
botton_region=np.array([np.random.normal(5,2,n_pts),np.random.normal(6,2,n_pts),bias]).transpose()

#classifiaction line
all_points=np.vstack((top_region,botton_region))#to store all points in stack
# w1=-0.1#random values, initially we hard code this later we find the most suitable points using gradient descend
# w2=-0.15
# b=0
# line_parameters=np.matrix([w1,w2,b]).T#matrix mul only works if no of col is equal to no of rows in other
line_parameters=np.matrix([np.zeros(3)]).T
#x1=np.array([botton_region[:,0].min(),top_region[:,0].max()])
#w1x1+w2x2+b=0
#x2=-b/w2 +(x1*(-w1/w2))#it will compute x2 value of each val of x1
y=np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2, 1)#? part of cross entropy

#cross entropy
#used to calculate total error in the model, more incorrect model, larger value of cross entropy
# points above=0 and points below=1
calculate_error(line_parameters,all_points,y)

#gradient descend
#means find right position of classification line withh minimal error


#plot top and botton region
_,ax=plt.subplots(figsize=(4,4))
ax.scatter(top_region[:,0],top_region[:,1],color='r')#top region is 2 dimensional, 0 means 0th array having x axis coords, while 1 has y axis coords
ax.scatter(botton_region[:,0],botton_region[:,1],color='b')

#gradient descend
#means find right position of classification line withh minimal error
gradient_descent(line_parameters,all_points,y,0.06)



plt.show()