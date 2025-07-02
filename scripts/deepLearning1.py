import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets# used to plot points and make y label of 0's and 1's using 1 single line of code
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

#1) plotting points
#points on outer circle 0 while inside circle 1
np.random.seed(0)
n_pts=500
X,y=datasets.make_circles(n_samples=n_pts,random_state=123,noise=0.1,factor=0.2)# radom state consist random number,noise will make model very complex but model will work better,factor means diameter of inside smaller than larger
# #The labels are going to be organized in a way that the larger outer circle of data points are going to be able to zero in the smaller circle of data points.
#Inside the large circle corresponds to the labels of one that's going to be our positive region. And going further from the center, we approach the negative region and so data come in many different
# inshort this function creates points which will lie in cicrle with small circle of size 0.2 times of bigger circle.if point on larger cirlce then y label 0 and if inside then y label 1. Note no learning has happend yet,while this function gives points it knows whether it will be inside or outide the circle

# plt.scatter(X[y==0,0],X[y==0,1])#plotting outer circle#selecting all x points with y label 0
# plt.scatter(X[y==1,0],X[y==1,1])
#plt.show()














#2-deep neural networkk
model=Sequential()
#hidden layer
model.add(Dense(4,input_shape=(2,),activation='sigmoid'))#4 is number of neuons in hidden layer,2 refers number of nodes in input layer
#final layer
model.add(Dense(1,activation='sigmoid'))#1 is number of neuron,input arg not required bcoz already mentioned in hidden layer
# structure of deep neural network is complete

#now we need to compile model
model.compile(Adam(learning_rate=0.01),'binary_crossentropy',metrics=['accuracy'])
h=model.fit(x=X,y=y,verbose=1,batch_size=20,epochs=60,shuffle=True)# verbose shows progress bar,it will take 500/20 times to complete 1 epoch
# this function is learning with 2 inputs X and y, x contains point and y containes 1 or 0

#plotting accuracy
# plt.plot(h.history['accuracy'])
# plt.xlabel('epoch')
# plt.legend(['accuracy'])
# plt.title('accuracy')
#plt.show()


#plotting error/loss
# plt.plot(h.history['loss'])
# plt.xlabel('epoch')
# plt.legend(['loss'])
# plt.title('loss')
#plt.show()


#plotting classifications lines
def plot_decision_boundary(X, y, model):#?
    x_span = np.linspace(min(X[:, 0]) - 0.25, max(X[:, 0]) + 0.25)
    y_span = np.linspace(min(X[:, 1]) - 0.25, max(X[:, 1]) + 0.25)
    xx, yy = np.meshgrid(x_span, y_span)
    grid = np.c_[xx.ravel(), yy.ravel()]
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)


plot_decision_boundary(X, y, model)
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plot_decision_boundary(X, y, model)
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])

# testing a point
x = 0
y = 0.75

point = np.array([[x, y]])
predict = model.predict(point)
plt.plot([x], [y], marker='o', markersize=10, color="black")
print("Prediction is: ", predict)
plt.show()











