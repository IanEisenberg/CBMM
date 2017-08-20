# Kernel Regression
from matplotlib import pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.kernel_ridge import KernelRidge

data = sio.loadmat('data/moons_dataset.mat')
xtrain, ytrain = data['Xtr'], data['Ytr']
xtest, ytest= data['Xte'], data['Yte']

plt.scatter(xtrain[:, 0], xtrain[:, 1], c=ytrain, cmap=plt.cm.Paired)

def KernelMatrix(x1,x2,param=1,kernel='linear'):
    def squaredist(x1,x2):
        n=x1.shape[0]
        m=x2.shape[0]
        sq1 = np.sum(x1*x1,1).reshape(n,1)
        sq2 = np.sum(x2*x2,1).reshape(m,1)
        D=sq1.dot(np.ones([1,m])) + np.ones([n,1]).dot(sq2.T) - x1.dot(x2.T)*2
        return D
    
    if kernel=='linear':
        K=x1.dot(x2.T)
    elif kernel =='polynomial':
        K = (1+ x1.dot(x2.T))**param
    elif kernel == 'gaussian':
        K = np.exp(-1/(2*param**2)*squaredist(x1,x2))
    return K

def regularizerdKernLSTrain(x,y,kernel,lam):
    n=x.shape[0]
    mapped_x = kernel(x,x)
    C = np.linalg.inv(mapped_x+np.identity(n)*n*lam).dot(y)
    return C
    
def regularizedKernLSTest(C,X,kernel,Xtest):
    mapped_x = kernel(Xtest,X)
    return mapped_x.dot(C)
    
    
# kernel regression using sklearn
rgr = KernelRidge(kernel=KernelMatrix, 
                  kernel_params={'kernel': 'polynomial',
                                 'param': 3})
rgr.fit(xtrain,ytrain)

# my own code !
kernel = lambda x,y: KernelMatrix(x, y, .5, 'gaussian')
C = regularizerdKernLSTrain(xtrain,ytrain,kernel,.01)
    
# plot the decision surface
step = .05
xx, yy = np.meshgrid(np.arange(min(xtrain[:,0]), max(xtrain[:,0]), step),
                     np.arange(min(xtrain[:,1]), max(xtrain[:,1]), step))
Z = rgr.predict(np.c_[xx.ravel(), yy.ravel()])
Z=regularizedKernLSTest(C,xtrain,kernel,np.c_[xx.ravel(), yy.ravel()])
Z = np.sign(Z.reshape(xx.shape))
plt.figure(figsize=(12,8))
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=.5)
plt.scatter(xtrain[:, 0], xtrain[:, 1], c=ytrain, cmap=plt.cm.Paired)

# animate

from matplotlib import animation

def animate(i):
    kernel = lambda x,y: KernelMatrix(x, y, i, 'gaussian')
    C = regularizerdKernLSTrain(xtrain,ytrain,kernel,.01)
    Z=regularizedKernLSTest(C,xtrain,kernel,np.c_[xx.ravel(), yy.ravel()])
    Z = np.sign(Z.reshape(xx.shape))
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=.5)
    return contour,

fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
contour, = [ax.contourf(xx,yy,np.zeros(xx.shape))]

anim = animation.FuncAnimation(fig, animate, frames=100, interval=20, blit=True)


# plot eigenvalue distribution for different kernels
for p in range(1,6):
    m=KernelMatrix(xtrain,xtrain,param=p,kernel='polynomial')
    plt.plot(np.linalg.eig(m)[0], linewidth=5, label=p)
plt.xlim([0,20])
plt.legend()
