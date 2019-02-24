import matplotlib.pyplot as plt
import numpy as np
from math import exp, sin, cos, pi

m = 10
hx = 1/m
hy = 1/m

nx = int(1/hx)+1
ny = int(1/hy)+1

u = np.zeros((ny, nx))
f = np.zeros((ny, nx))
ru = np.zeros((ny, nx))

for i in range(0, nx):
    for j in range(0, ny):
        ru[j][i] = exp(i*hx)*cos(j*hy)

for i in range(0, nx):
    for j in range(0, ny):
        f[j][i] = 0.2*exp(i*hx)*cos(j*hy)





xx = np.arange(0, 1+hx, hx)

yy = np.arange(0, 1+hy, hy)


for i in range(0, 11):
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 1,  1)
    plt.plot(xx, u[i * int(0.1 / hy)], 'bo')
    plt.title('y = ' + str(i/10))
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.grid(True)
    plt.subplot(1, 1,  1)
    plt.plot(xx, ru[i * int(0.1 / hy)], 'r+')
    plt.grid(True)
    plt.show()

plt.figure(figsize=(20, 10))
plt.subplot(1, 1,  1)
plt.plot(yy, u[:, int(0.5 / hx)], 'bo')
plt.title('x = 0.5')
plt.xlabel('y')
plt.ylabel('Value')
plt.grid(True)
plt.subplot(1, 1,  1)
plt.plot(yy, ru[:, int(0.5 / hx)], 'r+')
plt.grid(True)
plt.show()

