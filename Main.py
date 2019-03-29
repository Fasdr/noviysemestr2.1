# import matplotlib.pyplot as plt
import numpy as np
import copy
import math


def funca(s):
    n = copy.deepcopy(s)
    for j in range(1, N-1):
        for i in range(1, j):
            n[j][i] = (-a*(s[j][i-1]-2*s[j][i]+s[j][i+1])-b*(s[j-1][i]-2*s[j][i]+s[j+1][i]))/(h**2)
    return n


def funcb(s, l):
    n = copy.deepcopy(s)
    for j in range(1, N-1):
        for i in range(1, j):
            n[j][i] = l*s[j][i]-(-a*(s[j][i-1]-2*s[j][i]+s[j][i+1])-b*(s[j-1][i]-2*s[j][i]+s[j+1][i]))/(h**2)
    return n


def prod(s1, s2):
    prsum = 0
    for j in range(N-1):
        for i in range(j+1):
            prsum += s1[j][i] * s2[j][i]
    return prsum


def norm(s):
    return math.sqrt(prod(s, s))


def normal(s):
    n = copy.deepcopy(s)
    normn = norm(n)
    for j in range(1, N-1):
        for i in range(1, j):
            n[j][i] = n[j][i]/normn
    return n


N = 5
h = 1/N
a = 1
b = 1.2
d = 1/10**6
S = [[np.sign(i*(j-i)*(N-j-1)) for i in range(j+1)] for j in range(N)]
n = 0
S1 = [S, funca(normal(S))]
lamax = [prod(normal(S1[0]), S1[1])]


while True:
    n += 1
    S1.append(funca(normal(S1[n])))
    lamax.append(prod(normal(S1[n]), S1[n+1]))
    if abs(lamax[n]-lamax[n-1])/lamax[n-1] < d:
        break

S2 = [S, funcb(normal(S), lamax[0])]
lamin = [prod(normal(S2[0]), S2[1])]
for k in range(1, n+1):
    S2.append(funcb(normal(S2[k]), lamax[k]))
    lamin.append(prod(normal(S2[k]), S2[k+1]))

print(lamax[n], lamin[n])



# m = 10
# hx = 1/m
# hy = 1/m
#
# nx = int(1/hx)+1
# ny = int(1/hy)+1
#
# u = np.zeros((ny, nx))
# f = np.zeros((ny, nx))
# ru = np.zeros((ny, nx))
#
# for i in range(0, nx):
#     for j in range(0, ny):
#         ru[j][i] = exp(i*hx)*cos(j*hy)
#
# for i in range(0, nx):
#     for j in range(0, ny):
#         f[j][i] = 0.2*exp(i*hx)*cos(j*hy)
#
#
#
#
#
# xx = np.arange(0, 1+hx, hx)
#
# yy = np.arange(0, 1+hy, hy)
#
#
# for i in range(0, 11):
#     plt.figure(figsize=(20, 10))
#     plt.subplot(1, 1,  1)
#     plt.plot(xx, u[i * int(0.1 / hy)], 'bo')
#     plt.title('y = ' + str(i/10))
#     plt.xlabel('x')
#     plt.ylabel('Value')
#     plt.grid(True)
#     plt.subplot(1, 1,  1)
#     plt.plot(xx, ru[i * int(0.1 / hy)], 'r+')
#     plt.grid(True)
#     plt.show()
#
# plt.figure(figsize=(20, 10))
# plt.subplot(1, 1,  1)
# plt.plot(yy, u[:, int(0.5 / hx)], 'bo')
# plt.title('x = 0.5')
# plt.xlabel('y')
# plt.ylabel('Value')
# plt.grid(True)
# plt.subplot(1, 1,  1)
# plt.plot(yy, ru[:, int(0.5 / hx)], 'r+')
# plt.grid(True)
# plt.show()

