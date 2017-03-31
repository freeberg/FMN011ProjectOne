import numpy as np
from numpy import power, linspace, sqrt
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy as sp
from scipy.optimize import fsolve, root
from numpy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D


def geval(b, d, l):
    if(l == 8 or l == 15):
        l1 = l2 = l3 = l4 = l5 = l6 = l
        print(l1, l2, l3, l4, l5, l6)
    else:
        l1, l2, l3, l4, l5, l6 = l
        print(l1, l2, l3, l4, l5, l6)

    # pi = b/2 (b**2 + l2im1**2 - l2i**2 )
    p1 = 1/(2*b) * (b**2 + l1**2 - l2**2 )
    p2 = 1/(2*b) * (b**2 + l3**2 - l4**2 )
    p3 = 1/(2*b) * (b**2 + l5**2 - l6**2 )
    
    print("p1: ", p1)
    print("p2: ", p2)
    print("p3: ", p3)

    # hi = sqrt(L2im1**2 - Pi**2)
    h1 = sqrt(l1**2 - p1**2)
    h2 = sqrt(l3**2 - p2**2) 
    h3 = sqrt(l5**2 - p3**2)

    print("h1: ", h1)
    print("h2: ", h2)
    print("h3: ", h3)

    xp1=(sqrt(3)/6)*(2*b+d-3*p1)
    xp2=-(sqrt(3)/6)*(b+2*d)
    xp3=-(sqrt(3)/6)*(b-d-3*p3)

    print("xp1: ", xp1)
    print("xp2: ", xp2)
    print("xp3: ", xp3)

    yp1=1/2*(d+p1)
    yp2=1/2*(b-2*p2)
    yp3=-1/2*(b+d-p3)

    print("yp1: ", yp1)
    print("yp2: ", yp2)
    print("yp3: ", yp3)

    return p1, p2, p3, h1, h2, h3, xp1, xp2, xp3, yp1, yp2, yp3

def stf(xt1, xt2, xt3):
    a = 10
    b = 15
    d = 1
    l = 8
    p1, p2, p3, h1, h2, h3, xp1, xp2, xp3, yp1, yp2, yp3 = geval(b, d, l)
    
    def f1(x1, x2, x3) : return a**2 + 2*x1*x2 - 2*x1*(xp1 + sqrt(3)*(yp1 - yp2)) - 2*xp2*x2-((sqrt(3)*xp1 - yp1 + yp2)**2 + (h1**2 + h2**2) - 4*xp1**2 - xp2**2) + 2*sqrt((h1**2 - 4*(x1-xp1)**2)*(h2**2 - (x2 - xp2)**2))                          
    def f2(x1, x2, x3) : return  a**2 - 4*x1*x3 - 2*x1*(xp1 - 3*xp3 + sqrt(3)*(yp1 - yp3)) - 2*x3*(-3*xp1 + xp3 + sqrt(3)*(yp1 - yp3)) - ((sqrt(3)*(xp1 + xp3) - yp1 + yp3)**2 + (h1**2+ h3**2) - 4*xp1**2 - 4*xp3**2) + 2*sqrt((h1**2 - 4*(x1-xp1)**2)*(h3**2 - 4*(x3 - xp3)**2))    
    def f3(x1, x2, x3) : return  a**2 + 2*x2*x3 - 2*x3*(xp3 + sqrt(3)*(yp2 - yp3)) - 2*xp2*x2-((sqrt(3)*xp3 - yp2 + yp3)**2 + (h2**2 + h3**2)- xp2**2 - 4*xp3**2) + 2*sqrt((h2**2 - (x2-xp2)**2)*(h3**2 - 4*(x3 - xp3)**2))

    print("2.3a = ", f1(xt1, xt2, xt3))
    print("2.3b = ", f2(xt1, xt2, xt3))
    print("2.3c = ", f3(xt1, xt2, xt3))

def residual(a, b, d, l, stVal1, stVal2, stVal3):
    p1, p2, p3, h1, h2, h3, xp1, xp2, xp3, yp1, yp2 ,yp3 = geval(b, d, l)
    
    def f1(x1, x2, x3) : return a**2 + 2*x1*x2 - 2*x1*(xp1 + sqrt(3)*(yp1 - yp2)) - 2*xp2*x2-((sqrt(3)*xp1 - yp1 + yp2)**2 + (h1**2 + h2**2) - 4*xp1**2 - xp2**2) + 2*sqrt((h1**2 - 4*(x1-xp1)**2)*(h2**2 - (x2 - xp2)**2))                          
    def f2(x1, x2, x3) : return a**2 - 4*x1*x3 - 2*x1*(xp1 - 3*xp3 + sqrt(3)*(yp1 - yp3)) - 2*x3*(-3*xp1 + xp3 + sqrt(3)*(yp1 - yp3)) - ((sqrt(3)*(xp1 + xp3) - yp1 + yp3)**2 + (h1**2+ h3**2) - 4*xp1**2 - 4*xp3**2) + 2*sqrt((h1**2 - 4*(x1-xp1)**2)*(h3**2 - 4*(x3 - xp3)**2))    
    def f3(x1, x2, x3) : return a**2 + 2*x2*x3 - 2*x3*(xp3 + sqrt(3)*(yp2 - yp3)) - 2*xp2*x2-((sqrt(3)*xp3 - yp2 + yp3)**2 + (h2**2 + h3**2)- xp2**2 - 4*xp3**2) + 2*sqrt((h2**2 - (x2-xp2)**2)*(h3**2 - 4*(x3 - xp3)**2))
    
    def fA(p) : 
        x1, x2, x3 = p
        return [f1(x1, x2, x3), f2(x1, x2, x3), f3(x1, x2, x3)]
    
    x, y, z = fsolve(fA, (stVal1, stVal2, stVal3))
    print(x, y, z)

    x1a = sqrt(3)/6 * 10
    x2a = -sqrt(3)/3 * 10
    x3a = sqrt(3)/6 * 10

    e1 = x - x1a
    e2 = y - x2a
    e3 = z - x3a
    print(e1, e2, e3)

    return e1, e2, e3

def solveEqs(a, b, d, l, stVal1, stVal2, stVal3):
    p1, p2, p3, h1, h2, h3, xp1, xp2, xp3, yp1, yp2 ,yp3 = geval(b, d, l)
    
    def f1(x1, x2, x3) : return a**2 + 2*x1*x2 - 2*x1*(xp1 + sqrt(3)*(yp1 - yp2)) - 2*xp2*x2-((sqrt(3)*xp1 - yp1 + yp2)**2 + (h1**2 + h2**2) - 4*xp1**2 - xp2**2) + 2*sqrt((h1**2 - 4*(x1-xp1)**2)*(h2**2 - (x2 - xp2)**2))                          
    def f2(x1, x2, x3) : return a**2 - 4*x1*x3 - 2*x1*(xp1 - 3*xp3 + sqrt(3)*(yp1 - yp3)) - 2*x3*(-3*xp1 + xp3 + sqrt(3)*(yp1 - yp3)) - ((sqrt(3)*(xp1 + xp3) - yp1 + yp3)**2 + (h1**2+ h3**2) - 4*xp1**2 - 4*xp3**2) + 2*sqrt((h1**2 - 4*(x1-xp1)**2)*(h3**2 - 4*(x3 - xp3)**2))    
    def f3(x1, x2, x3) : return a**2 + 2*x2*x3 - 2*x3*(xp3 + sqrt(3)*(yp2 - yp3)) - 2*xp2*x2-((sqrt(3)*xp3 - yp2 + yp3)**2 + (h2**2 + h3**2)- xp2**2 - 4*xp3**2) + 2*sqrt((h2**2 - (x2-xp2)**2)*(h3**2 - 4*(x3 - xp3)**2))
    
    def fA(p) : 
        x1, x2, x3 = p
        return [f1(x1, x2, x3), f2(x1, x2, x3), f3(x1, x2, x3)]
    
    x, y, z = fsolve(fA, (stVal1, stVal2, stVal3))
    return x, y, z


def plotDifferentPos(input):
    a = 10
    b = 15
    d = 1
    if(input == 0):
        l = 8
    elif(input == 1):
        l = 15
    elif(input == 2):
        l = [15, 15, 8, 8, 8, 8]
    else:
        l = [8, 15, 8, 15, 8, 15]
    
    p1, p2, p3, h1, h2, h3, xp1, xp2, xp3, yp1, yp2, yp3 = geval(b, d, l)


    if(condition(8, 15, a, b)):
        xt1, xt2, xt3 = solveEqs(a, b, d, l, 2.886751345948, -5.773502691896257, 2.886751345948128)
        yt1 = sqrt(3)*xt1-(sqrt(3)*xp1-yp1)
        yt2 = yp2
        yt3 = -sqrt(3)*xt3+(sqrt(3)*xp3+yp3)
        
        zt1 = sqrt(h1**2 - 4*(xt1-xp1)**2)
        zt2 = sqrt(h2**2 - (xt2-xp2)**2)
        zt3 = sqrt(h3**2 - 4*(xt3-xp3)**2)

    else:
        lmin = 8
        lmax = 15
        xt1 = sqrt(3) / (4*b) * (lmin**2 - lmax**2) + 1/2 * sqrt(1/3*a**2 - 1/(4*b**2)*(lmax**2 - lmin**2)**2)
        yt1 = 1 / (4*b) * (lmax**2 - lmin**2) + sqrt(3) / 2 * sqrt(1/3*a**2 - 1/(4*b**2)*(lmax**2 - lmin**2)**2)
        xt2 = -sqrt(1/3*a**2 - 1/(4*b**2)*(lmax**2 - lmin**2)**2)
        yt2 = -1/(2*b) * (lmax**2 - lmin**2)
        xt3 = sqrt(3) / (4*b) * (lmax**2 - lmin**2) + 1/2 * sqrt((1/3)*a**2 - 1/(4*b**2)*(lmax**2 - lmin**2)**2)
        yt3 = 1/(4*b) * (lmax**2 - lmin**2) - sqrt(3)/2 * sqrt((1/3)*a**2 - 1/(4*b**2) * (lmax**2 - lmin**2)**2)
        zt1 = zt2 =zt3 = sqrt(((1/2*(lmax**2 + lmin**2) - 1/3*(a**2 + b**2 + b*d + d**2) + (b + 2*d)/sqrt(3) * sqrt(1/3*a**2 - 1/(4*b**2)*(lmax**2-lmin**2)**2))))
        

        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = (xt1, xt2, xt3, xt1)
    Y = (yt1, yt2, yt3, yt1)
    Z = (zt1, zt2, zt3, zt1)
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    #ax.auto_scale_xyz([-7, 7], [-7, 7], [0, 15])
    ax.plot_trisurf(X, Y, Z)
    print("xt1: ", xt1, " xt2: ", xt2, " xt3: ", xt3, " yt1: ", yt1, " yt2: ", yt2, " yt3: ", yt3, " zt1: ", zt1, " zt2: ", zt2, " zt3: ", zt3)
    plt.show()


def condition(lmin, lmax, a, b):
    return (lmax**2 - lmin**2) >= 2*a*b / sqrt(3)



    



#a0 = float(input("Chose values (a): "))
#b0 = float(input("Chose values (b): "))
#d0 = float(input("Chose values (d): "))
#l0 = float(input("Chose values (l): "))
#stVal10 = float(input("Chose values (stVal1): "))
#stVal20 = float(input("Chose values (stVal2): "))
#stVal30 = float(input("Chose values (stVal3): "))

#residual(10, 15, 1, [8, 15, 8, 15, 8, 15], 2.8867, -5.7735, 2.8867)

#geval(b0, d0, l0)

#xt10 = float(input("Chose values (x1): "))
#xt20 = float(input("Chose values (x2): "))
#xt30 = float(input("Chose values (x3): "))

#stf(sqrt(3)/6 * 10, -sqrt(3)/3 * 10, sqrt(3)/6 * 10)
pos = float(input("Chose figure: "))
plotDifferentPos(pos)

