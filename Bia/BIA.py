import numpy as np
import math as m
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
import Functions as f
import time

def FindMinimum(field,w):    
    for i in range(len(field)):
            if field[i].z < w.z:
                w = field[i]

    return w

def FindMinimum2(field):
    init = field[0]
    for i in range(len(field)):
        if field[i].z < init.z:
            init = field[i]

    return init

def GetWalkerOnCoordinates(field,x,y):
    for i in range(len(field)):
        for j in range(len(field[i])):
            if field[i][j].coordinates[0] == x and field[i][j].coordinates[1] == y:
                return field[i][j]

def GenerateField(w_x,w_y):
    field = []
    mean = [w_x, w_y]
    cov = [[1, 0], [0, 100]]
    x = np.random.multivariate_normal(mean, cov, 10)
    for i in range(len(x)):
        field.append(f.Walker((x[i][0],x[i][1]),0))
    return field

def GenerateRandomUniformField(min,max,size):
    x = np.random.uniform(min,max,size)
    y = np.random.uniform(min,max,size)
    field = []
    for i in range(len(x)):
        field.append(f.Walker((x[i],y[i]),0))
    return field


def GenerateRandomWalker(w_x,w_y):
    mean = [w_x, w_y]
    cov = [[1, 0], [0, 100]]
    x = np.random.multivariate_normal(mean, cov, 1)
    w = f.Walker((x[0][0],x[0][1]),0)    
    return w

def HillClimb(iterations,x,y,func):

    field = GenerateField(x,y)
    func.CalculateField(field)
    init_w = f.Walker((x,y),0)
    init_w.z = func.CalculateVector(init_w.coordinates)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x = y = np.arange(func.Min, func.Max, func.Density)
    # X, Y = np.meshgrid(x, y)
    # zs = np.array([func.CalculateVector((x,y)) for x,y in zip(np.ravel(X), np.ravel(Y))])  
    # Z = zs.reshape(X.shape)
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # ax.scatter(init_w.coordinates[0],init_w.coordinates[1],init_w.z,color="k",s=20)

    for i in range(iterations):
        w = FindMinimum(field,init_w)
        # ax.scatter(init_w.coordinates[0],init_w.coordinates[1],init_w.z,color="r",s=20)
        field = GenerateField(w.coordinates[0],w.coordinates[1])
        func.CalculateField(field)
        init_w = w

    return init_w.z
    # plt.show()

def AnnealingAlgorithm(x,y,func):

    temperature = 5000
    final_temperature = 0.00001
    alpha = 0.99
    init_w = f.Walker((x,y),0)
    init_w.z = func.CalculateVector(init_w.coordinates)
    # fig = plt.figure()
    # ax = fig.add_subplot(111,projection='3d')
    # x = y = np.arange(func.Min, func.Max, func.Density)
    # X, Y = np.meshgrid(x, y)
    # zs = np.array([func.CalculateVector((x,y)) for x,y in zip(np.ravel(X), np.ravel(Y))])  
    # Z = zs.reshape(X.shape)
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # ax.scatter(init_w.coordinates[0],init_w.coordinates[1],init_w.z,color="b",s=20)

    iter = 0
    while temperature > final_temperature:
        for i in range(1):
            iter +=1
            neighbour_w = GenerateRandomWalker(init_w.coordinates[0],init_w.coordinates[1])
            neighbour_w.z = func.CalculateVector(neighbour_w.coordinates)
            delta = neighbour_w.z - init_w.z
            if delta < 0:
                init_w = neighbour_w
                # ax.scatter(init_w.coordinates[0],init_w.coordinates[1],init_w.z,color="b",s=20)
            else:
                r = np.random.uniform(0,1)
                if r < m.exp(-delta/temperature):
                    init_w = neighbour_w
                    # ax.scatter(init_w.coordinates[0],init_w.coordinates[1],init_w.z,color="b",s=20)

            #print(init_w.z)

        temperature = alpha * temperature
    # ax.scatter(init_w.coordinates[0],init_w.coordinates[1],init_w.z,color="r",s=20)
    # plt.show()
    return init_w.z, iter

def BlindAlgorithm(iterations,range_x,range_y,func):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(func.Min, func.Max, func.Density)
    X, Y = np.meshgrid(x, y)
    zs = np.array([func.CalculateVector((x,y)) for x,y in zip(np.ravel(X), np.ravel(Y))])
    
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    fitness = None
    last_w = None
    x, y = 0, 0
    for i in range(iterations):
        x = random.uniform(range_x[0], range_x[1])
        y = random.uniform(range_y[0], range_y[1])  
        fit = func.CalculateVector((x,y))
        if fitness == None or fit < fitness:
            fitness = fit
            last_w = f.Walker((x,y),fit)
        ax.scatter(x,y,fit,color="r",s=20)

    
    ax.scatter(last_w.coordinates[0],last_w.coordinates[1],last_w.z,color="r",s=20)
    plt.show()


#HillClimb(-2,1,f.SphereFunction(-2,2,0.1))
# HillClimb(-4,2,f.RastriginFunction(-5,5,0.1))
# HillClimb(-2,2,f.RosenbrockFunction(-2,3,0.1))
# HillClimb(-40,40,f.AckleyFunction(-40,40,0.1))
# HillClimb(-320,180,f.SchwefelFunction(-500,500,1))
# BlindAlgorithm(20,(-2,2),(-2,2),f.SphereFunction(-2,2,0.1))

# AnnealingAlgorithm(-40,40,f.SphereFunction(-50,50,1))

# res = 0
# res2 = 0
# iterations = 0
# for i in range(30):
#     tmp, iterations = AnnealingAlgorithm(-2,1,f.SchwefelFunction(-500,500,1))
#     print(iterations)
#     res2 += tmp
#     res += HillClimb(iterations,-2,1,f.SchwefelFunction(-500,500,1))

# print(res/30)
# print(res2/30)



# 30 krat spustit kazdu funkciu cez hillclimb a pocitat kolko krat som zavolal fitness + ukladat najlepsiu hodnotu a vypocitat priemer pre kazdu 
# potom spustit anneailing algorithm s rovnakym poctom fitness a opakovaniami + ukladat najlepsiu hodnotu a vypocitat priemer pre kazdu
# nakoniec ulozit do tabulky
#     hillclimb(priemer)        annealing(priemer)
# f1    0.0011081368048272478   0.016460038186587873
# f2    0.28496469037391975     1.7615023764806759
# f3    0.012414146683253805    5.276187574210139
# f4    1.9785660576933395      16.713162055571043
# f5    809.9377074209818       691.8634345551397

def GetPerturbationVector(dim,ptr_value):
    x = np.random.uniform(0,1,dim)
    for i in range(len(x)):
        if (x[i] < ptr_value):
            x[i] = 1
        else:
            x[i] = 0
    
    return x


def SomaAlgorithm(func):
    pathLength = 3
    step = 0.11
    ptrValue = 0.1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(func.Min, func.Max, func.Density)
    X, Y = np.meshgrid(x, y)
    zs = np.array([func.CalculateVector((x,y)) for x,y in zip(np.ravel(X), np.ravel(Y))])  
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    field = GenerateRandomUniformField(func.Min,func.Max,20)
    func.CalculateField(field)
    leader = FindMinimum2(field)

    for u in range(len(field)):
        ax.scatter(field[u].coordinates[0],field[u].coordinates[1],field[u].z-0.2,color="r",s=20)

    ax.scatter(leader.coordinates[0],leader.coordinates[1],leader.z-0.2,color="yellow",s=20)

    for s in range(10):
        for k in range(len(field)):
            
            jumps = []
            for i in range(0,int(pathLength/step),1):
                ptr = GetPerturbationVector(2,ptrValue)
                w_x = field[k].coordinates[0] + (leader.coordinates[0] - field[k].coordinates[0]) * (i*  step)*ptr[0]
                w_y = field[k].coordinates[1] + (leader.coordinates[1] - field[k].coordinates[1]) * (i*step)*ptr[1]
                w = f.Walker((w_x,w_y),0)
                w.z = func.CalculateVector(w.coordinates)
                jumps.append(w)
            
            new_pos = FindMinimum2(jumps)
            field[k] = new_pos
        
        ax.clear()
        x = y = np.arange(func.Min, func.Max, func.Density)
        X, Y = np.meshgrid(x, y)
        zs = np.array([func.CalculateVector((x,y)) for x,y in zip(np.ravel(X), np.ravel(Y))])  
        Z = zs.reshape(X.shape)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

        for h in range(len(field)):
            ax.scatter( field[h].coordinates[0],field[h].coordinates[1],field[h].z-0.2,color="r",s=20)

        leader = FindMinimum2(field)
        ax.scatter(leader.coordinates[0],leader.coordinates[1],leader.z-0.2,color="yellow",s=20)
        plt.pause(1.0)
    plt.show()


# SomaAlgorithm(f.SphereFunction(-2,2,0.1))


def ParticalSwarnAlgorithm(func):

    iterations = 20
    particles = 10;
    c1 = 2
    c2 = 2
    weightStart = 0.9
    wightEnd = 0.4

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(func.Min-3, func.Max+3, func.Density)
    X, Y = np.meshgrid(x, y)
    zs = np.array([func.CalculateVector((x,y)) for x,y in zip(np.ravel(X), np.ravel(Y))])  
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    field = GenerateRandomUniformField(func.Min,func.Max,particles)    
    func.CalculateField(field)
    gBest = FindMinimum2(field)

    for u in range(len(field)):
        ax.scatter(field[u].coordinates[0],field[u].coordinates[1],field[u].z-0.2,color="black",s=20)

    initialVelocity = (abs(func.Min) + abs(func.Max)) / 20
    for i in range(particles):
        field[i].rand1 = np.random.uniform(0,1,1)
        field[i].rand2 = np.random.uniform(0,1,1)
        field[i].velocity = np.tile(initialVelocity,len(field[0].coordinates))

    for i in range(iterations):
        for j in range(particles):
            newVelocityVector = []
            newCoordinatesVector = []
            weight = weightStart - ((weightStart - wightEnd)*i)/iterations
            for k in range(len(field[j].coordinates)):
                vel = 0.0
                if (field[j].pBest == None):
                    field[j].pBest = field[j]

                vel = weight * field[j].velocity[k] + c1 * field[j].rand1 * (field[j].pBest.coordinates[k] - field[j].coordinates[k]) + c2 * field[j].rand2 * (gBest.coordinates[k] - field[j].coordinates[k]) 
                
                if (vel > initialVelocity):
                    vel = initialVelocity
                newVelocityVector.append(vel)
                pos = field[j].coordinates[k] + vel
                newCoordinatesVector.append(pos)
                           
            field[j].coordinates = newCoordinatesVector
            field[j].velocity = newVelocityVector

            field[j].z = func.CalculateVector(field[j].coordinates)
            if (field[j].z < field[j].pBest.z):                
                field[j].pBest = field[j]
            if (field[j].pBest.z < gBest.z):
                gBest = field[j].pBest

        plt.pause(1.0)
        ax.clear()
        x = y = np.arange(func.Min-3, func.Max+3, func.Density)
        X, Y = np.meshgrid(x, y)
        zs = np.array([func.CalculateVector((x,y)) for x,y in zip(np.ravel(X), np.ravel(Y))])  
        Z = zs.reshape(X.shape)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

        for h in range(len(field)):
            ax.scatter( field[h].coordinates[0],field[h].coordinates[1],field[h].z-0.2,color="r",s=20)

        ax.scatter(gBest.coordinates[0],gBest.coordinates[1],gBest.z-0.2,color="yellow",s=20)
    plt.show()

ParticalSwarnAlgorithm(f.SphereFunction(-2,2,0.1))

