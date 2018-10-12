import math as m
import abc

class Walker:
    def __init__(self, coordinates, z=0):
        self.coordinates = coordinates
        self.z = z

    def setZ(self, z):
        self.z = z

    def __str__(self):
        return ("Walker on:",self.coordinates,"With z:",self.z)

class Function(metaclass=abc.ABCMeta):
    def __init__(self,minim,maxim,dense):
        self.Min = minim
        self.Max = maxim
        self.Density = dense

    def CalculateField(self,field): 
        for i in range(len(field)):
            field[i].z = self.CalculateVector(field[i].coordinates) 

    @abc.abstractmethod
    def CalculateVector(self,vector):
        """
        Calculate one vector
        """

class SphereFunction(Function):
    def __init__(self, minim, maxim, dense):
        super().__init__(minim, maxim, dense)
        
    def CalculateVector(self,vector):
        result = 0.0
        for k in range(0,len(vector)):
            result += m.pow(vector[k],2)
        return result

class RastriginFunction(Function):
    def __init__(self, minim, maxim, dense):
        super().__init__(minim, maxim, dense)
        
    def CalculateVector(self,vector):
        result = 0.0
        for i in range(0,len(vector)):
            temp = pow(vector[i],2) - 10 * m.cos(2*m.pi * vector[i])
            result += temp
        return 10 * len(vector) + result


class RosenbrockFunction(Function):
    def __init__(self, minim, maxim, dense):
        super().__init__(minim, maxim, dense)
        
    def CalculateVector(self,vector):
        result = 0.0
        for i in range(0,len(vector)-1):
            temp = 100* pow((vector[i+1]-pow(vector[i],2)),2) + pow((1 -vector[i]),2)
            result += temp
        return result


class AckleyFunction(Function):
    def __init__(self, minim, maxim, dense):
        super().__init__(minim, maxim, dense)
        
    def CalculateVector(self,vector):
        a = 20.0
        b = 0.2
        c = 2 * m.pi
        temp1 = 0.0
        temp2 = 0.0
        result = 0.0
        for i in range(0,len(vector)):
            temp1 += m.pow(vector[i],2)
            temp2 += m.cos(c*vector[i])

        temp11  = -a * m.exp(-b * m.sqrt(temp1/ len(vector))) 
        temp21 = m.exp(m.cos(temp2) / 2)
        result = temp11 - temp21 + a + m.exp(1)
        return result


class SchwefelFunction(Function):
    def __init__(self, minim, maxim, dense):
        super().__init__(minim, maxim, dense)
        
    def CalculateVector(self,vector):
        result = 0.0
        for i in range(0,len(vector)):
            result += vector[i] * m.sin(m.sqrt(abs(vector[i])))

        result = 418.9829 * len(vector) - result
        return result 