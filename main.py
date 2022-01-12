import math

import numpy as np
import sympy as sp
import matplotlib as plot


class Boundary():
    def __init__(self, func1, func2, p):
        self.exp_x1 =[func1, func2]
        self.exp_dfx = [sp.diff(func1, p), sp.diff(func2, p)]
        self.exp_ddfx = [sp.diff(self.exp_dfx[0], p), sp.diff(self.exp_dfx[1], p)]
        self.exp_dddfx = sp.diff(self.exp_ddfx[0], p), sp.diff(self.exp_ddfx[1], p)

        self.x = [sp.lambdify(p,self.exp_x1[0]), sp.lambdify(p,self.exp_x1[1])]
        self.dx = [sp.lambdify(p,self.exp_dfx[0]),sp.lambdify(p,self.exp_dfx[1])]
        self.ddx = [sp.lambdify(p,self.exp_ddfx[0]),sp.lambdify(p,self.exp_ddfx[1])]
        self.dddx = [sp.lambdify(p,self.exp_dddfx[0]),sp.lambdify(p,self.exp_dddfx[0])]


normL2 = lambda x: math.sqrt(x[0]**2+x[1]**2)
ort = lambda x, t: [x[1](t), -x[0](t)]
nu = lambda x, t: [-x[1](t)/normL2([x[0](t), x[1](t)]), x[0](t)/normL2([x[0](t), x[1](t)])]
product = lambda x1, x2: x1[0]*x2[0]+x1[1]*x2[1]
normal = lambda x, t: [x[1](t)/normL2([x[0](t), x[1](t)]), -x[0](t)/normL2([x[0](t), x[1](t)])]


def K11(G1, t, tau):
    if t!=tau:
        A=product([G1.dx[0](t), G1.dx[1](t)], [G1.x[0](tau)-G1.x[0](t), G1.x[1](tau)-G1.x[1](t)])* product([G1.dx[0](tau), G1.dx[1](tau)], [G1.x[0](tau)-G1.x[0](t), G1.x[1](tau)-G1.x[1](t)])
        A/=normL2([G1.x[0](t)-G1.x[0](tau), G1.x[1](t)-G1.x[1](tau)])**4
        B = product([G1.dx[0](t), G1.dx[1](t)],[G1.dx[0](tau), G1.dx[1](tau)])/normL2([G1.x[0](t)-G1.x[0](tau), G1.x[1](t)-G1.x[1](tau)])**2
        C=1/(2*math.sin((tau-t)/2)**2)
        return 4*A-2*B-C
    else:
        A=(1/3)*product([G1.dx[0](t), G1.dx[1](t)],[G1.dddx[0](t), G1.dddx[1](t)])/normL2([G1.dx[0](t), G1.dx[1](t)])**2
        B=(1/2)*product([G1.ddx[0](t), G1.ddx[1](t)],[G1.ddx[0](t), G1.ddx[1](t)])/normL2([G1.dx[0](t), G1.dx[1](t)])**2
        C=product([G1.dx[0](t), G1.dx[1](t)],[G1.ddx[0](t), G1.ddx[1](t)])**2/normL2([G1.dx[0](t), G1.dx[1](t)])**4
        return -(1/6)+A+B-C
def K12(G0, G1, t, tau):
    return product(nu(G1.x, t), [G0.x[0](tau)-G1.x[0](t), G0.x[1](tau)-G1.x[1](t)])/(normL2([G1.x[0](t)-G0.x[0](tau), G1.x[1](t)-G0.x[1](tau)])**2)

def K21(G0, G1, t, tau):
    return product(ort(G1.dx, tau), [G0.x[0](t)-G1.x[0](tau), G0.x[1](t)-G1.x[1](tau)])/(normL2([G0.x[0](t)-G1.x[0](tau), G0.x[1](t)-G1.x[1](tau)])**2)

def K22(G0, t, tau):
    if t != tau:
        return (1/2)*math.log((4/math.e)*math.sin((t-tau)/2)**2/normL2([G0.x[0](t)-G0.x[0](tau), G0.x[1](t)-G0.x[1](tau)])**2)
    else:
        return (1/2)*math.log(1/(math.e*(normL2([G0.dx[0](t), G0.dx[1](t)])**2)))

def H00(G0, t, tau):
    if t != tau:
        return (1/2)*math.log((4/math.e)*math.sin((t-tau)/2)**2/normL2([G0.x[0](t)-G0.x[0](tau), G0.x[1](t)-G0.x[1](tau)])**2)
    else:
        return (1/2)*math.log(1/(math.e*(normL2([G0.dx[0](t), G0.dx[1](t)])**2)))

def H10(G0, G1, t, tau):
    return 0.5*math.log(1/(normL2([G1.x[0](t)-G0.x[0](tau),G1.x[1](t)-G0.x[1](tau)])))

def H01(G0, G1, t, tau):
    A = product([G0.x[0](tau)-G1.x[0](t),G0.x[1](tau)-G1.x[1](t)],normal([G1.dx[0], G1.dx[1]],t))
    B = normL2([G1.x[0](t)-G0.x[0](tau), G1.x[1](t)-G0.x[1](tau)])**2
    return A/B

def H11(G1, t, tau):
    if t != tau:
        A = product([G1.x[0](tau) - G1.x[0](t), G1.x[1](tau) - G1.x[1](t)], normal([G1.dx[0], G1.dx[1]], t))
        B = normL2([G1.x[0](t) - G1.x[0](tau), G1.x[1](t) - G1.x[1](tau)]) ** 2
        return A/B
    else:
        A = product([G1.ddx[0](t), G1.ddx[1](t)], normal([G1.dx[0], G1.dx[1]], t))
        B = 2 * normL2([G1.dx[0](t), G1.dx[1](t)])**2
        return A/B


def K00(G0, t, tau):
    if (t != tau):
        A=product([G0.x[0](tau)-G0.x[0](t), G0.x[1](tau)-G0.x[1](t)], normal([G0.dx[0], G0.dx[1]], t))
        B = normL2([G0.x[0](t)-G0.x[0](tau), G0.x[1](t)-G0.x[1](tau)])**2
        return A/B
    else:
        A=product([G0.ddx[0](tau), G0.ddx[1](t)], normal([G0.dx[0], G0.dx[1]], t))
        B = 2**normL2([G0.dx[0](t), G0.dx[1](t)])**2
        return A/B

def K10(G0, G1, t, tau):
    A=product([G1.x[0](tau)-G0.x[0](t), G1.x[1](tau)-G0.x[1](t)], normal([G0.dx[0], G0.dx[1]], t))
    B = normL2([G0.x[0](t)-G1.x[0](tau), G0.x[1](t)-G1.x[1](tau)])**2
    return A/B


def create_matrix(G0, G1, m, t):
    A = np.zeros((4*m, 4*m))

    for i in range(2*m):
        for j in range(2*m):
            A[i,j] =  H00(G0, t[i], t[j]) / (2*m) - 0.5 * R(m, t[j], t[i])
            A[i, j + (2 * m)] = H10(G0, G1, t[i], t[j]) / (2 * m)
            A[i + (2 * m), j] = H01(G0, G1, t[i], t[j]) / (2 * m)
            A[i + (2 * m), j + (2 * m)] = H11(G1, t[i], t[j]) / (2 * m)
        A[i + (2 * m), i + (2 * m)] += 1.0 / (2.0 * normL2([G1.dx[0](t[i]), G1.dx[1](t[i])]))
    return A
def R(m, tj, ti):
    tempSum = 0
    for i in range(1, m):
        tempSum += (1.0 / i) * math.cos(i * (ti - tj))
    return -(1.0 / (2 * m)) * (1.0 + (2.0 * tempSum) + (math.cos(ti - tj) / m))

def create_vector(G0, G1, f, g, m, t):
    vector = np.zeros(4*m)
    for j in range(2 * m):
        vector[j] = f(G0.x[0](t[j]), G0.x[1](t[j]))
        vector[j + (2 * m)] = g(G1.x[0](t[j]), G1.x[1](t[j]))
    return vector




def get_boundary_value(G0, G1, index, x, m , vector, n):
    result = 0
    t = [i * math.pi / n for i in range(2 * n)]
    for i in range(2 * m):
        if index==i:
            result+=-1.0 /(2*normL2([G0.dx[0](t[i]),G0.dx[1](t[i])]))
        result += (vector[i] * K00(G0, x, t[i]) / (2 * m)) + (vector[i + 2 * m] * K10(G0, G1,x, t[i]) / (2 * m))



def DN(G0, G1, f, g, m, x):
    t = [i * math.pi / m for i in range(2 * m)]
    A = create_matrix(G0, G1, m, t)
    F = create_vector(G0, G1, f, g, m, t)
    R = np.linalg.solve(A, F)
    result = 0.0
    for i in range(2 * m):
        result += R[i] * math.log(1.0 / normL2([x[0]-G0.x[0](t[i]), x[1]-G0.x[1](t[i])])) + R[i + 2 * m] * math.log(1.0 / normL2([x[0]-G1.x[0](t[i]), x[1]-G1.x[1](t[i])]))
    return (1.0 / (2.0 * m)) * result

p = [3,3]

G0 = Boundary("2*cos(t)", "2*sin(t)", "t")
G1 = Boundary("4*cos(t)", "4*sin(t)", "t")
f = sp.lambdify(["x","y"],"10")
g = sp.lambdify(["x","y"],"0")

exact = sp.lambdify(["x","y"],"10")
print(DN(G0, G1, f, g, 64,p))
print(exact(p[0], p[1]))