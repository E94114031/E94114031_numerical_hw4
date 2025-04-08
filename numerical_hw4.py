import numpy as np
from scipy.integrate import quad
import math

# ---------- 第一題 ----------
def func1(x):
    return math.exp(x) * math.sin(4 * x)

def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    total = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        total += f(a + i * h)
    return total * h

def simpsons_rule(f, a, b, n):
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += (4 if i % 2 else 2) * f(a + i * h)
    return total * h / 3

def midpoint_rule(f, a, b, n):
    h = (b - a) / n
    total = sum(f(a + (i + 0.5) * h) for i in range(n))
    return total * h

def run_problem1():
    a, b, h = 1, 2, 0.1
    n = int((b - a) / h)
    print("1")
    print(f"a. Trapezoidal Rule: {trapezoidal_rule(func1, a, b, n)}")
    print(f"b. Simpson's Rule: {simpsons_rule(func1, a, b, n)}")
    print(f"c. Midpoint Rule: {midpoint_rule(func1, a, b, n)}")

# ---------- 第二題 ----------
def func2(x):
    return x**2 * np.log(x)

def gauss_legendre(f, a, b, n):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    transformed = 0.5 * (nodes * (b - a) + (b + a))
    return 0.5 * (b - a) * np.sum(weights * f(transformed))

def run_problem2():
    a, b = 1, 1.5
    result_true, _ = quad(func2, a, b)
    print("2")
    print(f"n = 3: {gauss_legendre(func2, a, b, 3)}")
    print(f"n = 4: {gauss_legendre(func2, a, b, 4)}")
    print(f"Exact: {result_true}")

# ---------- 第三題 ----------
def func3(x, y):
    return 2 * y * np.sin(x) + np.cos(x)**2

def bodes_double_integral(f, a, b, nx, ny):
    hx = (b - a) / nx
    total = 0
    for i in range(nx + 1):
        x = a + i * hx
        ya, yb = np.sin(x), np.cos(x)
        hy = (yb - ya) / ny
        inner_sum = 0
        for j in range(ny + 1):
            y = ya + j * hy
            cy = 7 if j in (0, ny) else 32 if j % 3 != 2 else 12
            inner_sum += cy * f(x, y)
        cx = 7 if i in (0, nx) else 32 if i % 3 != 2 else 12
        total += cx * (2 * hy / 45) * inner_sum
    return total * (2 * hx / 45)

def gauss_legendre_2d(f, a, b, nx, ny):
    x_nodes, x_weights = np.polynomial.legendre.leggauss(nx)
    x_mapped = 0.5 * (b - a) * x_nodes + 0.5 * (b + a)
    total = 0
    for i in range(nx):
        x = x_mapped[i]
        wx = x_weights[i]
        y1, y2 = np.sin(x), np.cos(x)
        y_nodes, y_weights = np.polynomial.legendre.leggauss(ny)
        y_mapped = 0.5 * (y2 - y1) * y_nodes + 0.5 * (y2 + y1)
        total += wx * (0.5 * (y2 - y1)) * np.sum(y_weights * f(x, y_mapped))
    return 0.5 * (b - a) * total

def run_problem3():
    a, b = 0, np.pi / 4
    exact, _ = quad(lambda x: quad(lambda y: func3(x, y), np.sin(x), np.cos(x))[0], a, b)
    print("3")
    print(f"a. Bode's Rule: {bodes_double_integral(func3, a, b, 4, 4)}")
    print(f"b. Gauss-Legendre 2D: {gauss_legendre_2d(func3, a, b, 3, 3)}")
    print(f"c. Exact: {exact}")

# ---------- 第四題 ----------
def func4(x):
    return x**(-1/4) * np.sin(x)

def transformed_func4(t):
    return t**2 * np.sin(1/t) if t > 0 else 0

def run_problem4():
    a, b = 1e-6, 1
    print("4")
    print(f"a. Simpson's ∫ x^(-1/4) sin(x) dx: {simpsons_rule(func4, a, b, 4)}")
    print(f"b. Transformed ∫ x^(-4) sin(x) dx: {simpsons_rule(transformed_func4, 0, 1, 4)}")

# -------- 直接執行所有題目 --------
run_problem1()
run_problem2()
run_problem3()
run_problem4()