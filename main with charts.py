from sympy import *
import matplotlib.pyplot as plt
import sys

maxn = int(1e3 + 10)

coef_ab = maxn * [-1]
coef_am = maxn * [-1]
coef_inv = [
    [1],
    [1, 1],
    [2/3, 4/3, -1/3],
    [6/11, 18/11, -9/11, 2/11],
    [12/25, 48/25, -36/25, 16/25, -3/25],
    [60/137, 300/137, -300/137, 200/137, -75/137, 12/137],
    [60/147, 360/147, -450/147, 400/147, -255/147, 72/147, -10/147]        
]

def freopen(filename: str, mode: str):
    if mode == 'r':
        sys.stdin = open(filename, mode)
    elif mode == 'w':
        sys.stdout = open(filename, mode)

def beta_ab(order: int, i: int):
    if(coef_ab[order] == -1):
        s = order
        t = symbols('t')
        coef_ab[order] = []
        for j in range(0, s):
            args = '1'
            for k in range(0, s):
                if k == s - j - 1:
                    continue
                args += ' * (t + ' + str(k) + ')'
            f = sympify(args)
            f = lambdify((t), f, 'numpy')
            coef_ab[s].append(integrate(f(t), (t, 0, 1)) * ((-1) ** (s - j - 1)) / (factorial(j) * factorial(s - j - 1)))
    return coef_ab[order][i]

def beta_am(order: int, i: int):
    if(coef_am[order] == -1):
        s = order
        t = symbols('t')
        coef_am[order] = []
        for j in range(0, s + 1):
            args = '1'
            for k in range(0, s + 1):
                if k == s - j:
                    continue
                args += ' * (t + ' + str(k) + ' - 1)'
            f = sympify(args)
            f = lambdify((t), f, 'numpy')
            coef_am[s].append(integrate(f(t), (t, 0, 1)) * ((-1) ** (s - j)) / (factorial(j) * factorial(s - j)))
    return coef_am[order][i]

def evalf(expr: str):
    expr = sympify(expr)
    t, y = symbols('t y')
    return lambdify((t, y), expr, 'numpy')

def euler(y: list, t0: float, h: float, steps: int, f):
    ans = []
    for i in range(0, steps + 1):
        ans.append(y[0])
        y[0] += h * f(t0, y[0])
        t0 += h
    return ans

def euler_inverso(y: list, t0: float, h: float, steps: int, f):
    ans = []
    for i in range(0, steps + 1):
        ans.append(y[0])
        y[0] += h * f(t0 + h, y[0] + h * f(t0, y[0]))
        t0 += h
    return ans

def euler_aprimorado(y: list, t0: float, h: float, steps: int, f):
    ans = []
    for i in range(0, steps + 1):
        ans.append(y[0])
        y[0] += (h / 2) * (f(t0, y[0]) + f(t0 + h, y[0] + h * f(t0, y[0])))
        t0 += h
    return ans

def runge_kutta(y: list, t0: float, h: float, steps: int, f):
    k = 5 * [0]
    ans = []
    for i in range(0, steps + 1):
        ans.append(y[0])
        k[1] = h * f(t0, y[0])
        k[2] = h * f(t0 + (h / 2), y[0] + (k[1] / 2))
        k[3] = h * f(t0 + (h / 2), y[0] + (k[2] / 2))
        k[4] = h * f(t0 + h, y[0] + k[3])
        y[0] += (1 / 6) * (k[1] + 2 * k[2] + 2 * k[3] + k[4])
        t0 += h
    return ans

def get_next_ab(y: list, t0: float, h: float, steps: int, f, order: int):
    y_n = y[-1]; tot = 0
    for j in range(-order, 0):
        tot += beta_ab(order, order + j) * f(t0 + (order + j) * h, y[j])
    return y_n + h * tot

def get_next_am(y: list, t0: float, h: float, steps: int, f, order: int):
    y_n = y[-2]; tot = 0
    sz = order + 1
    for j in range(-sz, 0):
        tot += beta_am(order, sz + j) * f(t0 + (sz + j) * h, y[j])
    return y_n + h * tot

def get_next_inv(y: list, t0: float, h: float, steps: int, f, order: int):
    sz = order + 1
    y_n = coef_inv[order][0] * h * f(t0 + order * h, y[-1])
    for j in range(-sz, -1):
        y_n += coef_inv[order][-(sz + j + 1)] * y[j]
    return y_n

def adam_bashforth(y: list, t0: float, h: float, steps: int, f, order: int):
    for j in range(len(y), steps + 1):
        y.append(get_next_ab(y, t0, h, steps, f, order))
        t0 += h
    return y

def adam_multon(y: list, t0: float, h: float, steps: int, f, order: int):
    for j in range(len(y), steps + 1):
        y.append(get_next_ab(y, t0, h, steps, f, order - 1))
        y[-1] = get_next_am(y, t0, h, steps, f, order - 1)
        t0 += h
    return y

def formula_inversa(y: list, t0: float, h: float, steps: int, f, order: int):
    for j in range(len(y), steps + 1):
        y.append(get_next_ab(y, t0, h, steps, f, order - 1))
        y[-1] = get_next_inv(y, t0, h, steps, f, order - 1)
        t0 += h
    return y

def main():
    freopen('input.txt', 'r')
    freopen('output.txt', 'w')
    for line in sys.stdin:
        line = line.split()
        method = line[0].split('_')
        points = []
        print('Metodo', end = ' ')
        if method[0] == 'adam' or method[0] == 'formula':
            order = int(line[-1]); f = evalf(line[-2])
            steps = int(line[-3]); h = float(line[-4])
            t0 = float(line[-5]); y0 = float(line[-6])
            if method[1] == 'bashforth':
                print('Adan-Bashforth', end = ' ')
            elif method[1] == 'multon':
                print('Adan-Multon', end = ' ')
            else:
                print('Formula Inversa de Diferenciacao', end = ' ')
            if not method[-1] in ['bashforth', 'multon', 'inversa']:
                aux = method[1] == 'bashforth'
                if method[-1] == 'euler':
                    print('por Euler')
                    points = euler([y0], t0, h, order - 2 + aux, f)
                elif method[-1] == 'inverso':
                    print('por Euler Inverso')
                    points = euler_inverso([y0], t0, h, order - 2 + aux, f)
                elif method[-1] == 'aprimorado':
                    print('por Euler Aprimorado')
                    points = euler_aprimorado([y0], t0, h, order - 2 + aux, f)
                elif method[-1] == 'kutta':
                    print('por Runge-Kutta ( ordem =', order, ')')
                    points = runge_kutta([y0], t0, h, order - 2 + aux, f)        
            else:
                sz = order
                if method[1] != 'bashforth':
                    sz -= 1
                for i in range(1, sz + 1):
                    points.append(float(line[i]))
                print('')
            if method[1] == 'bashforth':
                points = adam_bashforth(points, t0, h, steps, f, order)
            elif method[1] == 'multon':
                points = adam_multon(points, t0, h, steps, f, order)
            elif method[1] == 'inversa':
                points = formula_inversa(points, t0, h, steps, f, order)
        else:
            f = evalf(line[-1]); steps = int(line[-2]);
            h = float(line[-3]); t0 = float(line[-4]);
            y0 = float(line[-5])
            if method[-1] == 'euler':
                print('de Euler')
                points = euler([y0], t0, h, steps, f)
            elif method[-1] == 'inverso':
                print('de Euler Inverso')
                points = euler_inverso([y0], t0, h, steps, f)
            elif method[-1] == 'aprimorado':
                print('de Euler Aprimorado')
                points = euler_aprimorado([y0], t0, h, steps, f)
            elif method[-1] == 'kutta':
                print('de Runge-Kutta')
                points = runge_kutta([y0], t0, h, steps, f)
        print('y(', t0, ') =', points[0])
        print('h =', h)
        t_s = []; t_i = t0
        for i in range(0, len(points)):
            t_s.append(t_i + i * h)
            t_i += i * h
        for i in range(0, len(points)):
            print(i, points[i])
        plt.plot(t_s, points, 'ro')
        print(t_s)
        print(points)
        plt.axis([0, t_s[-1], 0, points[-1]])
        plt.show()
        print('')
    return 0

if __name__ == '__main__':
    main()
