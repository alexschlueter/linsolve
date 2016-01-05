import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from timeit import Timer

def cholesky_decomp(A):
    L = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(i + 1):
            if i == j:
                rowsum = 0
                for k in range(j):
                    rowsum += L[j][k] * L[j][k]
                L[j][j] = np.sqrt(A[j][j] - rowsum)
            else:
                rowsum = 0
                for k in range(j):
                    rowsum += L[i][k] * L[j][k]
                L[i][j] =  1 / L[j][j] * (A[i][j] - rowsum)
    return L

def cholesky_solve(A, b):
    x = np.empty_like(b)
    L = cholesky_decomp(A)
    for i in range(b.shape[0]):
        rowsum = b[i]
        for j in range(i):
            rowsum -= L[i][j] * x[j]
        x[i] = 1 / L[i][i] * rowsum
    for i in reversed(range(b.shape[0])):
        rowsum = x[i]
        for j in range(i + 1, b.shape[0]):
            rowsum -= L[j][i] * x[j]
        x[i] = 1 / L[i][i] * rowsum
    return x

def richardson(A, b, emin, emax, sol):
    x = np.zeros(b.shape[0])
    tau = 2 / (emin + emax)
    res = b - A.dot(x)
    while True:
        x = x + tau * res
        res = b - A.dot(x)
        if np.linalg.norm(x - sol)/np.linalg.norm(sol) <= 1e-6:
            return x

def conjugate_gradient(A, b, sol):
    x = np.zeros(b.shape[0])
    r = b - A.dot(x)
    t = r
    while True:
        ralt = r
        alpha = r.dot(r) / t.dot(A.dot(t))
        x = x + alpha * t
        r = ralt - alpha * A.dot(t)
        if np.linalg.norm(x - sol)/np.linalg.norm(sol) <= 1e-6:
            return x
        beta = r.dot(r) / ralt.dot(ralt)
        t = r + beta * t

def data_gen(ITERATIONS, STEP):
    Q = np.random.rand(ITERATIONS,ITERATIONS)
    D = np.diag(np.random.rand(ITERATIONS))
    A = Q.transpose().dot(D).dot(Q)
    x = np.random.rand(ITERATIONS)
    n = 1
    for n in range(1, ITERATIONS + 1, STEP):
        M = A[:n, :n]
        d = x[:n]
        c = M.dot(d)
        eigs = np.linalg.eigvals(M)
        rich, chol, con = [np.empty(n)], [np.empty(n)], [np.empty(n)]
        def runrich():
            rich[0] = richardson(M, c, np.amin(eigs), np.amax(eigs), d)
        def runchol():
            chol[0] = cholesky_solve(M, c)
        def runcon():
            con[0] = conjugate_gradient(M, c, d)
        sim = Timer(runrich).timeit(number=1)
        tim = Timer(runchol).timeit(number=1)
        qim = Timer(runcon).timeit(number=1)
        richres = np.linalg.norm(d - rich[0])
        cholres = np.linalg.norm(d - chol[0])
        conres = np.linalg.norm(d - con[0])
        yield (sim, tim, qim, richres, cholres, conres)

def main():
    ITERATIONS = 500
    STEP = 10
    fig, ax = plt.subplots()
    Lnrich, = ax.plot([], 'r')
    Lnchol, = ax.plot([], 'b')
    Lncon, = ax.plot([], 'g')
    s = []
    t = []
    q = []
    txt = ax.text(0.05, 0.95, "", transform=ax.transAxes, ha="left", va="top", family="monospace")
    ax.set_ylabel("Zeit")
    ax.set_xlabel("Dimensionen")
    def update(data):
        s.append(data[0])
        t.append(data[1])
        q.append(data[2])
        Lnrich.set_xdata(range(1, STEP * len(s), STEP))
        Lnrich.set_ydata(s)
        Lnchol.set_xdata(range(1, STEP * len(t), STEP))
        Lnchol.set_ydata(t)
        Lncon.set_xdata(range(1, STEP * len(q), STEP))
        Lncon.set_ydata(q)
        txt.set_text("Abs. Fehler:\nRichardson: " + str(data[3])
                     + "\nCholesky:   " + str(data[4]) + "\nConjGrad:   " + str(data[5]))
        ax.relim()
        ax.autoscale_view(True,True,True)

    ani = anim.FuncAnimation(fig, update, data_gen(ITERATIONS, STEP), interval=1)
    plt.show()

if __name__ == '__main__':
    main()
