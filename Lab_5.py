import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from functools import reduce

# return F, Psi, H, x_t0
def initVariables(tetta, mode):
    if mode == 2:
        F = np.array([[-0.8, 1.0], [tetta[0][0], 0]])
        Psi = np.array([[tetta[1][0]], [1.0]])
        H = np.array([[1.0, 0]])
        R = np.array([[0.1]])
        x0 = np.zeros((n, 1))
        u = np.zeros((N, 1))
        for i in range(N):
            u[i] = 1.0
    if mode == 1:
        F = np.array([[0]])
        Psi = np.array([[tetta[0][0], tetta[1][0]]])
        H = np.array([[1.0]])
        R = np.array([[0.3]])
        x0 = np.zeros((n, 1))
        u = np.array([[[2.], [1.]], [[1.], [2.]]])
    return F, Psi, H, R, x0, u

# return dF, dPsi, dH, dR, dx0, du_dua
def initGraduates(mode):
    if mode == 2:
        dF = np.array([np.array([[0, 0], [1, 0]]), np.array([[0, 0], [0, 0]])])
        dPsi = np.array([np.array([[0], [0]]), np.array([[1], [0]])])
        dH = np.array([np.array([[0, 0]]), np.array([[0, 0]])])
        dR = np.array([np.array([[0]]), np.array([[0]])])
        dx0 = np.array([np.zeros((n, 1)) for i in range(s)])
        du_dua = 1
    elif mode == 1:
        dF = np.array([np.array([[0]]), np.array([[0]])])
        dPsi = np.array([np.array([[1, 0]]), np.array([[0, 1]])])
        dH = np.array([np.array([[0]]), np.array([[0]])])
        dR = np.array([np.array([[0]]), np.array([[0]])])
        dx0 = np.array(np.zeros((n, 1)) for i in range(s))
        du_dua = np.array([[[1], [0]], [[0], [1]]])
    return dF, dPsi, dH, dR, dx0, du_dua


def MinimizeFirst(tettaMin, yRun, plan, k, v, m, R):
    res = []
    ############__2__##########
    lim = [-2.0, 0.01, -0.05, 1.5]
    bounds = Bounds([lim[0], lim[1]],  # [min x0, min x1]
                    [lim[2], lim[3]])  # [max x0, max x1]
    result = minimize(fun=Xi, x0=tettaMin, args={"y": yRun,"plan": plan, "k": k, "v": v, "m": m, "R": R}, jac=dXi,  method='SLSQP', bounds=bounds)
    # res.append(minimize(Xi, x_start, method='SLSQP', jac=dXi, bounds=bounds))
    # print("Тетты для нулевого порядка:", result.__getitem__("x"))
    print("Тетты для первого порядка:", result)
    return np.array(result.__getitem__("x")). reshape(2, 1)


def y(R, tetta, N, plan):
    mode = n
    F, Psi, H, o, xt0, l = initVariables(tetta, mode)
    xtk = 0
    yEnd = []
    q = len(plan)
    for _ in range(q):
        yPlus = []
        for stepj in range(N):
            if stepj == 0:
                xtk = np.array([[0], [0]])
            xPlus = np.add(np.dot(F, xtk), np.dot(Psi, plan[_][stepj][0]))
            yPlus.append(np.add(np.dot(H, xPlus)[0][0], (np.random.normal(0, 1.) * R)))
            xtk = xPlus
        yEnd.append(np.array(yPlus).reshape(N, 1))
    yEnd = np.array(yEnd)
    return yEnd

def Xi(tetta, params):
    # Инициализация матриц/векторов, 1, 2 пункты:
    y = params['y'].copy()
    plan = params['plan'].copy()
    k = params['k']
    v = params['v']
    m = params['m']
    R = params['R']

    tetta = np.array(tetta).reshape(2, 1)

    mode = n
    F, Psi, H, o, xt0, l = initVariables(tetta, mode)
    q = len(k)
    xtk = np.array([[np.full(shape=2, fill_value=0, dtype=float).reshape(2, 1) for stepj in range(N)] for stepi in range(q)])
    xi = N * m * v * math.log(2 * math.pi) + N * v * math.log(R)  # Calculate Xi constant
    # Point 4
    u = plan

    for kCount in range(0, N):
        Triangle = 0  # Инициализация треугольничка
        for i in range(0, q):

            # Point 5
            if kCount == 0:
                xtk[i][kCount] = xt0

            # Расчёт критерия индентификации для ложных тет
            xPlusOne = np.dot(F, xtk[i][kCount]) + np.dot(Psi, u[i][kCount][0])

            for j in range(int(k[i])):
                epsPlusOne = y[i][kCount][0] - np.dot(H, xPlusOne)
                Triangle += np.multiply(epsPlusOne.transpose(), epsPlusOne) * pow(R, -1)
            if kCount + 1 < N:
                xtk[i][kCount + 1] = xPlusOne
            else:
                xtk[i][kCount] = xPlusOne
        xi += Triangle

    return 0.5 * xi

def dXi(tetta, params):
    ###############___0.5____###################################
    mode = 2
    y = params['y'].copy()
    plan = params['plan'].copy()
    k = params['k']
    v = params['v']
    m = params['m']
    R = params['R']

    tetta = np.array(tetta).reshape(2, 1) # Разкомментируй, если функцию минимизации используешь
    # 1
    mode = n
    F, Psi, H, o, xt0, l = initVariables(tetta, mode=mode)
    dF, dPsi, dH, dR, dxt0, du_dua = initGraduates(mode=mode)

    # 2
    gradient = (np.array([v / 2. * N * 1. / R * dR[alfa] for alfa in range(s)])).reshape(s, 1)

    q = len(k)
    # Point 4
    u = plan
    # Point 5
    xtk = np.array([[np.full(shape=2, fill_value=0, dtype=float).reshape(2, 1) for stepj in range(N)] for stepi in range(q)])
    dxtk = np.array([[[np.full(shape=2, fill_value=0, dtype=float).reshape(2, 1) for stepj in range(N)] for stepi in range(q)] for alpha in range(s)])
    dxPlusOne = np.array([np.zeros((2, 1)) for alpha in range(s)])
    depsPlusOne = np.array([0 for alpha in range(s)])

    for kCount in range(0, N):
        Triangle = np.array([0 for alpa in range(s)])  # Инициализация треугольничка
        for i in range(0, q):

            # Point 5
            if kCount == 0:
                xtk[i][kCount] = xt0
                for alpha in range(s):
                    dxtk[alpha][i][kCount] = dxt0[alpha]
            # Point 6
            xPlusOne = np.dot(F, xtk[i][kCount]) + np.dot(Psi, u[i][kCount][0])

            # Point 7
            for alpha in range(s):
                dxPlusOne[alpha] = np.dot(dF[alpha], xtk[i][kCount]) + np.dot(F, dxtk[alpha][i][kCount]) + np.dot(dPsi[alpha], u[i][kCount][0])
                depsPlusOne[alpha] = (-1) * np.dot(dH[alpha], xtk[i][kCount]) - np.dot(H, dxPlusOne[alpha])

            for j in range(int(k[i])):
                epsPlusOne = y[i][kCount][0] - np.dot(H, xPlusOne)
                for alpha in range(s):
                    Triangle[alpha] += np.dot(np.dot(depsPlusOne[alpha].transpose(), pow(R, -1)), epsPlusOne) - \
                                       (0.5) * np.dot(np.dot(depsPlusOne[alpha].transpose(), pow(R, -1)), dR[alpha]) * \
                                       np.dot(pow(R, -1), epsPlusOne)
            if kCount + 1 < N:
                xtk[i][kCount + 1] = xPlusOne
                for alpha in range(s):
                    dxtk[alpha][i][kCount + 1] = dxPlusOne[alpha]
            else:
                xtk[i][kCount] = xPlusOne
                for alpha in range(s):
                    dxtk[alpha][i][kCount] = dxPlusOne[alpha]
        for alpha in range(s):
            gradient[alpha] += Triangle[alpha]


    return gradient

def KsiStart(q):
    ksi = []
    for stepi in range(q):
        Ui = []
        Ui = (np.full(shape=N, fill_value=1, dtype=float)).reshape(N, 1)
        ksi.append(Ui)
    return np.array(ksi)

###############################################################################################
def Ci(I, n, s):
    matrix_ci = np.zeros((n, n * (s + 1)))
    for _ in range(n):
        matrix_ci[_][I * n + _] = 1
    return matrix_ci


# Здесь можно проверять сразу основную лабу(mode=2) и 6 пункт из контрольных вопросов(mode=1). При смене mode не забывай менять n
def IMF(U, tetta):
    mode = n
    if mode == 1:
        F, Psi, H, R, x0, U = initVariables(tetta, mode=mode) # mode - задание, которое мы сейчас отрабатываем. 0 - обычная лаба(n==2), 1 - пункт 6 2 лр(n==1),
    elif mode == 2:
        F, Psi, H, R, x0, o = initVariables(tetta, mode=mode)

    dF, dPsi, dH, dR, dx0, du_dua = initGraduates(mode=mode)
    Fa = np.zeros((n * (s + 1), n * (s + 1)))



    ####################_____2 пункт____________##############
    # Заполняем матрицу Fa (вертикальные элементы)
    for i in range(n):
        for j in range(n):
            Fa[i][j] = F[i][j]

    for _ in range(s):
        for i in range(n):
            for j in range(n):
                Fa[_*n + n + i][j] = dF[_][i][j]

    # Заполняем матрицу Fa (диагональные элементы)
    for _ in range(s):
        for i in range(n):
            for j in range(n):
                Fa[_*n + n + i][_*n + n + j] = F[i][j]

    # Определение PsiA
    PsiA = np.zeros((n * (s + 1), len(Psi[0])))

    # Заполнение первых двух элементов PsiA значениями из Psi
    for i in range(len(Psi)):
        for j in range(len(Psi[0])):
            PsiA[i][j] = Psi[i][j]

    # Заполнение остальных элементов PsiA значениями из dPsi
    for _ in range(s):
        for i in range(len(Psi)):
            for j in range(len(Psi[0])):
                PsiA[_*n + n + i][j] = dPsi[_][i][j]

    ####################_____3 пункт____________##############
    # Инициализация Матрицы Фишера
    delta_M = np.zeros((2, 2))

    # Инициализация Xatk
    Xa_tk_plus_one = np.zeros((n * (s + 1), 1))
    Xa_tk = np.zeros((n * (s + 1), 1))
    M = N / 2. * (dR[0] * pow(R, -1) * dR[1] * pow(R, -1))


    ####################_____4, 5, 6 пункты____________##############
    # Расчёт матрицы Фишера
    for k in range(N):
        # Расчёт первого значения для Xa_tk, k == 0:

        if n >= 2:
            if k == 0:
                x_0 = np.matmul(F, x0) + np.dot(Psi, U[k][0])
                x_1 = np.matmul(dF[0], x0) + np.matmul(F, dx0[0]) + np.dot(dPsi[0], U[k][0])
                x_2 = np.matmul(dF[1], x0) + np.matmul(F, dx0[1]) + np.dot(dPsi[1], U[k][0])
                Xa_tk_plus_one = np.concatenate((x_0, x_1, x_2))

            # Расчёт последующих значений для Xa_tk, k > 0:
            if k > 0:
                Xa_tk_plus_one = np.add(np.matmul(Fa, Xa_tk), PsiA * U[k][0])
            # print("\nXa_tk_plus_one\n", Xa_tk_plus_one)
        elif n == 1:
            if k == 0:
                x_0 = np.matmul(F, x0) + np.dot(Psi, U[k])
                x_1 = np.matmul(dF[0], x0) + np.matmul(F, dx0[0]) + np.dot(dPsi[0], U[k])
                x_2 = np.matmul(dF[1], x0) + np.matmul(F, dx0[1]) + np.dot(dPsi[1], U[k])
                Xa_tk_plus_one = np.concatenate((x_0, x_1, x_2))
                # print(Xa_tk_plus_one)

            # Расчёт последующих значений для Xa_tk, k > 0:
            if k > 0:
                Xa_tk_plus_one = np.matmul(Fa, Xa_tk) + np.dot(PsiA, U[k])
                Xa_tk = Xa_tk_plus_one
        Xa_tk = Xa_tk_plus_one

        for i in range(2):
            for j in range(2):
                A0 = reduce(np.dot, [dH[i], Ci(I=0, n=n, s=s),
                                     Xa_tk_plus_one, Xa_tk_plus_one.transpose(),
                                     Ci(I=0, n=n, s=s).transpose(), dH[j].transpose(), pow(R, -1)])
                # print("A0_M2", A0)

                A1 = reduce(np.dot, [dH[i], Ci(I=0, n=n, s=s),
                                     Xa_tk_plus_one, Xa_tk_plus_one.transpose(),
                                      Ci(I=j + 1, n=n, s=s).transpose(), H.transpose(), pow(R, -1)])
                # print("A1_M2", A1)

                A2 = reduce(np.dot, [H, Ci(I=i + 1, n=n, s=s),
                                     Xa_tk_plus_one, Xa_tk_plus_one.transpose(),
                                     Ci(I=0, n=n, s=s).transpose(), dH[j].transpose(), pow(R, -1)])
                # print("A2_M2", A2)

                A3 = reduce(np.dot, [H, Ci(I=i + 1, n=n, s=s),
                                     Xa_tk_plus_one, Xa_tk_plus_one.transpose(),
                                     Ci(I=j + 1, n=n, s=s).transpose(), H.transpose(), pow(R, -1)])
                # print("A3_M2", A3)
                delta_M[i][j] = A0 + A1 + A2 + A3
        M = np.add(M, delta_M)
    return M


def dIMF(U, tetta):
    ####################_____1 пункт____________##############
    mode = n
    if mode == 1:
        F, Psi, H, R, x0, U = initVariables(tetta, mode=mode)  # mode - задание, которое мы сейчас отрабатываем. 0 - обычная лаба(n==2), 1 - пункт 6 2 лр(n==1),
    elif mode == 2:
        F, Psi, H, R, x0, o = initVariables(tetta, mode=mode)

    dF, dPsi, dH, dR, dx0, du_dua = initGraduates(mode=mode)
    Fa = np.zeros((n * (s + 1), n * (s + 1)))

    # Заполняем матрицу Fa (вертикальные элементы)
    for i in range(n):
        for j in range(n):
            Fa[i][j] = F[i][j]

    for _ in range(s):
        for i in range(n):
            for j in range(n):
                Fa[_ * n + n + i][j] = dF[_][i][j]

    # Заполняем матрицу Fa (диагональные элементы)
    for _ in range(s):
        for i in range(n):
            for j in range(n):
                Fa[_ * n + n + i][_ * n + n + j] = F[i][j]

    # Определение PsiA
    PsiA = np.zeros((n * (s + 1), len(Psi[0])))

    # Заполнение первых двух элементов PsiA значениями из Psi
    for i in range(len(Psi)):
        for j in range(len(Psi[0])):
            PsiA[i][j] = Psi[i][j]

    # Заполнение остальных элементов PsiA значениями из dPsi
    for _ in range(s):
        for i in range(len(Psi)):
            for j in range(len(Psi[0])):
                PsiA[_ * n + n + i][j] = dPsi[_][i][j]

    ####################_____2 пункт____________##############
    dMdu = [np.zeros((2, 2)) for alpha in range(r) for betta in range(N)]
    # Psi_du_dua_betta = [np.zeros((6, 1)) for alpha in range(r)]
    if mode == 2:
        dxa_tk_plus_one_dua_tbetta = [np.zeros((6, 1)) for alpha in range(r) for betta in range(N)]
        dxa_tk_dua_tbetta = [np.zeros((6, 1)) for alpha in range(r) for betta in range(N)]
    elif mode == 1:
        dxa_tk_plus_one_dua_tbetta = [np.zeros((3, 1)) for alpha in range(r) for betta in range(N)]
        dxa_tk_dua_tbetta = [np.zeros((3, 1)) for alpha in range(r) for betta in range(N)]

    delta_M = np.zeros((2, 2))

    # Инициализация Xatk
    Xa_tk_plus_one = np.zeros((n * (s + 1), 1))
    Xa_tk = np.zeros((n * (s + 1), 1))

    # Определяю размерности для Psi_du_dua_betta dxa_tk_dua_tbetta
    Psi_du_dua_betta = PsiA

    for k in range(N):
        count = 0
        ####################_____4 пункт____________##############
        if n == 2:
            if k == 0:
                x_0 = np.matmul(F, x0) + np.dot(Psi, U[k][0])
                x_1 = np.matmul(dF[0], x0) + np.matmul(F, dx0[0]) + np.dot(dPsi[0], U[k][0])
                x_2 = np.matmul(dF[1], x0) + np.matmul(F, dx0[1]) + np.dot(dPsi[1], U[k][0])
                Xa_tk_plus_one = np.concatenate((x_0, x_1, x_2))
                # print(Xa_tk_plus_one)

            # Расчёт последующих значений для Xa_tk, k > 0:
            if k > 0:
                Xa_tk_plus_one = np.matmul(Fa, Xa_tk) + np.dot(PsiA, U[k][0])

        elif n == 1:
            if k == 0:
                x_0 = np.matmul(F, x0) + np.dot(Psi, U[k])
                x_1 = np.matmul(dF[0], x0) + np.matmul(F, dx0[0]) + np.dot(dPsi[0], U[k])
                x_2 = np.matmul(dF[1], x0) + np.matmul(F, dx0[1]) + np.dot(dPsi[1], U[k])
                Xa_tk_plus_one = np.concatenate((x_0, x_1, x_2))
                # print(Xa_tk_plus_one)

            # Расчёт последующих значений для Xa_tk, k > 0:
            if k > 0:
                Xa_tk_plus_one = np.matmul(Fa, Xa_tk) + np.dot(PsiA, U[k])
        Xa_tk = Xa_tk_plus_one

        ####################_____5,6 пункт____________##############
        for betta in range(N):
            for alpha in range(r):
                if mode == 1:
                    if betta == k:
                        Psi_du_dua_betta = np.dot(PsiA, du_dua[alpha])
                    else:
                        Psi_du_dua_betta = np.dot(PsiA, [[0], [0]])
                elif mode == 2:
                    if betta == k:
                        Psi_du_dua_betta = PsiA
                    else:
                        Psi_du_dua_betta = np.dot(PsiA, [[0]])

                if k == 0:
                    dxa_tk_plus_one_dua_tbetta[betta] = Psi_du_dua_betta

                else:
                    dxa_tk_plus_one_dua_tbetta[betta] = np.add(np.dot(Fa, dxa_tk_dua_tbetta[betta]), Psi_du_dua_betta)
                dxa_tk_dua_tbetta[betta] = dxa_tk_plus_one_dua_tbetta[betta]

                coeff_dxa_tk_plus_one = np.add(np.dot(dxa_tk_plus_one_dua_tbetta[betta], Xa_tk_plus_one.transpose()),
                                       np.dot(Xa_tk_plus_one, dxa_tk_plus_one_dua_tbetta[betta].transpose()))

                for i in range(2):
                    for j in range(2):
                        A0 = reduce(np.dot, [dH[i], Ci(I=0,n=n,s=s),
                                    coeff_dxa_tk_plus_one,
                                     Ci(I=0,n=n,s=s).transpose(), dH[j].transpose(), pow(R, -1)])
                        # print("A0", A0)

                        A1 = reduce(np.dot, [dH[i], Ci(I=0,n=n,s=s),
                                    coeff_dxa_tk_plus_one,
                                    Ci(I=j + 1,n=n,s=s).transpose(), H.transpose(), pow(R, -1)])
                        # print("A1", A1)

                        A2 = reduce(np.dot, [H, Ci(I=i + 1,n=n,s=s),
                                    coeff_dxa_tk_plus_one,
                                    Ci(I=0,n=n,s=s).transpose(), dH[j].transpose(), pow(R, -1)])
                        # print("A2", A2)

                        A3 = reduce(np.dot, [H, Ci(I=i + 1,n=n,s=s),
                                    coeff_dxa_tk_plus_one,
                                    Ci(I=j + 1,n=n,s=s).transpose(), H.transpose(), pow(R, -1)])
                        delta_M[i][j] = A0 + A1 + A2 + A3
                dMdu[count] = np.add(dMdu[count], delta_M)
                count += 1

    # for i in range(r*N):
    #     print("\nM[", i, "]\n",dMdu[i])
    return dMdu

def LineForU(U):
    return U.reshape(N, )
def LineForKsi(Ksik):
    KsikLine = []
    for stepz in range(len(Ksik)):
        for stepj in range(len(Ksik[0])):
            for stepi in range(1):
                KsikLine.append(Ksik[stepz][stepj][stepi])
    return np.array(KsikLine)

def MatrixForPlan(U, p, tetta):
    # Принимает в себя U в матричном виде
    MatrixPlan = np.array([[0, 0], [0, 0]])
    for step in range(len(p)):
        MatrixPlan = p[step] * (np.add(MatrixPlan, IMF(U=U[step], tetta=tetta)))
    return MatrixPlan

def startPlan(tetta):
    q = int(1 + (s * (s + 1)) / 2)
    startMatrixPlan = np.array([[0, 0], [0, 0]])
    uDraft = np.zeros((1, N))
    uClean = [np.zeros((N, 1)) for stepj in range(q)]

    # Создание начального плана
    # Для этого мы генерируем матрицу значений U, а после её транспонируем, т.к
    # нам нужны столбцы-вектора U, а после генерации появляются не совсем они.

    p = np.array([0.25 for step in range(q)]).transpose()
    U_ = np.array([[np.random.uniform(0.001, 10.) for stepj in range(N)] for stepi in range(q)])
    for stepj in range(q):
        uDraft = U_[stepj]
        for stepi in range(N):
            uClean[stepj][stepi] = uDraft[stepi]
    U_ = uClean
    startMatrixPlan = MatrixForPlan(U_, p, tetta)
    return startMatrixPlan, U_, p
def NewPlan(tk, Ksik, Uk, p):
    ksikNew = np.hstack((Ksik, Uk))
    for i in range(len(p)):
        p[i] = (1. - tk) * p[i]
    pNew = np.hstack((p, tk))
    return ksikNew, pNew
def CleaningPlan(Ksik, p):
    # Функция получает Ksik в виде вектора*
    newp = [] # p, содержащая только веса, которые встречаются больше двух раз
    newKsik = []
    pDelPoint = []
    Q = len(p)

    Ksik = Ksik.reshape(Q, N, 1)
    for stepi in range(Q - 1):
        for stepj in range(stepi + 1, Q - stepi):
            if np.dot(Ksik[stepi].transpose(), Ksik[stepj]) <= delta:
                p[stepi] += p[stepj]
                pDelPoint.append(stepj)

    for stepi in range(Q):
        if p[stepi] < 0.001:
            pDelPoint.append(stepi)

    for stepi in range(Q):
        if stepi not in pDelPoint:
            newKsik.append(Ksik[stepi])
            newp.append(p[stepi])

    # Уравновешиваю веса, приводя их сумму к 1:
    pSum = sum(newp)
    if pSum != (1.0 - delta) or pSum != (1.0 + delta):
        for stepi in range(len(newp)):
            newp[stepi] *= (1.0 / pSum)

    newKsik = (np.array(newKsik)).reshape(len(newp), N, 1)
    newp = np.array(newp)
    return newKsik, newp

def AOptimality(U, Ksik, p, number, tetta):
    # Добавил к U reshape, т.к. функция минимизации превращает мою матрицу N на 1 в дурацкое (N, )
    Q = len(p)

    if (number == 1) or (number == 2):
        U = U.reshape(N, 1)
        Ksik = Ksik.reshape(Q, N, 1)
        if number == 1:
            return (-1) * (np.dot(pow(np.linalg.inv(MatrixForPlan(Ksik, p, tetta)), 2), IMF(U=U, tetta=tetta))).trace()
        return (np.dot(pow(np.linalg.inv(MatrixForPlan(Ksik, p,tetta)), 2), IMF(U=U, tetta=tetta))).trace()
    if number == 3:
        Ksik = U.reshape(Q, N, 1)
        return (np.linalg.inv(MatrixForPlan(Ksik, p, tetta))).trace()
    if number == 4:
        Ksik = U.reshape(Q, N, 1)
        return (np.linalg.inv(MatrixForPlan(Ksik, p, tetta))).trace()

def XMKsi(tk, params):
    # Принимает в себя Ksik, U в виде векторов (n, )
    Ksik = params['Ksik'].copy()
    Uk = params['U'].copy()
    pi = params['pi'].copy()
    tetta = params['tetta'].copy()

    KsikNew, pNew = NewPlan(tk, Ksik, Uk, pi)
    KsikNew = KsikNew.reshape(len(pNew), N, 1)
    return (np.linalg.inv(MatrixForPlan(KsikNew, pNew, tetta))).trace()

def dAOptimality(U, Ksik, p, number, tetta):
    dnu = [np.zeros((2, 2)) for betta in range(N)]
    Q = len(p)
    U = U.reshape((N, 1))
    Ksik = Ksik.reshape(Q, N, 1)
    dImf = dIMF(U, tetta_true)

    for stepi in range(N):
        dnu[stepi] = (-1) * (np.dot(pow(np.linalg.inv(MatrixForPlan(Ksik, p, tetta)), 2), dImf[stepi])).trace()
    return dnu
def dDOptimality(U, Ksik, p, tetta):
    # Здесь мне Кси нужен в виде матрицы
    return (np.dot(np.linalg.inv(MatrixForPlan(Ksik, p, tetta)), dIMF(U=U, tetta=tetta_true))).trace()
def Optimalitys(U, Ksik, p, number, tetta):
    # В этой функции использую вектора (n, ) для минимизации
    XM = ''
    Ksik = np.array(LineForKsi(Ksik))
    U = np.array(LineForU(U))

    print("\nmimimize...\n")
    # Расчёт ню для А-оптимальности, при этом number == 1!
    if number == 1:
        result = minimize(AOptimality, U, args=(Ksik, p, number, tetta, ), method='SLSQP', jac=dAOptimality, bounds=Bounds([0]*N, [10]*N))
        return result.__getitem__("x")

    # Расчёт nuUk, при этом number == 2!:
    if number == 2:
        return AOptimality(U=U, Ksik=Ksik, p=p, number=number, tetta=tetta)

    # Расчёт эты, при этом number == 3!:
    if number == 3:
        # print("\nKsik:\n", Ksik)
        return AOptimality(U=Ksik, Ksik=U, p=p, number=number, tetta=tetta)

    # Расчёт tk, при этом number == 4!:
    if number == 4:
        XM = minimize(XMKsi, x0=np.random.uniform(0, 1), args={"Ksik":Ksik, "U":U, "pi": p, "tetta": tetta}, method='SLSQP', bounds=Bounds(0, 1))
    return XM.__getitem__("x")

def ADPlan_third_lab(tetta):
    # 1 пункт:
    startMatrixPlan, Ksik, p = startPlan(tetta)
    count = 0
    print(startMatrixPlan)

    # 2 пункт:
    while count != 2:
        # Данный список мне необходим для запуска всех минимизаций, т.к. Эти самые минимизации требуют списки, размером (n, )
        while 1:
            U0 = np.array([[float(np.random.uniform(0.1, 10.)) for stepi in range(1)] for stepj in range(N)])
            if count == 0:
                print("\nA-Optimality start:\n", AOptimality(U=LineForKsi(Ksik), Ksik=U0, p=p, number=4, tetta=tetta))
                count += 1
            Uk = Optimalitys(U=U0, Ksik=Ksik, p=p, number=1, tetta=tetta)
            nuUk = Optimalitys(U=Uk, Ksik=Ksik, p=p, number=2, tetta=tetta)
            eta = Optimalitys(U=U0, Ksik=Ksik, p=p, number=3, tetta=tetta)
            # print("\nUk:\n", Uk,
            #       "\neta:\n", eta,
            #       "\nnuUk:\n", nuUk)
            if abs(nuUk - eta) <= delta:
                count = 2
                break
            elif nuUk > eta:
                print("\nLet's go to the third point!\n")
                break
            else:
                print("\nLet's go to the second point!\n")
        if count != 2:
        # Третий шаг:
            tk = Optimalitys(U0, Ksik, p, 4, tetta=tetta)

        # Четвёртый шаг, создаём новый план и производим его очистку:
            Ksik, p = NewPlan(tk, Ksik=LineForKsi(Ksik), Uk=Uk, p=p)
            Ksik, p = CleaningPlan(Ksik, p)
    print("\nPlan A is done!\n",
          "\nResults:\n", "\nKsik:\n", Ksik, "\np:\n", p)
    print("\nA-Optimality end:\n", AOptimality(U=Ksik, Ksik=U0, p=p, number=4, tetta=tetta))
    return Ksik, p

def rounding(pNew):
    sigmHatch, sigmTwiceHatch, vHatch, vTwiceHatch, sigm, v1, pointThree = [], [], 0, 0, [], 0, []
    q, v = len(pNew), 10
    for i in range(q):
        sigmHatch.append((v - q) * pNew[i])
        sigmTwiceHatch.append(int(v * pNew[i]))
    # print("\nsigmHatch:\n", sigmHatch, "\nsigmTwiceHatch:\n", sigmTwiceHatch)
    vHatch, vTwiceHatch  = v, v
    for i in range(q):
        vHatch -= sigmHatch[i]
        vTwiceHatch -= sigmTwiceHatch[i]

    # Point 2
    if vHatch < vTwiceHatch:
        for i in range(q):
            sigm.append(sigmHatch[i])
        v1 = vHatch
    else:
        for i in range(q):
            sigm.append(sigmTwiceHatch[i])
        v1 = vTwiceHatch

    # Point 3
    for i in range(q):
        pointThree.append(v * pNew[i] - sigm[i])
    pointThree = sorted(pointThree, reverse=True)
    # print("\npointThree:\n", pointThree, "\nsigm\n", sigm, "\nv1:\n", v1)

    s = np.zeros(q)
    for i in range(v1):
        for j in range(q):
            if pointThree[i] == (v * pNew[j] - sigm[j]):
                s[j] = 1
            else:
                s[j] = 0
    kNew = np.zeros(q)
    for i in range(q):
        kNew[i] = sigm[i] + s[i]
    return kNew, v

def main(tettaTrue, tettaFalse):
    R, m = 0.1, 1. # Nubmber of derivatives
    q = int(1 + s * (s + 1) / 2) # Number of k
    k = [1.0 for stepi in range(q)]  # Initial number of system start
    v = q

    startPlan = KsiStart(q)
    yRun = y(R, tettaTrue, N, startPlan)

    tettaNew = MinimizeFirst(tettaFalse, yRun, startPlan, k, v, m, R)
    newPlan, pNew = ADPlan_third_lab(tettaNew.reshape(2, 1))
    # 6 lr:
    k, v = rounding(pNew)
    yRun = y(R, tettaTrue, N, newPlan)
    tettaNew = MinimizeFirst(tettaNew, yRun, newPlan, k, v, m, R)

if __name__ == '__main__':
    # Определение переменных
    r = 2  # Количество начальных сигналов, альфа
    n = 2  # Размерность вектора х0
    s = 2  # Количество производных по тетта
    N = 4  # Число испытаний

    delta = 0.0001

    tetta_true = (np.array([-1.5, 1.0])).reshape(2, 1)
    tetta_false = (np.array([-2, 0.1])).reshape(2, 1)

    main(tetta_true, tetta_false)

