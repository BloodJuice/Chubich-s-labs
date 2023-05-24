from functools import reduce
import numpy as np
from scipy.optimize import minimize, Bounds
from collections import Counter

# return matrix_ci
def Ci(I, n, s):
    matrix_ci = np.zeros((n, n * (s + 1)))
    for _ in range(n):
        matrix_ci[_][I * n + _] = 1
    return matrix_ci

# return F, Psi, H, R, x0, u
def initVariables(tetta, mode):
    if mode == 2:
        F = np.array([[-0.8, 1.0], [tetta[0], 0]])
        Psi = np.array([[tetta[1]], [1.0]])
        H = np.array([[1.0, 0]])
        R = np.array([[0.1]])
        x0 = np.zeros((n, 1))
        return F, Psi, H, R, x0

    if mode == 1:
        F = np.array([[0]])
        Psi = np.array([[tetta[0], tetta[1]]])
        H = np.array([[1.0]])
        R = np.array([[0.3]])
        x0 = np.zeros((n, 1))
        u = np.array([[[2], [1]], [[1], [2]]])
        return F, Psi, H, R, x0, u


# return dF, dPsi, dH, dR, dx0
def initGraduates(mode):
    if mode == 2:
        dF = [np.array([[0, 0], [1, 0]]), np.array([[0, 0], [0, 0]])]
        dPsi = [np.array([[0], [0]]), np.array([[1], [0]])]
        dH = [np.array([[0, 0]]), np.array([[0, 0]])]
        dR = [np.array([[0]]), np.array([[0]])]
        dx0 = [np.zeros((n, 1)) for i in range(s)]
        du_dua = 1
    elif mode == 1:
        dF = [np.array([[0]]), np.array([[0]])]
        dPsi = [np.array([[1, 0]]), np.array([[0, 1]])]
        dH = [np.array([[0]]), np.array([[0]])]
        dR = [np.array([[0]]), np.array([[0]])]
        dx0 = [np.zeros((n, 1)) for i in range(s)]
        du_dua = np.array([[[1], [0]], [[0], [1]]])
    return dF, dPsi, dH, dR, dx0, du_dua


# Здесь можно проверять сразу основную лабу(mode=2) и 6 пункт из контрольных вопросов(mode=1). При смене mode не забывай менять n
def IMF(U, tetta):
    mode = n
    if mode == 1:
        F, Psi, H, R, x0, U = initVariables(tetta, mode=mode) # mode - задание, которое мы сейчас отрабатываем. 0 - обычная лаба(n==2), 1 - пункт 6 2 лр(n==1),
    elif mode == 2:
        F, Psi, H, R, x0 = initVariables(tetta, mode=mode)

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
        # print("\nM:\n", M)
    # print("\nResult:\n", M)
    return M


def dIMF(U, tetta):
    ####################_____1 пункт____________##############
    mode = n
    if mode == 1:
        F, Psi, H, R, x0, U = initVariables(tetta, mode=mode)  # mode - задание, которое мы сейчас отрабатываем. 0 - обычная лаба(n==2), 1 - пункт 6 2 лр(n==1),
    elif mode == 2:
        F, Psi, H, R, x0 = initVariables(tetta, mode=mode)

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

def MatrixForPlan(U, p):
    MatrixPlan = np.array([[0, 0], [0, 0]])
    for step in range(len(p)):
        MatrixPlan = p[step] * (np.add(MatrixPlan, IMF(U=U[step], tetta=tetta_true)))
    return MatrixPlan

def startPlan():
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
    startMatrixPlan = MatrixForPlan(U_, p)
    return startMatrixPlan, U_, p
def NewPlan(tk, Ksik, Uk, p):
    ksikNew = np.hstack((Ksik, Uk))
    for i in range(len(p)):
        p[i] = (1 - tk) * p[i]
    pNew = np.hstack((p, tk))
    return ksikNew, pNew
def CleaningPlan(Ksik, p):
    newp = [] # p, содержащая только веса, которые встречаются больше двух раз
    newKsik = []
    Q = len(p)
    print("\np:\n", p)
    Ksik = Ksik.reshape(Q, N, 1)
    pDelPoint = []

    for stepi in range(Q - 1):
        for stepj in range(stepi + 1, Q - stepi):
            if np.dot(Ksik[stepi].transpose(), Ksik[stepj]) <= delta:
                p[stepi] += p[stepj]
                pDelPoint.append(stepj)

    for stepi in range(Q):
        if p[stepi] < delta:
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

    newKsik = np.array(newKsik)
    newp = np.array(newp)
    return newKsik, newp

def AOptimality(U, Ksik, p, number):
    # Добавил к U reshape, т.к. функция минимизации превращает мою матрицу N на 1 в дурацкое (N, )
    Q = len(p)
    if (number == 1) or (number == 2):
        U = U.reshape(N, 1)
        Ksik = Ksik.reshape(Q, N, 1)
        if number == 1:
            return (-1) * (np.dot(pow(np.linalg.inv(MatrixForPlan(Ksik, p)), 2), IMF(U=U, tetta=tetta_true))).trace()
        return (np.dot(pow(np.linalg.inv(MatrixForPlan(Ksik, p)), 2), IMF(U=U, tetta=tetta_true))).trace()
    if number == 3:
        Ksik = U.reshape(Q, N, 1)
        return (np.linalg.inv(MatrixForPlan(Ksik, p))).trace()
    if number == 4:
        Ksik = U.reshape(Q, N, 1)
        return (np.linalg.inv(MatrixForPlan(Ksik, p))).trace()

def dAOptimality(U, Ksik, p, number):
    dnu = [np.zeros((2, 2)) for betta in range(N)]
    Q = len(p)
    U = U.reshape((N, 1))
    Ksik = Ksik.reshape(Q, N, 1)
    dImf = dIMF(U, tetta_true)

    for stepi in range(N):
        dnu[stepi] = (np.dot(pow(np.linalg.inv(MatrixForPlan(Ksik, p)), 2), dImf[stepi])).trace()
    return dnu

def Optimalitys(U, Ksik, p, number, lenQ):
    XMKsi = ''
    print("\nmimimize...\n")
    # Расчёт ню для А-оптимальности, при этом number == 1!
    if number == 1:
        result = minimize(AOptimality, U, args=(Ksik, p, number, ), method='SLSQP', jac=dAOptimality, bounds=Bounds(0, 10))
        # print("\nresult:\n", result)
        return result.__getitem__("x")

    # Расчёт nuUk, при этом number == 2!:
    if number == 2:
        return AOptimality(U=U, Ksik=Ksik, p=p, number=number)

    # Расчёт эты, при этом number == 3!:
    if number == 3:
        # print("\nKsik:\n", Ksik)
        return AOptimality(U=Ksik, Ksik=U, p=p, number=number)

    # Расчёт tk, при этом number == 4!:
    if number == 4:
        XMKsi = minimize(AOptimality, x0=Ksik, args=(U, p, number, ), method='cobyla')
        # print("\nXMKsi:\n", XMKsi)
    return AOptimality(U=XMKsi.__getitem__('x'), Ksik=U, p=p, number=number)




def ADPlan_third_lab():
    # 1 пункт:
    startMatrixPlan, Ksik, p = startPlan()
    count = 0
    # 2 пункт:
    while count != 2:
        # Данный список мне необходим для запуска всех минимизаций, т.к. Эти самые минимизации требуют списки, размером (n, )
        KsikLine = []
        for stepz in range(len(Ksik)):
            for stepj in range(len(Ksik[0])):
                for stepi in range(1):
                    KsikLine.append(Ksik[stepz][stepj][stepi])
        KsikLine = np.array(KsikLine)
        print("\nKsikLine:\n", KsikLine,
              "\np:\n", p)
        while 1:
            U0 = np.array([[float(np.random.uniform(0.1, 10.)) for stepi in range(1)] for stepj in range(N)])
            if count == 0:
                print("\nA-Optimality start:\n", Optimalitys(U=U0, Ksik=KsikLine, p=p, number=4, lenQ=len(p)))
            Uk = Optimalitys(U=U0, Ksik=KsikLine, p=p, number=1, lenQ=len(p))
            nuUk = Optimalitys(U=Uk, Ksik=KsikLine, p=p, number=2, lenQ=len(p))
            eta = Optimalitys(U=U0, Ksik=KsikLine, p=p, number=3, lenQ=len(p))
            print("\nUk:\n", Uk,
                  "\neta:\n", eta,
                  "\nnuUk:\n", nuUk)
            if abs(nuUk - eta) <= delta:
                count = 2
                break
            elif nuUk > eta:
                print("\nLet's go to the third point!\n")
                break
            else:
                print("\nLet's go to the second point!\n")
        if count == 0:
            KsikLine, p = NewPlan(Optimalitys(U=Uk, Ksik=KsikLine, p=p, number=4, lenQ=len(p)), Ksik=KsikLine, Uk=Uk, p=p)
            count += 1
        if count != 2:
            tk = Optimalitys(U=Uk, Ksik=KsikLine, p=p, number=4, lenQ=len(p))
            Ksik, p = NewPlan(tk, Ksik=KsikLine, Uk=Uk, p=p)
            Ksik, p = CleaningPlan(Ksik, p)
    print("\nPlan A is done!\n",
          "\nResults:\n", "\nKsik:\n", Ksik, "\np:\n", p)
    print("\nA-Optimality end:\n", Optimalitys(U=U0, Ksik=KsikLine, p=p, number=4, lenQ=len(p)))

if __name__ == '__main__':
    # Определение переменных
    m = v = nu = 1

    r = 1 # Количество начальных сигналов, альфа
    n = 2 # Размерность вектора х0
    s = 2 # Количество производных по тетта
    N = 4 # Число испытаний

    q = int(1 + (s*(s + 1)) / 2)
    delta = 0.001

    tetta_true = np.array([-1.5, 1.0])
    tetta_false = np.array([-2, 0.01])

    ADPlan_third_lab()



