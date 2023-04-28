from functools import reduce
import numpy as np


# return matrix_ci
def Ci(I, n, s):
    matrix_ci = np.zeros((n, n * (s + 1)))
    for _ in range(n):
        matrix_ci[_][I * n + _] = 1
    return matrix_ci

# return F, Psi, H, R, x0, u
def initVariables(tetta, mode):
    if mode == 0:
        F = np.array([[-0.8, 1.0], [tetta[0], 0]])
        Psi = np.array([[tetta[1]], [1.0]])
        H = np.array([[1.0, 0]])
        R = np.array([[0.1]])
        x0 = np.zeros((n, 1))
        u = 1.0
    if mode == 1:
        F = np.array([[0]])
        Psi = np.array([[tetta[0], tetta[1]]])
        H = np.array([[1.0]])
        R = np.array([[0.3]])
        x0 = np.zeros((n, 1))
        u = np.array([[2, 1], [1, 2]])
    return F, Psi, H, R, x0, u


# return dF, dPsi, dH, dR, dx0
def initGraduates(mode):
    if mode == 0:
        dF = [np.array([[0, 0], [1, 0]]), np.array([[0, 0], [0, 0]])]
        dPsi = [np.array([[0], [0]]), np.array([[1], [0]])]
        dH = [np.array([[0, 0]]), np.array([[0, 0]])]
        dR = [np.array([[0]]), np.array([[0]])]
        dx0 = [np.zeros((n, 1)) for i in range(s)]
    elif mode == 1:
        dF = [np.array([[0]]), np.array([[0]])]
        dPsi = [np.array([[1, 0]]), np.array([[0, 1]])]
        dH = [np.array([[0]]), np.array([[0]])]
        dR = [np.array([[0]]), np.array([[0]])]
        dx0 = [np.zeros((n, 1)) for i in range(s)]
    return dF, dPsi, dH, dR, dx0


# Здесь можно проверять сразу основную лабу(mode=0) и 6 пункт из контрольных вопросов(mode=1). При смене mode не забывай менять n
def IMF(tetta):
    mode = 0
    F, Psi, H, R, x0, u = initVariables(tetta, mode=mode) # mode - задание, которое мы сейчас отрабатываем. 0 - обычная лаба(n==2), 1 - пункт 6 2 лр(n==1),
    dF, dPsi, dH, dR, dx0 = initGraduates(mode=mode)
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
    M = N / 2. * (dR[0] * pow(R, -1) * dR[1] * pow(R, -1))
    delta_M = np.zeros((2, 2))

    # Инициализация Xatk
    Xa_tk_plus_one = np.zeros((n * (s + 1), 1))
    Xa_tk = np.zeros((n * (s + 1), 1))


    ####################_____4, 5, 6 пункты____________##############
    # Расчёт матрицы Фишера
    for _ in range(N):
        # Расчёт первого значения для Xa_tk, k == 0:
        if _ == 0:
            x0 = np.matmul(F, x0) + np.dot(Psi, u)
            x1 = np.matmul(dF[0], x0) + np.matmul(F, dx0[0]) + np.dot(dPsi[0], u)
            x2 = np.matmul(dF[1], x0) + np.matmul(F, dx0[1]) + np.dot(dPsi[1], u)
            Xa_tk_plus_one = np.concatenate((x0, x1, x2))
            # print(Xa_tk_plus_one)

        # Расчёт последующих значений для Xa_tk, k > 0:
        if _ > 0:
            Xa_tk_plus_one = np.matmul(Fa, Xa_tk) + np.dot(PsiA, u)
            Xa_tk = Xa_tk_plus_one
        # print("\nXa_tk_plus_one\n", Xa_tk_plus_one)

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
    print(M)


#
# def dIMF(tetta):
#     dF_dtetta = np.array([[0, 0], [1, 0],   # dF_dtetta_1
#                           [0, 0], [0, 0]])  # dF_dtetta_2
#     dF_dtetta = dF_dtetta.reshape(2, 2, 2)
#
#     dPsi_dtetta = np.array([[0, 0],         # dPsi_dtetta_1
#                            [1, 0]])         # dPsi_dtetta_2
#     dPsi_dtetta = dPsi_dtetta.reshape(2, 2, 1)
#
#
#     dH_dtetta = np.array([[0, 0],           # dH_dtetta_1 [1][2][1]
#                           [0, 0]])          # dH_dtetta_2 [2][2][1]
#     dH_dtetta = dH_dtetta.reshape(2, 1, 2)
#
#     dR_dtetta = np.array([0,                # dR_dtetta_1 [1][1]
#                           0])               # dR_dtetta_2 [2][1]
#
#     dxa_tk_dua_tbetta = []
#
#     C = np.array([[[1, 0, 0, 0, 0, 0],     # C0
#                     [0, 1, 0, 0, 0, 0]],
#                    [[0, 0, 1, 0, 0, 0],     # C1
#                     [0, 0, 0, 1, 0, 0]],
#                    [[0, 0, 0, 0, 1, 0],     # C2
#                     [0, 0, 0, 0, 0, 1]]])
#
#     F = np.array([[-0.8, 1.0], [tetta[0], 0]])
#     Psi = np.array([tetta[1], 1.0])
#     Psi = Psi.reshape(2, 1)
#
#     H = np.array([1.0, 0])
#     H = H.reshape(1, 2)
#
#     Fa = np.array([[-0.8, 1, 0, 0, 0, 0],
#                 [tetta[0], 0, 0, 0, 0, 0],
#                 [0, 0, -0.8, 1, 0, 0],
#                [1, 0, tetta[0], 0, 0, 0],
#                [0, 0, 0, 0, -0.8, 1],
#                [0, 0, 0, 0, tetta[0], 0]])
#
#     Psi_a = np.array([tetta[1], 1, 0, 0, 1, 0])
#     Psi_a = Psi_a.reshape(6, 1)
#
#     dxt0_dtetta = np.array([             #Размерность:   [2][2][1]
#                             [0, 0],  # dxt0_dtetta_1 [1][2][1]
#                             [0, 0]])   # dxt0_dtetta_2 [2][2][1]
#     dxt0_dtetta = dxt0_dtetta.reshape(2, 2, 1)
#
#     x_t0 = np.array([0, 0])
#     x_t0 = x_t0.reshape(2, 1)
#
#     # Расчёт M(tetta)
#     Xa_tk_plus_one = []
#     Xa_tk = []
#     delta_M = np.zeros((2, 2))
#
#     x0 = [0, 0]
#     x1 = [0, 0]
#     x2 = [0, 0]
#
#
#     for k in range(N):
#         M = np.zeros((2, 2))
#         # Расчёт первого значения для Xa_tk, k == 0:
#         if k == 0:
#             x0 = np.matmul(F, x_t0) + np.dot(Psi, u_t0)
#             x1 = np.matmul(dF_dtetta[0], x_t0) + np.matmul(F, dxt0_dtetta[0]) + np.dot(dPsi_dtetta[0], u_t0)
#             x2 = np.matmul(dF_dtetta[1], x_t0) + np.matmul(F, dxt0_dtetta[1]) + np.dot(dPsi_dtetta[1], u_t0)
#             Xa_tk_plus_one = np.concatenate((x0, x1, x2))
#             # print("Xa_tk_plus_one:\n", Xa_tk_plus_one)
#
#         # Расчёт последующих значений для Xa_tk, k > 0:
#         if k > 0:
#             Xa_tk_plus_one = np.matmul(Fa, Xa_tk) + np.dot(Psi_a, u_t0)
#         Xa_tk = Xa_tk_plus_one
#
#         for betta in range(N):
#             if betta == k:
#                 Psi_a_du_tk_dua_tbetta = Psi_a
#             else:
#                 Psi_a_du_tk_dua_tbetta = np.dot(Psi_a, 0)
#
#             if k == 0:
#                 dxa_tk_plus_one_dua_tbetta = Psi_a_du_tk_dua_tbetta
#             else:
#                 dxa_tk_plus_one_dua_tbetta = np.add(np.dot(Fa, dxa_tk_dua_tbetta), Psi_a_du_tk_dua_tbetta)
#             dxa_tk_dua_tbetta = dxa_tk_plus_one_dua_tbetta
#
#
#             coeff_dxa_tk_plus_one = np.add(np.dot(dxa_tk_plus_one_dua_tbetta, Xa_tk_plus_one.transpose()),
#                                            np.dot(Xa_tk_plus_one, dxa_tk_plus_one_dua_tbetta.transpose()))
#             # if betta == 1:
#                 # print("dxa_tk_plus_one_dua_tbetta\n", dxa_tk_plus_one_dua_tbetta,
#                 #     "\nPsi_a_du_tk_dua_tbetta\n", Psi_a_du_tk_dua_tbetta,
#                 #       "\ncoeff_dxa_tk_plus_one\n", coeff_dxa_tk_plus_one)
#
#             for i in range(2):
#                 for j in range(2):
#                     A0 = reduce(np.dot, [dH_dtetta[i], C[0],
#                                 coeff_dxa_tk_plus_one,
#                                 C[0].transpose(), dH_dtetta[j].transpose(), pow(R, -1)])
#                     # print("A0", A0)
#
#                     A1 = reduce(np.dot, [dH_dtetta[i], C[0],
#                                 coeff_dxa_tk_plus_one,
#                                 C[j + 1].transpose(), H.transpose(), pow(R, -1)])
#                     # print("A1", A1)
#
#                     A2 = reduce(np.dot, [H, C[i + 1],
#                                 coeff_dxa_tk_plus_one,
#                                 C[0].transpose(), dH_dtetta[j].transpose(), pow(R, -1)])
#                     # print("A2", A2)
#
#                     A3 = reduce(np.dot, [H, C[i + 1],
#                                 coeff_dxa_tk_plus_one,
#                                 C[j + 1].transpose(), H.transpose(), pow(R, -1)])
#                     delta_M[i][j] = A0 + A1 + A2 + A3
#
#             M = np.add(M, delta_M)
#         if k == (N - 1):
#             print("Мартица для k =", k, " равна =", "\n", M)
#
# def dIMF_test(u, tetta):
#     dF_dtetta = np.array([[0, 0], [1, 0],  # dF_dtetta_1
#                           [0, 0], [0, 0]])  # dF_dtetta_2
#     dF_dtetta = dF_dtetta.reshape(2, 2, 2)
#
#     dPsi_dtetta = np.array([[0, 0],  # dPsi_dtetta_1
#                             [1, 0]])  # dPsi_dtetta_2
#     dPsi_dtetta = dPsi_dtetta.reshape(2, 2, 1)
#
#     dH_dtetta = np.array([[0, 0],  # dH_dtetta_1 [1][2][1]
#                           [0, 0]])  # dH_dtetta_2 [2][2][1]
#     dH_dtetta = dH_dtetta.reshape(2, 1, 2)
#
#     dR_dtetta = np.array([0,  # dR_dtetta_1
#                           0])  # dR_dtetta_2
#
#     dxt0_dtetta = np.array([  # Размерность:   [2][2][1]
#                             [0, 0],  # dxt0_dtetta_1 [1][2][1]
#                             [0, 0]])  # dxt0_dtetta_2 [2][2][1]
#     dxt0_dtetta = dxt0_dtetta.reshape(2, 2, 1)
#
#     C = np.array([[[1, 0, 0, 0, 0, 0],  # C0
#                    [0, 1, 0, 0, 0, 0]],
#                   [[0, 0, 1, 0, 0, 0],  # C1
#                    [0, 0, 0, 1, 0, 0]],
#                   [[0, 0, 0, 0, 1, 0],  # C2
#                    [0, 0, 0, 0, 0, 1]]])
#
#     F = np.array([[-0.8, 1.0], [tetta[0], 0]])
#     Psi = np.array([tetta[1], 1.0])
#     Psi = Psi.reshape(2, 1)
#
#     H = np.array([1.0, 0])
#     H = H.reshape(1, 2)
#
#     x_t0 = np.array([0, 0])
#     x_t0 = x_t0.reshape(2, 1)
#
#     Fa = np.array([[-0.8, 1, 0, 0, 0, 0],
#                    [tetta[0], 0, 0, 0, 0, 0],
#                    [0, 0, -0.8, 1, 0, 0],
#                    [1, 0, tetta[0], 0, 0, 0],
#                    [0, 0, 0, 0, -0.8, 1],
#                    [0, 0, 0, 0, tetta[0], 0]])
#
#     Psi_a = np.array([tetta[1], 1, 0, 0, 1, 0])
#     Psi_a = Psi_a.reshape(6, 1)
#
#
#
#     # Расчёт M(tetta)
#     Xa_tk_plus_one = []
#     Xa_tk = []
#     delta_M = np.zeros((2, 2))
#
#     # Объявляю компоненты производной
#     M1, M2 = np.zeros((2, 2))
#     x0 = [0, 0]
#     x1 = [0, 0]
#     x2 = [0, 0]
#
#     for w in range(2):
#         M = np.array([[0, 0], [0, 0]])
#
#         for _ in range(N):
#             # Расчёт первого значения для Xa_tk, k == 0:
#             if _ == 0:
#                 x0 = np.matmul(F, x_t0) + np.dot(Psi, u)
#                 x1 = np.matmul(dF_dtetta[0], x_t0) + np.matmul(F, dxt0_dtetta[0]) + np.dot(dPsi_dtetta[0], u)
#                 x2 = np.matmul(dF_dtetta[1], x_t0) + np.matmul(F, dxt0_dtetta[1]) + np.dot(dPsi_dtetta[1], u)
#                 Xa_tk_plus_one = np.concatenate((x0, x1, x2))
#                 # print(Xa_tk_plus_one)
#
#             # Расчёт последующих значений для Xa_tk, k > 0:
#             if _ > 0:
#                 Xa_tk_plus_one = np.matmul(Fa, Xa_tk) + np.dot(Psi_a, u)
#             Xa_tk = Xa_tk_plus_one
#             # print("\nXa_tk_plus_one\n", Xa_tk_plus_one)
#
#             for i in range(2):
#                 for j in range(2):
#                     A0 = reduce(np.dot, [dH_dtetta[i], C[0],
#                                          Xa_tk_plus_one, Xa_tk_plus_one.transpose(),
#                                          C[0].transpose(), dH_dtetta[j].transpose(), pow(R, -1)])
#                     # print("A0_M2", A0)
#
#                     A1 = reduce(np.dot, [dH_dtetta[i], C[0],
#                                          Xa_tk_plus_one, Xa_tk_plus_one.transpose(),
#                                          C[j + 1].transpose(), H.transpose(), pow(R, -1)])
#                     # print("A1_M2", A1)
#
#                     A2 = reduce(np.dot, [H, C[i + 1],
#                                          Xa_tk_plus_one, Xa_tk_plus_one.transpose(),
#                                          C[0].transpose(), dH_dtetta[j].transpose(), pow(R, -1)])
#                     # print("A2_M2", A2)
#
#                     A3 = reduce(np.dot, [H, C[i + 1],
#                                          Xa_tk_plus_one, Xa_tk_plus_one.transpose(),
#                                          C[j + 1].transpose(), H.transpose(), pow(R, -1)])
#                     # print("A3_M2", A3)
#                     delta_M[i][j] = A0 + A1 + A2 + A3
#             M = np.add(M, delta_M)
#         if w == 0:
#             M1 = M
#             u = u_t0
#         else:
#             M2 = M
#     # print("M1 =\n",M1,
#     #       "\nM2 =\n", M2)
#     M = (np.subtract(M1, M2)) / delta
#     return M

if __name__ == '__main__':
    # Определение переменных
    m = q = v = nu = 1

    n = 2 # Размерность вектора х0
    s = 2 # Количество производных по тетта
    N = 40 # Число испытаний

    count = 0

    delta = 0.001

    tetta_true = np.array([-1.5, 1.0])
    tetta_false = np.array([-2, 0.01])

    IMF(tetta_true)

    # Главная функция второй лабы:
    # dIMF(tetta_else, mode=count)
    # print("Проверка Матрицы Фишера c tetta_else:\n", dIMF_test(u=(u_t0 + delta), tetta=tetta_else))
    # print("Матрица Фишера с ложной теттой:\n",dIMF(tetta_else, mode=count))
    # print("Матрица Фишера с истинной теттой:\n", dIMF(tetta_true, mode=count))
    # c = dIMF_test(u=(u_t0 + delta), tetta=tetta_else)

