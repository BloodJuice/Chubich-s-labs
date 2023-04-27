from functools import reduce
import numpy as np


def dIMF(tetta, mode):
    n = N
    dF_dtetta = np.array([[0, 0], [1, 0],   # dF_dtetta_1
                          [0, 0], [0, 0]])  # dF_dtetta_2
    dF_dtetta = dF_dtetta.reshape(2, 2, 2)

    dPsi_dtetta = np.array([[0, 0],         # dPsi_dtetta_1
                           [1, 0]])         # dPsi_dtetta_2
    dPsi_dtetta = dPsi_dtetta.reshape(2, 2, 1)


    dH_dtetta = np.array([[0, 0],           # dH_dtetta_1 [1][2][1]
                          [0, 0]])          # dH_dtetta_2 [2][2][1]
    dH_dtetta = dH_dtetta.reshape(2, 1, 2)

    dR_dtetta = np.array([0,                # dR_dtetta_1 [1][1]
                          0])               # dR_dtetta_2 [2][1]

    dxa_tk_dua_tbetta = []

    C = np.array([[[1, 0, 0, 0, 0, 0],     # C0
                    [0, 1, 0, 0, 0, 0]],
                   [[0, 0, 1, 0, 0, 0],     # C1
                    [0, 0, 0, 1, 0, 0]],
                   [[0, 0, 0, 0, 1, 0],     # C2
                    [0, 0, 0, 0, 0, 1]]])

    F = np.array([[-0.8, 1.0], [tetta[0], 0]])
    Psi = np.array([tetta[1], 1.0])
    Psi = Psi.reshape(2, 1)

    H = np.array([1.0, 0])
    H = H.reshape(1, 2)

    Fa = np.array([[-0.8, 1, 0, 0, 0, 0],
                [tetta[0], 0, 0, 0, 0, 0],
                [0, 0, -0.8, 1, 0, 0],
               [1, 0, tetta[0], 0, 0, 0],
               [0, 0, 0, 0, -0.8, 1],
               [0, 0, 0, 0, tetta[0], 0]])

    Psi_a = np.array([tetta[1], 1, 0, 0, 1, 0])
    Psi_a = Psi_a.reshape(6, 1)

    dxt0_dtetta = np.array([             #Размерность:   [2][2][1]
                            [0, 0],  # dxt0_dtetta_1 [1][2][1]
                            [0, 0]])   # dxt0_dtetta_2 [2][2][1]
    dxt0_dtetta = dxt0_dtetta.reshape(2, 2, 1)

    x_t0 = np.array([0, 0])
    x_t0 = x_t0.reshape(2, 1)

    # Расчёт M(tetta)
    Xa_tk_plus_one = []
    Xa_tk = []
    delta_M = np.zeros((2, 2))

    x0 = [0, 0]
    x1 = [0, 0]
    x2 = [0, 0]


    for k in range(n):
        M = np.zeros((2, 2))
        # Расчёт первого значения для Xa_tk, k == 0:
        if k == 0:
            x0 = np.matmul(F, x_t0) + np.dot(Psi, u_t0)
            x1 = np.matmul(dF_dtetta[0], x_t0) + np.matmul(F, dxt0_dtetta[0]) + np.dot(dPsi_dtetta[0], u_t0)
            x2 = np.matmul(dF_dtetta[1], x_t0) + np.matmul(F, dxt0_dtetta[1]) + np.dot(dPsi_dtetta[1], u_t0)
            Xa_tk_plus_one = np.concatenate((x0, x1, x2))
            # print("Xa_tk_plus_one:\n", Xa_tk_plus_one)

        # Расчёт последующих значений для Xa_tk, k > 0:
        if k > 0:
            Xa_tk_plus_one = np.matmul(Fa, Xa_tk) + np.dot(Psi_a, u_t0)
        Xa_tk = Xa_tk_plus_one

        for betta in range(n):
            if betta == k:
                Psi_a_du_tk_dua_tbetta = Psi_a
            else:
                Psi_a_du_tk_dua_tbetta = np.dot(Psi_a, 0)

            if k == 0:
                dxa_tk_plus_one_dua_tbetta = Psi_a_du_tk_dua_tbetta
            else:
                dxa_tk_plus_one_dua_tbetta = np.add(np.dot(Fa, dxa_tk_dua_tbetta), Psi_a_du_tk_dua_tbetta)
            dxa_tk_dua_tbetta = dxa_tk_plus_one_dua_tbetta


            coeff_dxa_tk_plus_one = np.add(np.dot(dxa_tk_plus_one_dua_tbetta, Xa_tk_plus_one.transpose()),
                                           np.dot(Xa_tk_plus_one, dxa_tk_plus_one_dua_tbetta.transpose()))
            # if betta == 1:
                # print("dxa_tk_plus_one_dua_tbetta\n", dxa_tk_plus_one_dua_tbetta,
                #     "\nPsi_a_du_tk_dua_tbetta\n", Psi_a_du_tk_dua_tbetta,
                #       "\ncoeff_dxa_tk_plus_one\n", coeff_dxa_tk_plus_one)

            for i in range(2):
                for j in range(2):
                    A0 = reduce(np.dot, [dH_dtetta[i], C[0],
                                coeff_dxa_tk_plus_one,
                                C[0].transpose(), dH_dtetta[j].transpose(), pow(R, -1)])
                    # print("A0", A0)

                    A1 = reduce(np.dot, [dH_dtetta[i], C[0],
                                coeff_dxa_tk_plus_one,
                                C[j + 1].transpose(), H.transpose(), pow(R, -1)])
                    # print("A1", A1)

                    A2 = reduce(np.dot, [H, C[i + 1],
                                coeff_dxa_tk_plus_one,
                                C[0].transpose(), dH_dtetta[j].transpose(), pow(R, -1)])
                    # print("A2", A2)

                    A3 = reduce(np.dot, [H, C[i + 1],
                                coeff_dxa_tk_plus_one,
                                C[j + 1].transpose(), H.transpose(), pow(R, -1)])
                    delta_M[i][j] = A0 + A1 + A2 + A3

            M = np.add(M, delta_M)
        if k == (N - 1):
            print("Мартица для k =", k, " равна =", "\n", M)

def dIMF_test(u, tetta):
    dF_dtetta = np.array([[0, 0], [1, 0],  # dF_dtetta_1
                          [0, 0], [0, 0]])  # dF_dtetta_2
    dF_dtetta = dF_dtetta.reshape(2, 2, 2)

    dPsi_dtetta = np.array([[0, 0],  # dPsi_dtetta_1
                            [1, 0]])  # dPsi_dtetta_2
    dPsi_dtetta = dPsi_dtetta.reshape(2, 2, 1)

    dH_dtetta = np.array([[0, 0],  # dH_dtetta_1 [1][2][1]
                          [0, 0]])  # dH_dtetta_2 [2][2][1]
    dH_dtetta = dH_dtetta.reshape(2, 1, 2)

    dR_dtetta = np.array([0,  # dR_dtetta_1
                          0])  # dR_dtetta_2

    dxt0_dtetta = np.array([  # Размерность:   [2][2][1]
                            [0, 0],  # dxt0_dtetta_1 [1][2][1]
                            [0, 0]])  # dxt0_dtetta_2 [2][2][1]
    dxt0_dtetta = dxt0_dtetta.reshape(2, 2, 1)

    C = np.array([[[1, 0, 0, 0, 0, 0],  # C0
                   [0, 1, 0, 0, 0, 0]],
                  [[0, 0, 1, 0, 0, 0],  # C1
                   [0, 0, 0, 1, 0, 0]],
                  [[0, 0, 0, 0, 1, 0],  # C2
                   [0, 0, 0, 0, 0, 1]]])

    F = np.array([[-0.8, 1.0], [tetta[0], 0]])
    Psi = np.array([tetta[1], 1.0])
    Psi = Psi.reshape(2, 1)

    H = np.array([1.0, 0])
    H = H.reshape(1, 2)

    x_t0 = np.array([0, 0])
    x_t0 = x_t0.reshape(2, 1)

    Fa = np.array([[-0.8, 1, 0, 0, 0, 0],
                   [tetta[0], 0, 0, 0, 0, 0],
                   [0, 0, -0.8, 1, 0, 0],
                   [1, 0, tetta[0], 0, 0, 0],
                   [0, 0, 0, 0, -0.8, 1],
                   [0, 0, 0, 0, tetta[0], 0]])

    Psi_a = np.array([tetta[1], 1, 0, 0, 1, 0])
    Psi_a = Psi_a.reshape(6, 1)



    # Расчёт M(tetta)
    Xa_tk_plus_one = []
    Xa_tk = []
    delta_M = np.zeros((2, 2))

    # Объявляю компоненты производной
    M1, M2 = np.zeros((2, 2))
    x0 = [0, 0]
    x1 = [0, 0]
    x2 = [0, 0]

    for w in range(2):
        M = np.array([[0, 0], [0, 0]])

        for _ in range(N):
            # Расчёт первого значения для Xa_tk, k == 0:
            if _ == 0:
                x0 = np.matmul(F, x_t0) + np.dot(Psi, u)
                x1 = np.matmul(dF_dtetta[0], x_t0) + np.matmul(F, dxt0_dtetta[0]) + np.dot(dPsi_dtetta[0], u)
                x2 = np.matmul(dF_dtetta[1], x_t0) + np.matmul(F, dxt0_dtetta[1]) + np.dot(dPsi_dtetta[1], u)
                Xa_tk_plus_one = np.concatenate((x0, x1, x2))
                # print(Xa_tk_plus_one)

            # Расчёт последующих значений для Xa_tk, k > 0:
            if _ > 0:
                Xa_tk_plus_one = np.matmul(Fa, Xa_tk) + np.dot(Psi_a, u)
            Xa_tk = Xa_tk_plus_one
            # print("\nXa_tk_plus_one\n", Xa_tk_plus_one)

            for i in range(2):
                for j in range(2):
                    A0 = reduce(np.dot, [dH_dtetta[i], C[0],
                                         Xa_tk_plus_one, Xa_tk_plus_one.transpose(),
                                         C[0].transpose(), dH_dtetta[j].transpose(), pow(R, -1)])
                    # print("A0_M2", A0)

                    A1 = reduce(np.dot, [dH_dtetta[i], C[0],
                                         Xa_tk_plus_one, Xa_tk_plus_one.transpose(),
                                         C[j + 1].transpose(), H.transpose(), pow(R, -1)])
                    # print("A1_M2", A1)

                    A2 = reduce(np.dot, [H, C[i + 1],
                                         Xa_tk_plus_one, Xa_tk_plus_one.transpose(),
                                         C[0].transpose(), dH_dtetta[j].transpose(), pow(R, -1)])
                    # print("A2_M2", A2)

                    A3 = reduce(np.dot, [H, C[i + 1],
                                         Xa_tk_plus_one, Xa_tk_plus_one.transpose(),
                                         C[j + 1].transpose(), H.transpose(), pow(R, -1)])
                    # print("A3_M2", A3)
                    delta_M[i][j] = A0 + A1 + A2 + A3
            M = np.add(M, delta_M)
        if w == 0:
            M1 = M
            u = u_t0
        else:
            M2 = M
    # print("M1 =\n",M1,
    #       "\nM2 =\n", M2)
    M = (np.subtract(M1, M2)) / delta
    return M

if __name__ == '__main__':
    # Определение переменных
    k = 0
    m = q = i = j = v = nu = 1
    R = 0.1
    N = 2
    count = 0
    u_t0 = 1.0
    delta = 0.001

    tetta_true = np.array([-1.5, 1.0])
    tetta_else = np.array([-2, 0.01])



    # Главная функция второй лабы:
    dIMF(tetta_else, mode=count)
    print("Проверка Матрицы Фишера c tetta_else:\n", dIMF_test(u=(u_t0 + delta), tetta=tetta_else))
    # print("Матрица Фишера с ложной теттой:\n",dIMF(tetta_else, mode=count))
    # print("Матрица Фишера с истинной теттой:\n", dIMF(tetta_true, mode=count))
    # c = dIMF_test(u=(u_t0 + delta), tetta=tetta_else)

