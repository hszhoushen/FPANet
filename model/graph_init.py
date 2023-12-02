# by Liguang Zhou, 2020.11.20

import numpy as np
import torch
from numpy.linalg import eig, inv

class Graph_Init(object):
    def __init__(self, nodes_num, batch_size):
        # print('graph initialization!')
        # graph initialization
        self.K = nodes_num
        self.batch_size = batch_size

        if (self.K == 20):
            self.g1 = [0, 5, 15]
            self.c1 = [10]
            self.g2 = [1, 6, 16]
            self.c2 = [11]
            self.g3 = [2, 7, 17]
            self.c3 = [12]
            self.g4 = [3, 8, 18]
            self.c4 = [13]
            self.g5 = [4, 9, 19]
            self.c5 = [14]
            # self.g6 = [10, 12, 13]
            # self.c6 = [11]

        elif (self.K == 24):

            self.g1 = [0, 6, 18]
            self.c1 = [12]
            self.g2 = [1, 7, 19]
            self.c2 = [13]
            self.g3 = [2, 8, 20]
            self.c3 = [14]
            self.g4 = [3, 9, 21]
            self.c4 = [15]
            self.g5 = [4, 10, 22]
            self.c5 = [16]
            self.g6 = [5, 11, 23]
            self.c6 = [17]
            # self.g7 = [12, 13, 15, 16, 17]
            # self.c7 = [14]

        elif (self.K == 16):
            # self.g1 = [0, 2, 4, 8]
            # self.c1 = [3]
            # self.g2 = [1, 5, 7, 9]
            # self.c2 = [6]
            # self.g3 = [11, 12, 14, 15]
            # self.c3 = [13]
            # self.g4 = [3, 6, 13]
            # self.c4 = [10]

            self.g1 = [0, 4, 12]
            self.c1 = [8]
            self.g2 = [1, 5, 13]
            self.c2 = [9]
            self.g3 = [2, 6, 14]
            self.c3 = [10]
            self.g4 = [3, 7, 15]
            self.c4 = [11]

        elif (self.K == 4):
            self.g1 = [0,1,3]
            self.c1 = [2]

        elif (self.K == 8):
            self.g1 = [1, 3, 7]
            self.c1 = [5]
            self.g2 = [0, 2, 6]
            self.c2 = [4]

        elif (self.K == 12):
            self.g1 = [0,3,9]
            self.c1 = [6]
            self.g2 = [1,4,10]
            self.c2 = [7]
            self.g3 = [2,5,11]
            self.c3 = [8]
            # self.g4 = [6, 8]
            # self.c4 = [7]


        # adjacency matrix
        self.A = torch.zeros([self.K, self.K])
        self.Adjacency_matrix_top = torch.zeros([self.batch_size, self.K, self.K])
        self.Adjacency_matrix_med = torch.zeros([self.batch_size, self.K, self.K])

        self.Lnormtop = torch.zeros(self.batch_size, self.K, self.K)
        self.Lnormmed = torch.zeros(self.batch_size, self.K, self.K)

        # degree matrix
        self.D = torch.zeros([self.K, self.K])

        # create identity matrix
        self.I = torch.eye(self.K, self.K)

        # adjacency matrix init
        self.Adjacency_matrix_init()

    # input: symmetric positive definite matrix A
    # output: A^(-1/2)
    def spd(self, A):

        # eigen value decomposition
        eig_val, eig_vec = eig(A)

        # 特征值开方取zhi倒数对角dao化
        eig_diag = np.diag(1 / (eig_val ** 0.5))

        # inv为求逆属
        B = np.dot(np.dot(eig_vec, eig_diag), inv(eig_vec))

        # numpy to torch
        B = torch.from_numpy(B)

        # return the A's -1/2次幂
        return B


    def Adjacency_matrix_init(self):
        # calculate A
        if (self.K == 16):
            for i in range(self.K):
                for j in range(self.K):
                    if (i in self.g1 or j in self.g1) and (i in self.c1 or j in self.c1):
                        # print('i:', i, 'j:', j)
                        self.A[i][j] = 1

                    elif (i in self.g2 or j in self.g2) and (i in self.c2 or j in self.c2):
                        # print('i:', i, 'j:', j)
                        self.A[i][j] = 1
                    elif (i in self.g3 or j in self.g3) and (i in self.c3 or j in self.c3):
                        # print('i:', i, 'j:', j)
                        self.A[i][j] = 1

                    elif (i in self.g4 or j in self.g4) and (i in self.c4 or j in self.c4):
                        # print('i:', i, 'j:', j)
                        self.A[i][j] = 1

        elif (self.K == 4):
            for i in range(self.K):
                for j in range(self.K):
                    if (i in self.g1 or j in self.g1) and (i in self.c1 or j in self.c1):
                        # print('i:', i, 'j:', j)
                        self.A[i][j] = 1

        elif (self.K == 8):
            for i in range(self.K):
                for j in range(self.K):
                    if (i in self.g1 or j in self.g1) and (i in self.c1 or j in self.c1):
                        # print('i:', i, 'j:', j)
                        self.A[i][j] = 1

                    elif (i in self.g2 or j in self.g2) and (i in self.c2 or j in self.c2):
                        # print('i:', i, 'j:', j)
                        self.A[i][j] = 1

        elif (self.K == 12):
            for i in range(self.K):
                for j in range(self.K):

                    if (i in self.g1 or j in self.g1) and (i in self.c1 or j in self.c1):
                        # print('i:', i, 'j:', j)
                        self.A[i][j] = 1

                    elif (i in self.g2 or j in self.g2) and (i in self.c2 or j in self.c2):
                        # print('i:', i, 'j:', j)
                        self.A[i][j] = 1

                    elif (i in self.g3 or j in self.g3) and (i in self.c3 or j in self.c3):
                        # print('i:', i, 'j:', j)
                        self.A[i][j] = 1
        elif (self.K == 20):
            for i in range(self.K):
                for j in range(self.K):

                    if (i in self.g1 or j in self.g1) and (i in self.c1 or j in self.c1):
                        # print('i:', i, 'j:', j)
                        self.A[i][j] = 1

                    elif (i in self.g2 or j in self.g2) and (i in self.c2 or j in self.c2):
                        # print('i:', i, 'j:', j)
                        self.A[i][j] = 1

                    elif (i in self.g3 or j in self.g3) and (i in self.c3 or j in self.c3):
                        # print('i:', i, 'j:', j)
                        self.A[i][j] = 1

                    elif (i in self.g4 or j in self.g4) and (i in self.c4 or j in self.c4):
                        # print('i:', i, 'j:', j)
                        self.A[i][j] = 1

                    elif (i in self.g5 or j in self.g5) and (i in self.c5 or j in self.c5):
                        # print('i:', i, 'j:', j)
                        self.A[i][j] = 1

                    # elif (i in self.g6 or j in self.g6) and (i in self.c6 or j in self.c6):
                    #     # print('i:', i, 'j:', j)
                    #     self.A[i][j] = 1

        elif (self.K == 24):
            for i in range(self.K):
                for j in range(self.K):

                    if (i in self.g1 or j in self.g1) and (i in self.c1 or j in self.c1):
                        # print('i:', i, 'j:', j)
                        self.A[i][j] = 1

                    elif (i in self.g2 or j in self.g2) and (i in self.c2 or j in self.c2):
                        # print('i:', i, 'j:', j)
                        self.A[i][j] = 1

                    elif (i in self.g3 or j in self.g3) and (i in self.c3 or j in self.c3):
                        # print('i:', i, 'j:', j)
                        self.A[i][j] = 1

                    elif (i in self.g4 or j in self.g4) and (i in self.c4 or j in self.c4):
                        # print('i:', i, 'j:', j)
                        self.A[i][j] = 1

                    elif (i in self.g5 or j in self.g5) and (i in self.c5 or j in self.c5):
                        # print('i:', i, 'j:', j)
                        self.A[i][j] = 1

                    elif (i in self.g6 or j in self.g6) and (i in self.c6 or j in self.c6):
                        # print('i:', i, 'j:', j)
                        self.A[i][j] = 1

                    # elif (i in self.g7 or j in self.g7) and (i in self.c7 or j in self.c7):
                    #     # print('i:', i, 'j:', j)
                    #     self.A[i][j] = 1

    def Adjacency_matrix_calculation(self, rows, columns):
        if(rows.shape[0] != self.batch_size):
            print('rows.shape:', rows.shape)
            print('columns.shape:', columns.shape)

        for idx in range(rows.shape[0]):
            # e.g., 0*K~0*K+K (0-16), 2*K~2*K+K (32-48)
            rowsmaxK = rows[idx, 0:self.K]
            colsmaxK = columns[idx, 0:self.K]

            for i in range(len(self.A)):
                for j in range(len(self.A[i])):

                    if(self.A[i][j] == 1):
                        # print('before:', self.Adjacency_matrix[idx][i][j])
                        self.Adjacency_matrix_top[idx][i][j] = abs(rowsmaxK[i]-rowsmaxK[j]) + \
                                                             abs(colsmaxK[i]-colsmaxK[j])
                        # print('after:', self.Adjacency_matrix[idx][i][j])

            # e.g., 0*K+K, 0*K+2*K (16-32), 2*K+K, 2*K+2*K (48-64)

            rowsmedK = rows[idx, self.K:2*self.K]
            colsmedK = columns[idx, self.K:2*self.K]

            for i in range(len(self.A)):
                for j in range(len(self.A[i])):

                    if(self.A[i][j] == 1):
                        # print('before:', self.Adjacency_matrix[idx][i][j])
                        self.Adjacency_matrix_med[idx][i][j] = abs(rowsmedK[i]-rowsmedK[j]) + \
                                                             abs(colsmedK[i]-colsmedK[j])

        # print('Adjacency_matrix:', self.Adjacency_matrix.shape)
        # print(self.Adjacency_matrix[0])
        # print(self.Adjacency_matrix[1])
        # print(self.Adjacency_matrix[2])
        # print(self.Adjacency_matrix[31])


    def Degree_matrix(self):
        # calculate D
        for i in range(self.K):
            for j in range(self.K):
                self.D[i][i] = self.D[i][i] + self.A[i][j]

    def Fixed_Lnorm(self):
        self.Degree_matrix()
        # print('self.D:', self.D.shape, self.I.shape, self.A.shape)
        # print('self.D:', self.D, self.I, self.A)
        Lnorm = self.spd(self.D + self.I) * (self.A + self.I) * self.spd(self.D + self.I)
        return Lnorm

    def Dynamic_Lnorm(self, rows, columns):
        self.Degree_matrix()
        self.Adjacency_matrix_calculation(rows, columns)
        # print('self.D:', self.D.shape, self.I.shape, self.A.shape)
        # print('self.D:', self.D, self.I, self.A)
        for i in range(self.batch_size):
            self.Lnormtop[i] = self.spd(self.D + self.I) * (self.Adjacency_matrix_top[i] + self.I) * \
                            self.spd(self.D + self.I)
            self.Lnormmed[i] = self.spd(self.D + self.I) * (self.Adjacency_matrix_med[i] + self.I) * \
                               self.spd(self.D + self.I)

        return self.Lnormtop, self.Lnormmed