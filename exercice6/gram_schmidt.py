
import numpy as np

def gramschmidt(A):
    R = np.zeros((A.shape[1], A.shape[1]))
    Q = np.zeros(A.shape)
    for k in range(0, A.shape[1]):
        R[k, k] = np.sqrt(np.dot(A[:, k], A[:, k]))
        Q[:, k] = A[:, k]/R[k, k]
        for j in range(k+1, A.shape[1]):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] = A[:, j] - R[k, j]*Q[:, k]
    return Q, R

def main():
    A =np.array([[1.,-4.], [2.,3.],[2.,2.]])
    print('A = ')
    print(A)
    Q, R = gramschmidt(A)
    print('Q = ')
    print(Q)
    print('R = ')
    print(R)
    print('Q^T*Q = ')
    print(np.dot(Q.transpose(), Q))
    print('Q*R =')
    print(np.dot(Q, R))

main()
