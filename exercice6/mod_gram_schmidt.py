 
import numpy as np

def modgramschmidt(A):
    n = A.shape[1]
    Q = np.array(A, dtype='float64')
    R = np.zeros((n, n))
    for k in range(n):
        a_k = Q[..., k]
        R[k,k] = np.linalg.norm(a_k)
        a_k /= R[k, k]
        for i in range(k+1, n):
            a_i = Q[..., i]
            R[k,i] = np.transpose(a_k) @ a_i
            a_i -= R[k, i] * a_k
    return Q,R

def main():
    A =np.array([[1.,-4.], [2.,3.],[2.,2.]])
    print('A = ')
    print(A)
    Q, R = modgramschmidt(A)
    print('Q = ')
    print(Q)
    print('R = ')
    print(R)
    print('Q^T*Q = ')
    print(np.dot(Q.transpose(), Q))
    print('Q*R =')
    print(np.dot(Q, R))

main()

