import numpy as np
from scipy.sparse import csc_array
from scipy.sparse.linalg import cg

P = np.array([[4, 0, 1, 0],
              [0, 5, 0, 0],
              [1, 0, 3, 2],
              [0, 0, 2, 4]])
A = csc_array(P)
b = np.array([-1, -0.5, -1, 2])
print(f'For array A:\n{A}\nand vector b:\n{b}')
x, exit_code = cg(A, b, atol=1e-5)
print(f'Output: {x}')
print(np.allclose(A.dot(x), b))
