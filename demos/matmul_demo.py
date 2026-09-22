import numpy as np
from swiftpack import swiftpack

from swiftpack.scheduler import AutoTileAndParallelizeScheduler, Scheduler



if __name__ == "__main__":
    N = 256
    A = np.random.randn(N, N).astype(np.float64)
    B = np.random.randn(N, N).astype(np.float64)
    C = np.zeros((N, N), dtype=np.float64)


    # Function decorated with custom scheduling strategy
    @swiftpack(scheduler=AutoTileAndParallelizeScheduler(), parallel=True, fastmath=True)
    def matmul_swiftpack(A, B, C):
        n = A.shape[0]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    C[i, j] += A[i, k] * B[k, j]


    # Run transformed JIT function
    matmul_swiftpack(A, B, C)

    # Correctness check against standard matrix multiplication
    assert np.allclose(C, np.dot(A, B)), "Verification Failed!"
    print("\nResult verified successfully against NumPy!")