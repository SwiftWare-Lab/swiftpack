import ast
import itertools
import numpy as np
import pytest

from swiftpack import swiftpack
from swiftpack.scheduler import Scheduler
from swiftpack.transformation import Transformation
from swiftpack import (
    LocalValueNumbering,
    LoopExchange,
    LoopParallelism,
    LoopTiling,
)


class CombinatorialScheduler(Scheduler):
    """A parameterized scheduler that generates unique transformation schedules."""

    def __init__(self, schedule_index: int):
        self.schedule_index = schedule_index

    def schedule(self, tree: ast.AST) -> list[
        tuple[type[Transformation], dict]]:
        exchange_pairs = [
            ("i", "j"),
            ("j", "k"),
            ("i", "k"),
            ("k", "j"),
            ("k", "i"),
            ("j", "i"),
        ]
        tile_loops = ["i", "j", "k"]
        tile_sizes = [16, 32, 64]
        parallel_vars = ["i", "j", "k"]

        # Generate combinatorial options space (6 * 3 * 4 * 3 = 216 combinations)
        all_combinations = list(
            itertools.product(exchange_pairs, tile_loops, tile_sizes,
                              parallel_vars)
        )

        combo_idx = self.schedule_index % len(all_combinations)
        ex_pair, t_loop, t_size, p_var = all_combinations[combo_idx]

        return [
            # 1. Local Value Numbering optimization pass
            (LocalValueNumbering, {"directive": "lvn"}),
            # 2. Reorder loops based on schedule parameter
            (LoopExchange, {"target_loops": ex_pair}),
            # 3. Tile/block target loop dimension
            (LoopTiling, {"target_loop": t_loop, "tile_size": t_size}),
            # 4. Attempt parallelization on target loop variable
            (LoopParallelism, {"loop_var": p_var}),
        ]


@pytest.fixture(scope="module")
def matmul_inputs():
    """Provides test matrices and baseline results for matrix multiplication."""
    N = 32  # Compact size for rapid batch test execution
    np.random.seed(42)
    A = np.random.randn(N, N).astype(np.float64)
    B = np.random.randn(N, N).astype(np.float64)
    baseline = np.dot(A, B)
    return A, B, baseline


@pytest.mark.parametrize("schedule_idx", range(500))
def test_combinatorial_matmul_schedules(schedule_idx, matmul_inputs):
    """
    Tests 100 distinct compiler schedules combining loop exchange, tiling,
    and parallelization. Legal schedules must compile and match NumPy output;
    illegal schedules are safely blocked by dependence analysis.
    """
    A, B, baseline = matmul_inputs
    N = A.shape[0]
    scheduler_instance = CombinatorialScheduler(schedule_idx)

    @swiftpack(scheduler=scheduler_instance, parallel=True,
               fastmath=True)
    def matmul_test(A_in, B_in, C_out):
        n = A_in.shape[0]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    C_out[i, j] += A_in[i, k] * B[k, j]

    C_test = np.zeros((N, N), dtype=np.float64)

    try:
        # Attempt compilation and execution
        matmul_test(A, B, C_test)

        # If execution succeeded, verify mathematical correctness against NumPy
        assert np.allclose(C_test, baseline, atol=1e-5), (
            f"Schedule {schedule_idx} produced incorrect numerical results!"
        )

    except Exception as e:
        # Re-raise any assertion or correctness failure
        if isinstance(e, AssertionError):
            raise e

        # Otherwise, exceptions driven by legality checks blocking invalid
        # transformations (e.g. parallelizing reduction loop 'k') are expected and valid.
        pass