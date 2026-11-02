import ast
from typing import Any, Dict, List, Tuple, Type
from swiftpack.transformation import Transformation
from swiftpack.loop_exchange import LoopExchange
from swiftpack.loop_tiling import LoopTiling
from swiftpack.loop_parallelism import LoopParallelism
from swiftpack.loop_unrolling import LoopUnrolling
from swiftpack.lvn import LocalValueNumbering

class Scheduler:
    """Analyzes the input AST and produces an ordered sequence of transformations."""

    def schedule(self, tree: ast.AST) -> List[Tuple[Type[Transformation], Dict[str, Any]]]:
        """Returns array of tuples: [(TransformationClass, kwargs_dict), ...]"""
        raise NotImplementedError



# ===========================================================================
# Concrete Schedulers
# ===========================================================================

class AutoTileAndParallelizeScheduler(Scheduler):
    """Custom schedule generator for standard Matrix Multiplication kernels."""

    def schedule(self, tree: ast.AST) -> List[Tuple[Type[Transformation], Dict[str, Any]]]:
        return [
            # 1. Apply local value numbering and optimization within each basic block
            (LocalValueNumbering, {"directive": "lvn"}),
            # 2. Reorder loops from IJK -> IKJ for cache locality
            (LoopExchange, {"target_loops": ("j", "k")}),
            # 3. Block/Tile nested iteration space
            (LoopTiling, {"target_loop": "i", "tile_size": 32}),
            # 4. Unroll loop 'j' for better instruction-level parallelism
            (LoopUnrolling, {"loop_var": "j", "factor": 32}),
            # 5. Parallelize loop 'i' loop across CPU cores
            (LoopParallelism, {"loop_var": "k"}),
        ]

