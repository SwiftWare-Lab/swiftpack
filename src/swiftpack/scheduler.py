import ast
from typing import Any, Dict, List, Tuple, Type
from swiftpack.transformation import Transformation
from swiftpack.loop_exchange import LoopExchange
from swiftpack.tiling import LoopTiling
from swiftpack.parallelism import ParallelizeLoop

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
            # 1. Reorder loops from IJK -> IKJ for cache locality
            (LoopExchange, {"target_loops": ("j", "k")}),
            # 2. Block/Tile nested iteration space
            (LoopTiling, {"tile_size": 32}),
            # 3. Parallelize outermost 'i' loop across CPU cores
            (ParallelizeLoop, {"loop_var": "i"}),
        ]

