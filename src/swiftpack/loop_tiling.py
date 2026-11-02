import ast
from typing import Tuple
from swiftpack.transformation import Transformation
from swiftpack.ddg import generate_ddg


class LoopTiling(Transformation):
    """Tiles a loop nest (blocking) to improve cache locality using DDG legality analysis."""

    def check_legality(self, tree: ast.AST, target_loop: str,
                       tile_size: int = 32) -> bool:
        """
        Verifies the legality of tiling a loop. Tiling is legal if the target loop
        carries no negative distance dependencies (i.e., all distance vector components
        for the target loop are non-negative, ensuring no backward time/iteration flow).
        """
        ddg = generate_ddg(tree)
        found_loop = False

        for u, _, data in ddg.edges(data=True):
            dist = data.get('distance_vector')
            node_data = ddg.nodes[u]
            loop_nest = node_data.get('loop_nest', [])

            if target_loop in loop_nest:
                found_loop = True

            if dist is None or target_loop not in loop_nest:
                continue

            idx = loop_nest.index(target_loop)
            if idx < len(dist):
                val = dist[idx]
                # If the distance vector component is negative, tiling violates dependency
                if val is not None and val < 0:
                    return False

        # Structural fallback check if no explicit edges were evaluated
        if not found_loop:
            loop_vars = []

            class LoopVisitor(ast.NodeVisitor):
                def visit_For(self, node):
                    if isinstance(node.target, ast.Name):
                        loop_vars.append(node.target.id)
                    self.generic_visit(node)

            LoopVisitor().visit(tree)
            return target_loop in loop_vars

        return True

    def apply(self, tree: ast.AST, target_loop: str,
              tile_size: int = 32) -> ast.AST:
        """
        Applies loop tiling by strip-mining the target loop into outer tile
        and inner point loops (e.g., transforming `for i in range(N)` into
        `for i_tile in range(0, N, tile_size): for i in range(i_tile, min(i_tile + tile_size, N))`).
        """
        tile_var = f"{target_loop}_tile"

        class Tiler(ast.NodeTransformer):
            def visit_For(self, node):
                self.generic_visit(node)

                if isinstance(node.target,
                              ast.Name) and node.target.id == target_loop:
                    # Expecting node.iter to be range(stop) or range(start, stop) or range(start, stop, step)
                    if isinstance(node.iter, ast.Call) and isinstance(
                            node.iter.func,
                            ast.Name) and node.iter.func.id == 'range':
                        args = node.iter.args

                        if len(args) == 1:
                            start = ast.Constant(value=0)
                            stop = args[0]
                            step = ast.Constant(value=1)
                        elif len(args) == 2:
                            start = args[0]
                            stop = args[1]
                            step = ast.Constant(value=1)
                        elif len(args) == 3:
                            start = args[0]
                            stop = args[1]
                            step = args[2]
                        else:
                            return node

                        # Outer tile loop: for target_loop_tile in range(start, stop, tile_size)
                        outer_iter = ast.Call(
                            func=ast.Name(id='range', ctx=ast.Load()),
                            args=[start, stop,
                                  ast.Constant(value=tile_size)],
                            keywords=[]
                        )

                        # Inner point loop bounds: range(target_loop_tile, min(target_loop_tile + block_size, stop))
                        inner_start = ast.Name(id=tile_var,
                                               ctx=ast.Load())
                        inner_stop_add = ast.BinOp(
                            left=ast.Name(id=tile_var, ctx=ast.Load()),
                            op=ast.Add(),
                            right=ast.Constant(value=tile_size)
                        )
                        inner_stop = ast.Call(
                            func=ast.Name(id='min', ctx=ast.Load()),
                            args=[inner_stop_add, stop],
                            keywords=[]
                        )
                        inner_iter = ast.Call(
                            func=ast.Name(id='range', ctx=ast.Load()),
                            args=[inner_start, inner_stop],
                            keywords=[]
                        )

                        # Construct inner loop preserving original body
                        inner_loop = ast.For(
                            target=ast.Name(id=target_loop,
                                            ctx=ast.Store()),
                            iter=inner_iter,
                            body=node.body,
                            orelse=[]
                        )

                        # Construct outer loop wrapping inner loop
                        outer_loop = ast.For(
                            target=ast.Name(id=tile_var,
                                            ctx=ast.Store()),
                            iter=outer_iter,
                            body=[inner_loop],
                            orelse=[]
                        )

                        return outer_loop

                return node

        transformed = Tiler().visit(tree)
        ast.fix_missing_locations(transformed)
        return transformed