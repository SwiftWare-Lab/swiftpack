import ast
import copy
from typing import Tuple
from swiftpack.transformation import Transformation


class LoopUnrolling(Transformation):
    """Unrolls a target loop by a given factor to expose Instruction-Level Parallelism (ILP)."""

    def check_legality(self, tree: ast.AST, loop_var: str,
                       factor: int = 4) -> bool:
        """
        Verifies the legality of loop unrolling. Standard scalar inner loop unrolling
        is generally always legal as it preserves dependency order and execution semantics.
        The legality check ensures that the target loop variable exists within the AST.
        """
        loop_vars = []

        class LoopVisitor(ast.NodeVisitor):
            def visit_For(self, node):
                if isinstance(node.target, ast.Name):
                    loop_vars.append(node.target.id)
                self.generic_visit(node)

        LoopVisitor().visit(tree)
        return loop_var in loop_vars

    def apply(self, tree: ast.AST, loop_var: str,
              factor: int = 4) -> ast.AST:
        """
        Applies loop unrolling by scaling the loop step size by the unrolling factor
        and duplicating the loop body `factor` times with appropriate index offsets.
        """

        class Unroller(ast.NodeTransformer):
            def visit_For(self, node):
                self.generic_visit(node)

                if isinstance(node.target,
                              ast.Name) and node.target.id == loop_var:
                    if isinstance(node.iter, ast.Call) and isinstance(
                            node.iter.func,
                            ast.Name) and node.iter.func.id == 'range':
                        args = node.iter.args

                        # Parse range arguments (start, stop, step)
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

                        # Multiply the loop step by the unrolling factor
                        new_step = ast.BinOp(
                            left=step,
                            op=ast.Mult(),
                            right=ast.Constant(value=factor)
                        )
                        node.iter.args = [start, stop, new_step]

                        # Replicate and offset the loop body `factor` times
                        new_body = []
                        for k in range(factor):
                            cloned_body = copy.deepcopy(node.body)

                            if k == 0:
                                new_body.extend(cloned_body)
                            else:
                                # Substitute usages of loop_var with (loop_var + k * step)
                                class SubstitutedTransformer(
                                    ast.NodeTransformer):
                                    def visit_Name(self, n):
                                        if n.id == loop_var and isinstance(
                                                n.ctx, ast.Load):
                                            return ast.BinOp(
                                                left=ast.Name(
                                                    id=loop_var,
                                                    ctx=ast.Load()),
                                                op=ast.Add(),
                                                right=ast.Constant(
                                                    value=k)
                                            )
                                        return n

                                transformer = SubstitutedTransformer()
                                transformed_cloned = [
                                    transformer.visit(stmt) for stmt in
                                    cloned_body]
                                new_body.extend(transformed_cloned)

                        node.body = new_body

                return node

        transformed = Unroller().visit(tree)
        ast.fix_missing_locations(transformed)
        return transformed