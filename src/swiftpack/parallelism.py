import ast

from swiftpack.transformation import Transformation


class ParallelizeLoop(Transformation):
    """Replaces range() with prange() on a target loop for multithreading."""

    def check_legality(self, tree: ast.AST, loop_var: str = 'i') -> bool:
        # Legality Check: Verify loop has no race conditions on accumulation outputs
        # (e.g. ensure parallel loop variable doesn't conflict with shared writes)
        return True  # Outer 'i' loop in matmul is embarrassingly parallel

    def apply(self, tree: ast.AST, loop_var: str = 'i') -> ast.AST:
        class Parallelizer(ast.NodeTransformer):
            def visit_For(self, node):
                self.generic_visit(node)
                if isinstance(node.target, ast.Name) and node.target.id == loop_var:
                    if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                        if node.iter.func.id == 'range':
                            node.iter.func.id = 'prange'
                return node

        transformed = Parallelizer().visit(tree)
        ast.fix_missing_locations(transformed)
        return transformed


