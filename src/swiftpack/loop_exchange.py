import ast
from typing import Any, Dict, List, Tuple, Type

from swiftpack.transformation import Transformation

# ===========================================================================
# Concrete Transformation Passes (Legality + AST Mutation)
# ===========================================================================

class LoopExchange(Transformation):
    """Exchanges loop ordering (e.g., swapping 'j' and 'k' loops to 'ikj')."""

    def check_legality(self, tree: ast.AST, target_loops: Tuple[str, str] = ('j', 'k')) -> bool:
        # Legality Check: Verify nested loops exist and don't contain loop-carried
        # dependencies that break when reordered.
        l1, l2 = target_loops
        loop_vars = []

        class LoopVisitor(ast.NodeVisitor):
            def visit_For(self, node):
                if isinstance(node.target, ast.Name):
                    loop_vars.append(node.target.id)
                self.generic_visit(node)

        LoopVisitor().visit(tree)
        # Verify both loops are present in the AST nest
        return l1 in loop_vars and l2 in loop_vars

    def apply(self, tree: ast.AST, target_loops: Tuple[str, str] = ('j', 'k')) -> ast.AST:
        l1, l2 = target_loops

        class Exchanger(ast.NodeTransformer):
            def visit_For(self, node):
                self.generic_visit(node)
                # Locate loop containing target_loops[0] whose immediate body is target_loops[1]
                if isinstance(node.target, ast.Name) and node.target.id == l1:
                    if len(node.body) == 1 and isinstance(node.body[0], ast.For):
                        inner = node.body[0]
                        if isinstance(inner.target, ast.Name) and inner.target.id == l2:
                            # Swap target variables (IJK -> IKJ)
                            node.target.id = l2
                            inner.target.id = l1
                return node

        transformed = Exchanger().visit(tree)
        ast.fix_missing_locations(transformed)
        return transformed


