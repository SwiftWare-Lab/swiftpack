import ast
from typing import Tuple
from swiftpack.transformation import Transformation
from swiftpack.ddg import generate_ddg


class LoopExchange(Transformation):
    """Exchanges loop ordering (e.g., swapping 'j' and 'k' loops to 'ikj') using DDG legality analysis."""

    def check_legality(self, tree: ast.AST,
                       target_loops: Tuple[str, str] = ('j',
                                                        'k')) -> bool:
        """
        Verifies the legality of interchanging two loops (l1, l2) by analyzing
        the DDG distance vectors. Swapping is legal iff no dependence vector
        becomes lexicographically negative after the permutation.
        """
        l1, l2 = target_loops
        ddg = generate_ddg(tree)

        found_l1 = False
        found_l2 = False

        for u, _, data in ddg.edges(data=True):
            dist = data.get('distance_vector')
            node_data = ddg.nodes[u]
            loop_nest = node_data.get('loop_nest', [])

            if l1 in loop_nest:
                found_l1 = True
            if l2 in loop_nest:
                found_l2 = True

            # If there's no distance vector or either loop isn't in this node's nest, skip
            if dist is None or not loop_nest:
                continue

            if l1 in loop_nest and l2 in loop_nest:
                i1 = loop_nest.index(l1)
                i2 = loop_nest.index(l2)

                # Simulate interchanging the two loop levels in the distance vector
                new_dist = list(dist)
                if i1 < len(new_dist) and i2 < len(new_dist):
                    new_dist[i1], new_dist[i2] = new_dist[i2], new_dist[
                        i1]

                    # Legality check: The first non-zero element in the permuted
                    # distance vector must not be negative (< 0)
                    for val in new_dist:
                        if val is not None:
                            if val > 0:
                                break  # Lexicographically positive (safe)
                            elif val < 0:
                                return False  # Direction reversal violation found!

        # Fallback structural check if no edges with distance vectors were traversed
        if not found_l1 or not found_l2:
            loop_vars = []

            class LoopVisitor(ast.NodeVisitor):
                def visit_For(self, node):
                    if isinstance(node.target, ast.Name):
                        loop_vars.append(node.target.id)
                    self.generic_visit(node)

            LoopVisitor().visit(tree)
            if l1 not in loop_vars or l2 not in loop_vars:
                return False

        return True

    def apply(self, tree: ast.AST,
              target_loops: Tuple[str, str] = ('j', 'k')) -> ast.AST:
        """
        Applies the loop interchange transformation by reordering the matching loop nest.
        """
        l1, l2 = target_loops

        class Exchanger(ast.NodeTransformer):
            def visit_For(self, node):
                self.generic_visit(node)
                # Locate loop matching l1 whose immediate body contains loop matching l2
                if isinstance(node.target,
                              ast.Name) and node.target.id == l1:
                    if len(node.body) == 1 and isinstance(node.body[0],
                                                          ast.For):
                        inner = node.body[0]
                        if isinstance(inner.target,
                                      ast.Name) and inner.target.id == l2:
                            # Swap target variables to reflect interchange
                            node.target.id = l2
                            inner.target.id = l1
                return node

        transformed = Exchanger().visit(tree)
        ast.fix_missing_locations(transformed)
        return transformed