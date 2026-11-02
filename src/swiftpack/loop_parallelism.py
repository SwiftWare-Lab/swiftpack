import ast
from swiftpack.transformation import Transformation
from swiftpack.ddg import generate_ddg, generate_ddg_with_vectors


class LoopParallelism(Transformation):
    """Parallelizes a target loop by checking DDG for loop-carried dependencies and transforming range to prange."""

    def check_legality(self, tree: ast.AST, loop_var: str = 'i') -> bool:
        """
        Verifies the legality of parallelizing a loop. A loop can be parallelized
        if and only if it carries no loop-carried dependencies. That is, for every
        dependence vector, the distance component corresponding to the target loop
        must be 0 (or loop-independent). Non-zero distances indicate cross-iteration
        hazards that cause race conditions if parallelized.
        """
        ddg = generate_ddg_with_vectors(tree)
        found_loop = False
        print(f"[LoopParallelism] Checking legality for loop variable '{loop_var}'...")
        # print ddg edges and their distance vectors for debugging
        for u, v, data in ddg.edges(data=True):
            dist = data.get('distance_vector')
            dir = data.get('direction_vector')
            print(f"Edge from {u} to {v}: distance_vector={dist}, direction_vector={dir}")

        for u, _, data in ddg.edges(data=True):
            dist = data.get('distance_vector')
            dir = data.get('direction_vector')
            node_data = ddg.nodes[u]
            loop_nest = node_data.get('loop_nest', [])

            if loop_var in loop_nest:
                found_loop = True

            if dist is None or loop_var not in loop_nest:
                continue

            idx = loop_nest.index(loop_var)
            if idx < len(dist):
                val = dist[idx]
                # A non-zero distance at this loop depth means a loop-carried dependency exists
                if val is not None and val != 0:
                    return False

        # Fallback structural verification if no explicit edge vectors were evaluated
        if not found_loop:
            loop_vars = []
            class LoopVisitor(ast.NodeVisitor):
                def visit_For(self, node):
                    if isinstance(node.target, ast.Name):
                        loop_vars.append(node.target.id)
                    self.generic_visit(node)
            LoopVisitor().visit(tree)
            return loop_var in loop_vars

        return True

    def apply(self, tree: ast.AST, loop_var: str = 'i') -> ast.AST:
        """
        Applies loop parallelism by converting standard `range()` calls to parallel `prange()`
        on the targeted loop level.
        """
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