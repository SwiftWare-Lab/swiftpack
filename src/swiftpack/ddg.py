import ast
import matplotlib.pyplot as plt
import networkx as nx


class DefUseVisitor(ast.NodeVisitor):
    """AST Visitor to extract variable definitions, uses, and multi-dimensional subscript index expressions."""

    def __init__(self):
        self.defs = set()
        self.uses = set()
        self.subscripts = {}  # base_name -> list of index nodes (or combined slice)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.defs.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            self.uses.add(node.id)

    def visit_Subscript(self, node):
        # Extract the root base array name (e.g., 'C' from C[i][j])
        curr = node
        indices = []
        while isinstance(curr, ast.Subscript):
            indices.insert(0, curr.slice)
            curr = curr.value

        if isinstance(curr, ast.Name):
            base_name = curr.id
            subscript_str = ", ".join(
                ast.unparse(idx) for idx in indices)
            access_repr = f"{base_name}[{subscript_str}]"

            if isinstance(node.ctx, ast.Store):
                self.defs.add(access_repr)
            elif isinstance(node.ctx, ast.Load):
                self.uses.add(access_repr)

            # Store all index dimensions together as a tuple or list of slice nodes
            self.subscripts[access_repr] = (base_name, indices if len(
                indices) > 1 else indices[0])

        self.generic_visit(node)

    def visit_AugAssign(self, node):
        target_visitor = DefUseVisitor()
        target_visitor.visit(node.target)
        self.uses.update(target_visitor.defs | target_visitor.uses)
        self.defs.update(target_visitor.defs)
        self.subscripts.update(target_visitor.subscripts)
        self.visit(node.value)


class DDGState:
    """Encapsulates traversal state with strict types to satisfy static analyzers."""

    def __init__(self, ddg: nx.DiGraph):
        self.ddg: nx.DiGraph = ddg
        self.last_def: dict[str, int] = {}
        self.last_uses: dict[str, set[int]] = {}
        self.slice_registry: dict[tuple[str, int], ast.AST] = {}
        self.stmt_counter: int = 0


def extract_linear_offset(node, loop_var: str):
    """
    Attempts to extract a linear offset for `loop_var` from an index AST node.
    E.g., for `i - 1` with loop_var `i`, returns -1. For `i`, returns 0.
    """
    if node is None:
        return 0
    if isinstance(node, ast.Name) and node.id == loop_var:
        return 0
    if isinstance(node, ast.BinOp):
        left_offset = extract_linear_offset(node.left, loop_var)
        right_offset = extract_linear_offset(node.right, loop_var)

        if left_offset is not None and isinstance(node.right,
                                                  ast.Constant):
            val = node.right.value
            if isinstance(node.op, ast.Add):
                return left_offset + val
            elif isinstance(node.op, ast.Sub):
                return left_offset - val
        elif isinstance(node.left,
                        ast.Constant) and right_offset is not None:
            val = node.left.value
            if isinstance(node.op, ast.Add):
                return val + right_offset
            elif isinstance(node.op, ast.Sub):
                return val - right_offset
    return None


def loop_in_slice(node, loop_var):
    """Helper to check if a loop variable exists anywhere inside an AST node/subscript."""
    if node is None:
        return False
    found = False
    class Finder(ast.NodeVisitor):
        def visit_Name(self, n):
            nonlocal found
            if n.id == loop_var:
                found = True
            self.generic_visit(n)
    Finder().visit(node)
    return found
import ast

def compute_dependence_vector(producer_slice, consumer_slice, active_loops):
    """
    Computes distance and direction vectors across active loop nest levels by
    collecting all variable names present within the array subscript slices.
    """
    if not active_loops:
        return None, None

    # Helper to gather names from either a single AST node or a list of nodes
    def collect_names(node_or_list, target_set):
        if node_or_list is None:
            return
        nodes = node_or_list if isinstance(node_or_list, list) else [node_or_list]
        for node in nodes:
            class NameCollector(ast.NodeVisitor):
                def visit_Name(self, n):
                    target_set.add(n.id)
                    self.generic_visit(n)
            NameCollector().visit(node)

    # Gather all variable names present across the subscript slice(s)
    subscript_vars = set()
    collect_names(producer_slice, subscript_vars)
    collect_names(consumer_slice, subscript_vars)

    distance_vector = []
    direction_vector = []

    is_self_dep = (producer_slice == consumer_slice)

    for loop_var in active_loops:
        present = loop_var in subscript_vars

        if is_self_dep:
            # For self-dependence: presence in subscript means distance 0,
            # absence means 1 (indicating the enclosing loop carries the reduction recurrence).
            dist = 0 if present else 1
        else:
            # Inter-statement dependence: use offset subtraction if linear
            prod_offset = extract_linear_offset(producer_slice, loop_var) if present else None
            cons_offset = extract_linear_offset(consumer_slice, loop_var) if present else None
            if prod_offset is not None and cons_offset is not None:
                dist = cons_offset - prod_offset
            else:
                dist = 0 if present else 1

        distance_vector.append(dist)

        if dist > 0:
            direction_vector.append('<')
        elif dist < 0:
            direction_vector.append('>')
        else:
            direction_vector.append('=')

    return tuple(distance_vector), tuple(direction_vector)


def generate_ddg(tree: ast.AST) -> nx.DiGraph:
    """
    Generates a Data Dependence Graph (DDG) from the given AST.
    """
    return generate_ddg_with_vectors(tree)


import ast
import networkx as nx
from swiftpack.ddg import DefUseVisitor, compute_dependence_vector
# Assuming your functions are imported or in the same module
# from swiftpack.ddg import generate_ddg, visualize_ddg
import ast
import textwrap
from pathlib import Path


def generate_ddg_with_vectors(node, active_loops=None, state=None):
    """
    Recursively traverses AST to build a DDG annotated with distance and direction vectors,
    correctly capturing both inter-statement and loop-carried self-dependencies.
    """
    if active_loops is None:
        active_loops = []

    if state is None:
        state = {
            'ddg': nx.DiGraph(),
            'last_def': {},
            'last_uses': {},
            'slice_registry': {},
            'stmt_counter': 0
        }

    ddg = state['ddg']

    if isinstance(node, ast.Module):
        body = node.body
    elif isinstance(node, list):
        body = node
    else:
        body = [node]

    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            generate_ddg_with_vectors(stmt.body, active_loops, state)
            continue

        if isinstance(stmt, ast.For):
            loop_var = stmt.target.id if isinstance(stmt.target,
                                                    ast.Name) else "unknown"
            new_loops = active_loops + [loop_var]
            generate_ddg_with_vectors(stmt.body, new_loops, state)
            continue

        if isinstance(stmt, (ast.If, ast.While)):
            generate_ddg_with_vectors(stmt.body, active_loops, state)
            if stmt.orelse:
                generate_ddg_with_vectors(stmt.orelse, active_loops,
                                          state)
            continue

        idx = state['stmt_counter']
        stmt_str = ast.unparse(stmt)
        ddg.add_node(idx, statement=stmt_str, ast_node=stmt,
                     loop_nest=active_loops.copy())

        visitor = DefUseVisitor()
        visitor.visit(stmt)

        for v, (base, slice_node) in visitor.subscripts.items():
            state['slice_registry'][(v, idx)] = slice_node

        stmt_defs = visitor.defs
        stmt_uses = visitor.uses

        # Helper to add dependence edge (supports self-edges when prev_idx == idx)
        def add_dep_edge(prev_idx, curr_idx, dep_type, v):
            prev_slice = state['slice_registry'].get((v, prev_idx))
            curr_slice = state['slice_registry'].get((v, curr_idx))

            # If it's a self-dependence (same statement across loop iterations),
            # use the same slice for both source and sink to compute iteration offsets.
            if prev_idx == curr_idx:
                if not active_loops:
                    return  # No loops to carry a self-dependence
                dist_vec, dir_vec = compute_dependence_vector(
                    curr_slice, curr_slice, active_loops)
            else:
                dist_vec, dir_vec = compute_dependence_vector(
                    prev_slice, curr_slice, active_loops)

            # Avoid adding duplicate edges if one already exists with the same attributes
            if not ddg.has_edge(prev_idx,
                                curr_idx) or ddg.get_edge_data(prev_idx,
                                                               curr_idx).get(
                    'variable') != v:
                ddg.add_edge(prev_idx, curr_idx, dep_type=dep_type,
                             variable=v,
                             distance_vector=dist_vec,
                             direction_vector=dir_vec)

        # 1. True Dependencies (RAW) & Self-RAW
        for v in stmt_uses:
            if v in state['last_def']:
                prev_idx = state['last_def'][v]
                add_dep_edge(prev_idx, idx, 'RAW', v)
            elif active_loops and v in stmt_defs:
                # Standalone statement with recurrence (e.g. C[i][j] = C[i][j] + ...)
                add_dep_edge(idx, idx, 'RAW', v)

        # 2. Output Dependencies (WAW) & Self-WAW
        for v in stmt_defs:
            if v in state['last_def']:
                prev_idx = state['last_def'][v]
                if prev_idx != idx:
                    add_dep_edge(prev_idx, idx, 'WAW', v)
            elif active_loops and v in stmt_uses:
                add_dep_edge(idx, idx, 'WAW', v)

        # Update state tracking
        for v in stmt_defs:
            state['last_def'][v] = idx
            state['last_uses'][v] = {idx}

        for v in stmt_uses:
            if v not in state['last_uses']:
                state['last_uses'][v] = set()
            state['last_uses'][v].add(idx)

        state['stmt_counter'] += 1

    return ddg

def visualize_ddg(ddg: nx.DiGraph, filepath: str):
    """
    Takes a NetworkX DDG with distance/direction vectors, visualizes the nodes and
    dependency edges, and saves the resulting graph image to the given file path.
    """
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(ddg, seed=42, k=0.9)

    node_labels = {
        node: f"[{node}] {data.get('statement', '')}"
        for node, data in ddg.nodes(data=True)
    }

    nx.draw_networkx_nodes(ddg, pos, node_size=1000,
                           node_color='lightblue', edgecolors='black')
    nx.draw_networkx_labels(ddg, pos, labels=node_labels, font_size=9,
                            font_weight='bold')

    edge_colors = []
    edge_labels = {}
    for u, v, data in ddg.edges(data=True):
        dep_type = data.get('dep_type', 'DEP')
        var = data.get('variable', '')
        dist = data.get('distance_vector', None)
        direction = data.get('direction_vector', None)

        label_str = f"{dep_type}: {var}"
        if dist is not None and direction is not None:
            label_str += f"\nDist: {dist} | Dir: {direction}"

        edge_labels[(u, v)] = label_str

        if dep_type == 'RAW':
            edge_colors.append('#1f77b4')
        elif dep_type == 'WAR':
            edge_colors.append('#ff7f0e')
        elif dep_type == 'WAW':
            edge_colors.append('#2ca02c')
        else:
            edge_colors.append('gray')

    nx.draw_networkx_edges(ddg, pos, edge_color=edge_colors, width=2.0,
                           arrowsize=20, connectionstyle='arc3,rad=0.2')
    nx.draw_networkx_edge_labels(ddg, pos, edge_labels=edge_labels,
                                 font_size=7, font_color='darkred')

    plt.title(
        "Loop-Aware Data Dependence Graph (DDG)\n(Blue: RAW, Orange: WAR, Green: WAW with Vectors)",
        fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"DDG visualization successfully saved to: {filepath}")




## main entry point for testing
if __name__ == "__main__":
    """Test DDG generation and visualization export for a matrix multiplication (matmul) loop nest."""
    code = """
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
    """

    # Safely remove common leading indentation while preserving nested structure
    cleaned_code = textwrap.dedent(code).strip()

    tree = ast.parse(cleaned_code)
    ddg = generate_ddg(tree)
    # print dependence edges with their distance and direction vectors for debugging
    for u, v, data in ddg.edges(data=True):
        dist = data.get('distance_vector')
        dir = data.get('direction_vector')
        print(f"Edge from {u} to {v}: distance_vector={dist}, direction_vector={dir}")

    print(
        f"Generated DDG with {ddg.number_of_nodes()} nodes and {ddg.number_of_edges()} edges.")

    # Export visualization to a local file in the current directory
    out_file = Path("matmul_ddg.png")
    visualize_ddg(ddg, str(out_file))

    print(f"Visualization successfully saved to: {out_file.resolve()}")
