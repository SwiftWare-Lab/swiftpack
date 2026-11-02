import ast
import pytest
import networkx as nx
from swiftpack.ddg import generate_ddg, visualize_ddg


def test_raw_dependency():
    """Test Read-After-Write (RAW / True dependency) on scalars."""
    code = """
x = 1
y = x + 1
"""
    tree = ast.parse(code.strip())
    ddg = generate_ddg(tree)

    assert ddg.number_of_edges() == 1
    edges = list(ddg.edges(data=True))
    u, v, data = edges[0]

    assert u == 0
    assert v == 1
    assert data['dep_type'] == 'RAW'
    assert data['variable'] == 'x'


def test_war_dependency():
    """Test Write-After-Read (WAR / Anti dependency) on scalars."""
    code = """
y = x + 1
x = 5
    """
    tree = ast.parse(code.strip())
    ddg = generate_ddg(tree)

    assert ddg.number_of_edges() == 1
    edges = list(ddg.edges(data=True))
    u, v, data = edges[0]

    assert u == 0
    assert v == 1
    assert data['dep_type'] == 'WAR'
    assert data['variable'] == 'x'


def test_waw_dependency():
    """Test Write-After-Write (WAW / Output dependency) on scalars."""
    code = """
x = 1
x = 2
    """
    tree = ast.parse(code.strip())
    ddg = generate_ddg(tree)

    assert ddg.number_of_edges() == 1
    edges = list(ddg.edges(data=True))
    u, v, data = edges[0]

    assert u == 0
    assert v == 1
    assert data['dep_type'] == 'WAW'
    assert data['variable'] == 'x'


def test_array_subscript_dependencies():
    """Test mixed RAW, WAR, and WAW dependencies with array subscripts."""
    code = """
A[i] = B[i]         # Stmt 0
C[i] = A[i] + 1     # Stmt 1: Reads A[i] (RAW from 0)
A[i] = 5            # Stmt 2: Writes A[i] (WAW from 0, WAR from 1)
    """
    tree = ast.parse(code.strip())
    ddg = generate_ddg(tree)

    # Extract edges as hashable tuples excluding dictionaries to avoid TypeError
    edge_set = {
        (u, v, data['dep_type'], data['variable'])
        for u, v, data in ddg.edges(data=True)
    }

    expected_edges = {
        (0, 1, 'RAW', 'A[i]'),
        (0, 2, 'WAW', 'A[i]'),
        (1, 2, 'WAR', 'A[i]')
    }

    assert edge_set == expected_edges


def test_self_assignment_no_false_war():
    """Test that self-increment/updates like x = x + 1 don't create false WAR hazards on themselves."""
    code = """
x = x + 1
    """
    tree = ast.parse(code.strip())
    ddg = generate_ddg(tree)

    assert ddg.number_of_edges() == 0



def test_matmul_visualization(tmp_path):
    """Test DDG generation and visualization export for a matrix multiplication (matmul) loop nest."""
    code = """
for i in range(N):
    for j in range(N):
        C[i][j] = 0.0
        for k in range(N):
            C[i][j] = C[i][j] + A[i][k] * B[k][j]
    """
    tree = ast.parse(code.strip())
    ddg = generate_ddg(tree)

    # Verify that the DDG successfully captured structural nodes and dependence edges for matmul
    assert ddg.number_of_nodes() > 0
    assert ddg.number_of_edges() > 0

    # Test visualization export to a temporary path provided by pytest
    out_file = tmp_path / "matmul_ddg.png"
    visualize_ddg(ddg, str(out_file))

    # Assert the file was successfully generated and written to disk
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_matmul_function_definition():
    """DDG generation should descend into function bodies as well as modules."""
    code = """
def matmul(A, B, C, N):
    for i in range(N):
        for j in range(N):
            C[i][j] = 0.0
            for k in range(N):
                C[i][j] = C[i][j] + A[i][k] * B[k][j]
"""
    tree = ast.parse(code.strip())
    ddg = generate_ddg(tree)

    assert ddg.number_of_nodes() > 0
    assert ddg.number_of_edges() > 0
    assert any(data['loop_nest'] == ['i', 'j', 'k']
               for _, data in ddg.nodes(data=True))