import ast


# ===========================================================================
# 1. Base Classes: Transformations & Schedulers
# ===========================================================================

class Transformation:
    """Base class for AST transformations with explicit legality validation."""

    def check_legality(self, tree: ast.AST, **kwargs) -> bool:
        """Analyze loop bounds, variable usage, or data dependencies to prove safety."""
        raise NotImplementedError

    def apply(self, tree: ast.AST, **kwargs) -> ast.AST:
        """Mutates and returns the updated AST."""
        raise NotImplementedError
