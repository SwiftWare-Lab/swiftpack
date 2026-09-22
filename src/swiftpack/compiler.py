import ast
import inspect
import textwrap
from typing import Any, Dict, List, Tuple, Type
import numpy as np
from numba import njit, prange

from swiftpack.scheduler import Scheduler, AutoTileAndParallelizeScheduler



# ===========================================================================
# 4. Pipeline Execution & Legality Engine
# ===========================================================================

class TransformationPipeline:
    """Executes a schedule plan, enforcing legality prior to every AST pass."""

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    def execute(self, tree: ast.AST, verbose: bool = True) -> ast.AST:
        # Query scheduler for transformation array
        plan = self.scheduler.schedule(tree)
        curr_ast = tree

        for step_idx, (trans_cls, kwargs) in enumerate(plan):
            transformation = trans_cls()
            trans_name = trans_cls.__name__

            # --- Legality Check Pass ---
            is_legal = transformation.check_legality(curr_ast, **kwargs)

            if not is_legal:
                if verbose:
                    print(f"[Pipeline] Step {step_idx + 1}: {trans_name}({kwargs}) -> ILLEGAL. Skipping.")
                continue

            # --- Transformation Application Pass ---
            curr_ast = transformation.apply(curr_ast, **kwargs)

            if verbose:
                print(f"[Pipeline] Step {step_idx + 1}: {trans_name}({kwargs}) -> APPLIED successfully.")

        return curr_ast


# ===========================================================================
# 5. Decorator Infrastructure
# ===========================================================================

class SwiftPackCompiler:
    """Orchestrates source parsing, scheduling, AST rewriting, and Numba JIT execution."""

    def __init__(self, scheduler: Scheduler, njit_kwargs: dict = None):
        self.scheduler = scheduler
        self.pipeline = TransformationPipeline(scheduler)
        self.njit_kwargs = njit_kwargs or {"parallel": True, "fastmath": True}

    def __call__(self, fn):
        # 1. Obtain AST from function source
        source = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(source)

        # 2. Strip @swiftpack decorator from the FunctionDef node
        func_def = tree.body[0]
        func_def.decorator_list = [
            dec for dec in func_def.decorator_list
            if not (
                    (isinstance(dec, ast.Name) and dec.id == 'swiftpack') or
                    (isinstance(dec, ast.Attribute) and dec.attr == 'swiftpack') or
                    (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == 'swiftpack') or
                    (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == 'swiftpack')
            )
        ]

        # 3. Execute legal AST transformations
        transformed_ast = self.pipeline.execute(tree)

        # 4. Guarantee top-level ast.Module type for compile()
        if not isinstance(transformed_ast, ast.Module):
            if isinstance(transformed_ast, ast.stmt):
                transformed_ast = ast.Module(body=[transformed_ast], type_ignores=[])
            elif isinstance(transformed_ast, list):
                # Handle cases where a transformation returned a list of statement nodes
                stmts = [node for node in transformed_ast if isinstance(node, ast.stmt)]
                transformed_ast = ast.Module(body=stmts, type_ignores=[])
            else:
                raise TypeError(f"Cannot wrap non-statement AST node of type {type(transformed_ast)} in ast.Module")

        ast.fix_missing_locations(transformed_ast)

        # 5. Compile into bytecode
        code_obj = compile(transformed_ast, filename="<swiftpack>", mode="exec")

        # 6. Execute compiled bytecode to extract transformed function handle
        local_scope = {}
        global_scope = fn.__globals__.copy()
        global_scope["prange"] = prange

        exec(code_obj, global_scope, local_scope)
        transformed_fn = local_scope[fn.__name__]

        # 7. Return Numba JIT handle
        return njit(**self.njit_kwargs)(transformed_fn)


def swiftpack(fn=None, *, scheduler: Scheduler = None, **njit_kwargs):
    """Decorator interface taking a custom Scheduler strategy."""
    if scheduler is None:
        scheduler = AutoTileAndParallelizeScheduler()

    compiler = SwiftPackCompiler(scheduler, njit_kwargs)

    if fn is None:
        return compiler
    return compiler(fn)
