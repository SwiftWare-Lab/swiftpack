# Local Analysis and Local Value Numbering

Local analysis reasons about one basic block (a straight-line run of
statements with a single entry and exit) at a time. Nothing learned in a block
is assumed to hold in another one, which keeps the analysis cheap and safe.

## Enabling it: the `lvn` directive

`LocalValueNumbering` (`../../src/swiftpack/lvn.py`) is an ordinary `Transformation`.
It is scheduled with the `lvn` directive in `scheduler.py`:

```python
(LocalValueNumbering, {"directive": "lvn"})
```

`check_legality` returns `True` only when `directive == "lvn"` and the tree is
an `ast.Module` or `ast.FunctionDef`. Otherwise `TransformationPipeline` logs
`ILLEGAL` and skips the pass. `LocalAnalysis` remains as an alias of the class.

`AutoTileAndParallelizeScheduler` runs it first, so later passes see
simplified code.

## How the code works

`LocalValueNumberingTransformer` is an `ast.NodeTransformer`.

**Scopes.** A stack of tables is kept. Each table holds a
`copy_propagation_table` (name -> `Name`/`Constant`) and an
`expression_value_table` (expression signature -> temporary name). Entering a
function, `if`/`else` branch, loop body, or `with` body pushes a fresh table,
and leaving pops it, so facts never cross block boundaries.

**Optimizations**, applied per assignment:

1. *Copy/constant propagation*: after `a = b` or `a = 10`, later loads of `a`
   are replaced (alias chains are followed).
2. *Constant folding*: arithmetic, unary, and comparison operations on
   constants are evaluated at compile time.
3. *Algebraic simplification*: `x + 0`, `x * 1`, `x * 0`, `x / 1`, `x ** 1`,
   `x ** 0`, and similar.
4. *Common subexpression elimination (value numbering)*: each non-trivial
   expression is keyed by `ast.dump` of its structure. The first occurrence is
   assigned to a `__swiftpack_lvn_N` temporary; later occurrences in the same
   block reuse it.
5. *Dead code elimination*: assignments to names that are never loaded and
   whose right-hand side is pure (names, constants, unary/binary ops) are
   removed.

Example:

```python
x = a * b            __swiftpack_lvn_0 = a * b
y = a * b     ->     x = __swiftpack_lvn_0
return x + y         return x + __swiftpack_lvn_0
```

## Limitations

- Signatures are structural, so `a + b` and `b + a` are different values.
- Assignments do not invalidate cached expressions that read the reassigned
  variable; this is safe only for single-assignment code within a block.
- Dead code elimination only considers loads within the same statement list.
