# SwiftPack Overview

## Transformation Pipeline

`@swiftpack` compiles a decorated function by retrieving its Python source,
parsing it into an AST, and removing the `@swiftpack` decorator from the
function definition. The configured scheduler creates an ordered transformation
plan. For each step, the pipeline instantiates the transformation class and
checks legality against the current AST. Illegal steps are logged and skipped;
legal steps are applied in order and logged as successful.

After the passes run, the compiler ensures the result is an `ast.Module`,
fixes missing source locations, compiles the AST, and executes it to retrieve
the transformed function. The function is then wrapped with Numba's `njit`.
The compiler makes `prange` available to the transformed function's globals.

## Scheduler

A scheduler implements `schedule(tree)` and returns an ordered list of
`(TransformationClass, kwargs)` pairs. The tree is the input function's AST;
each kwargs dictionary is passed to that transformation's `check_legality`
and, if legal, its `apply` method.

`AutoTileAndParallelizeScheduler` is the default. It schedules these passes in
order:

1. `LocalValueNumbering` with `directive="lvn"`.
2. `LoopExchange` with `target_loops=("j", "k")`, to reorder loops from IJK to
   IKJ for cache locality.
3. `LoopTiling` with `tile_size=32`.
4. `ParallelizeLoop` with `loop_var="i"`, to parallelize the outermost `i`
   loop.

Use the `scheduler` argument to `@swiftpack` to provide a custom scheduler.
The decorator uses `parallel=True` and `fastmath=True` for Numba by default;
additional keyword arguments are passed to `njit`.

## Supported Transformations

The default schedule includes `LocalValueNumbering`, `LoopExchange`,
`LoopTiling`, and `ParallelizeLoop`. A transformation must implement
`check_legality(tree, **kwargs)` and `apply(tree, **kwargs)`. The pipeline only
applies a pass when its legality check returns true.

## Adding a pass

1. Implement a `Transformation` subclass with `check_legality` and `apply`
   methods that accept the AST and any pass-specific keyword arguments.
2. Import the class in `scheduler.py` and include
   `(YourTransformation, {"option": value})` in the desired position in a
   scheduler's returned plan. The kwargs are passed to both methods.
3. The `apply` method should return the updated AST. The pipeline feeds that
   result into the next legality check and transformation.
