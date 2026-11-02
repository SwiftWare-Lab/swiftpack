# Loop optimization

## Loop exchange
The directive `loop_exchange` is used to reorder nested loops. It 
takes a tuple of loop variable names, e.g., `("j", "k")`, and reorders 
the loops to match that order.

### Directive syntax
The scheduler tuple for the scheduler is:
```
   (LoopExchange, {"target_loops": ("j", "k")})
```
where `target_loops` is a tuple of loop variable names to reorder. 
The scheduler user directive is:
```
   @swiftpack(scheduler=[(LoopExchange, {"target_loops": ("j", "k")})])
```

### Legality check: 
The loop exchange transformation is legal if the specified loop variables 
exist within the nested loop hierarchy and the transformation does not 
violate any data dependencies. The legality checker generates a Data Dependence 
Graph (DDG) from the AST and evaluates all dependence distance vectors. 
Swapping two loop levels is proven legal if and only if **no dependence vector 
becomes lexicographically negative** (i.e., the first non-zero element in 
the permuted distance vector remains non-negative) after the exchange.

### Transformation:
The transformation reorders the nested loops according to the specified
order of loop variables. It updates the AST to reflect the new loop order,
ensuring that the loop bodies are correctly nested and that the loop
variables are updated accordingly.

## Loop Tiling
The directive loop_tiling is used to block or tile loops (combining 
strip-mining and loop permutation) to dramatically improve cache 
locality and spatial reuse. It takes a target loop variable name 
(e.g., "i") and an optional block size (e.g., block_size=32).

### Directive syntax 
The scheduler tuple for the scheduler is: 
```
   (LoopTiling, {"target_loop": "i", "tile_size": 32})
```
where `target_loop` is the loop variable to tile and `tile_size` is the
number of iterations per tile. The default tile size is 32.
The scheduler user directive is: 
```
   @swiftpack(scheduler=[(LoopTiling, {"target_loop": "i", "tile_size": 32})])
```

### Legality check:
The loop tiling transformation is legal if the target loop exists within 
the loop hierarchy and carries no backward (negative) distance 
dependencies. The legality checker generates a Data Dependence Graph (DG) 
and evaluates the distance vector components corresponding to the target 
loop level. Tiling is proven safe if all dependence vector entries for 
the targeted dimension are greater than or equal to zero ($\ge 0$), 
guaranteeing that block execution order does not violate program 
semantics or causality.

### Transformation:
The transformation performs strip-mining on the target loop, splitting 
it into two nested loops:
* Outer Tile Loop (_tile): Steps through the iteration space in 
  increments equal to the specified block_size.
* Inner Point Loop: Iterates locally within the current tile block, 
  bounded by the tile boundaries. bounded by min(tile_start + 
  block_size, upper_bound).
The original loop body is safely nested inside the inner loop, 
preserving all variable bindings and computational logic.

## Loop Parallelism
The directive LoopParallelism is used to enable multithreaded execution 
across iterations of a target loop (such as lowering to OpenMP or 
parallel runtimes via prange). It takes a loop variable name, e.g., 
"i".

### Directive syntax
The scheduler tuple for the scheduler is:
```
   (LoopParallelism, {"loop_var": "i"})
```
where `loop_var` is the loop variable to parallelize. The scheduler user 
directive is:
```
   @swiftpack(scheduler=[(LoopParallelism, {"loop_var": "i"})])
```

### Legality check:
The loop parallelism transformation is legal if and only if the target 
loop carries no loop-carried dependencies. The legality checker analyzes 
the Data Dependence Graph (DDG) and inspects the distance vector 
components at the depth of the target loop variable. If all distance 
entries for that loop level are equal to 0 (indicating no 
cross-iteration hazards), the loop is proven safe for parallel 
execution. Any non-zero distance indicates a data race hazard, causing 
the legality check to fail.

### Transformation:
The transformation searches the AST for the For loop matching the 
specified loop_var and replaces its iterator function from sequential 
range() to parallel prange(), enabling concurrent scheduling across 
available thread pools or hardware execution lanes.

## Loop unrolling
The directive LoopUnrolling is used to replicate the body of a loop 
multiple times, reducing branch overhead and expanding Instruction-Level 
Parallelism (ILP). It takes a target loop variable name (e.g., "k") and 
an unrolling factor (e.g., factor=4).

### Directive syntax
The scheduler tuple for the scheduler is:
```
   (LoopUnrolling, {"loop_var": "k", "factor": 4})
```
where `loop_var` is the loop variable to unroll and `factor` is the
number of times to replicate the loop body. The scheduler user directive 
is:
```
   @swiftpack(scheduler=[(LoopUnrolling, {"loop_var": "k", "factor": 4})])
``` 
### Legality check:
Standard scalar loop unrolling is always legal for inner loops because 
duplicating independent scalar iteration bodies does not alter data 
dependence distances or execution ordering across iterations. The 
legality check verifies that the specified loop_var exists in the 
targeted loop scope.

### Transformation:
The transformation modifies the loop AST in two ways:
1. Stride Scaling: Multiplies the step size of the loop's range() 
iterator by the specified factor (e.g., stepping by 4 instead of 1).
2. Body Duplication & Index Offsetting: Replicates the loop body 
   factor times, substituting the loop variable with static index 
   offsets (i, i + 1, i + 2, etc.) in subsequent unrolled copies.

## Loop fusion
 TODO: 
