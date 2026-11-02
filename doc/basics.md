# Code Representation and Optimization with Numba

## Code Representation

Code representation bridges human-readable source code and executable 
target instructions. Compilers parse and lower source code through multiple 
levels of Intermediate Representation (IR) to analyze semantics and 
apply performance optimizations.

---

### Abstract Syntax Tree (AST)

An **Abstract Syntax Tree (AST)** is a hierarchical, tree-structured 
representation of source code produced during syntactic analysis. 
It strips away purely syntactic tokens (such as semicolons, explicit 
parentheses, and whitespace) while retaining structural semantics, 
operator precedence, and parent-child execution scopes.

* **Nodes:** Represent structural constructs such as operators 
 (`+`, `*`), statements (`if`, `while`), or expressions (`BinaryOp`).
* **Leaves:** Represent terminal operands, such as variable 
  names (`Name`) or literals (`Constant`).
* **Usage:** Used by parsers, linters, semantic analyzers, static 
security checkers, and source-to-source compilers.

---

### List of Instructions

Lowering an AST into a linear **List of Instructions** flattens the 
tree structure into sequential abstract machine instructions. This flat 
Intermediate Representation (IR) often uses **Three-Address Code (3AC)** 
or register-based bytecode instructions where each operation has at most three operands.

* **Characteristics:** Simple instruction semantics (e.g., `LOAD`, `ADD`, `STORE`, `JUMP_IF_FALSE`).
* **Usage:** Simplifies register allocation, instruction selection, and target machine code emission. In **Numba**, Python AST is lowered into Numba IR instructions (such as `jump`, `branch`, `call`, and assignment operations) before being passed to LLVM.

---

### Control Flow Graph (CFG)

A **Control Flow Graph (CFG)** is a directed graph $(V, E)$ that explicitly models all possible execution paths through a program unit (such as a function or subroutine).

* **Nodes ($V$):** Basic Blocks containing sequential instructions.
* **Edges ($E$):** Control transfers via conditional branches, unconditional jumps, or function calls.
* 
### Basic Blocks
Basic blocks are sequences of instructions with no branches except at the end. They are the building blocks of control flow graphs.

CFG is applied to basic blocks.

## Code Optimization Techniques

Depending on the scope of optimization, code optimization is either **Local Optimization** (operating within a single basic block) or **Global Optimization** (operating across basic blocks within a CFG or across function boundaries via Interprocedural Analysis).

                    ┌────────────────────────────────────────┐
                    │      Code Optimization Scope           │
                    └───────────────────┬────────────────────┘
                                        │
              ┌─────────────────────────┴─────────────────────────┐
              ▼                                                   ▼
   ┌────────────────────┐                              ┌────────────────────┐
   │ Local Optimization │                              │ Global Optimization│
   │ (Single Basic Block│                              │ (Across CFG /      │
   │      Scope)        │                              │ Interprocedural)   │
   └────────────────────┘                              └────────────────────┘

Below is the classification of optimization techniques across four fundamental domains:

---

### 1. Loop Optimization

Loops represent the highest execution-density areas in programs. Optimizing loop bounds, array index access, and iteration order yields significant performance gains.

* **Loop Invariant Code Motion (Hoisting):** Moves computations whose values do not change across iterations outside the loop header.
* **Loop Unrolling:** Replicates the loop body to decrease loop overhead (branch checks, induction increment) and expose instruction-level parallelism (ILP).
* **Loop Tiling (Blocking):** Reorganizes loop access patterns to fit within CPU L1/L2 cache lines, minimizing cache misses.
* **Loop Exchange (Permutation):** Reorders nested loops to improve data locality and vectorization opportunities.

---

### 2. Data Structure Optimization

Optimizes memory access patterns, layout, and allocation overheads.

* **Structure-of-Arrays (SoA) vs. Array-of-Structures (AoS):** Reorganizes struct memory layouts to optimize spatial cache locality and improve vectorization.
* **Scalar Replacement of Aggregates (SRA):** Replaces aggregate objects/tuples with independent scalar variables that can fit directly in CPU registers.
* **Sparsity Exploitation:** Detects and optimizes operations on sparse data structures (e.g., sparse matrices) to avoid unnecessary computations and memory accesses.

---

### 3. Function Optimization

Reduces call-stack overhead and optimizes calling conventions.

* **Function Inlining:** Replaces a function call site with the body of the target function, eliminating call/return stack frames and enabling downstream compiler optimizations.
* **Dead Argument Elimination:** Removes unused parameters from function signatures to eliminate unnecessary argument passing in registers or stack frames.
* **Tail Call Optimization (TCO):** Replaces recursive function call stack frames with jumps when the call is in the tail position, converting recursion to $O(1)$ stack space.

---

### 4. Logic & Expression Optimization

Simplifies arithmetic and logical execution paths.

* **Constant Folding & Propagation:** Pre-calculates constant operations at compile time (e.g., `3 * 8` $\rightarrow$ `24`) and propagates known constant values through expressions.
* **Dead Code Elimination (DCE):** Identifies and removes unreachable blocks or instructions whose outputs are never read.
* **Strength Reduction:** Replaces expensive operations with faster equivalent operations (e.g., replacing `x * 8` with bitwise shift `x << 3`, or `x ** 2` with `x * x`).
* **Common Subexpression Elimination (CSE):** Detects redundant expressions and reuses previously computed values instead of recalculating them.