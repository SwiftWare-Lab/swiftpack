# Extending SwiftPack Compiler


## Adding a pass

1. Implement a `Transformation` subclass with `check_legality` and `apply`
   methods that accept the AST and any pass-specific keyword arguments.
2. Import the class in `scheduler.py` and include
   `(YourTransformation, {"option": value})` in the desired position in a
   scheduler's returned plan. The kwargs are passed to both methods.
3. The `apply` method should return the updated AST. The pipeline feeds that
   result into the next legality check and transformation.
