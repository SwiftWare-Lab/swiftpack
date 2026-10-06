import ast
from swiftpack.transformation import Transformation


class LocalAnalysis(Transformation):
    """Performs local value numbering and local optimization in each basic block."""

    def check_legality(self, tree: ast.AST, **kwargs) -> bool:
        return True

    def apply(self, tree: ast.AST, **kwargs) -> ast.AST:
        transformer = LocalValueNumberingTransformer()
        transformed = transformer.visit(tree)
        ast.fix_missing_locations(transformed)
        return transformed


class LocalValueNumberingTransformer(ast.NodeTransformer):
    """Local SSA-like optimization pass operating within basic block scope.

    Applies the following optimizations:
    1. Copy & Constant Propagation
    2. Constant Folding (Arithmetic, Unary, and Boolean)
    3. Algebraic Identity Simplification
    4. Common Subexpression Elimination (CSE) via Local Value Numbering
    """

    def __init__(self):
        # Stack of basic-block scopes to prevent optimization bleed across boundaries
        self.scope_stack = []
        self.temp_variable_counter = 0

    # -------------------------------------------------------------------------
    # Scope & Table Management
    # -------------------------------------------------------------------------

    def _create_scope_table(self):
        """Creates a fresh LVN mapping table for a new basic block scope.

        - copy_propagation_table: Maps variable identifiers to their known constant or
          aliased source variable.
        - expression_value_table: Maps structural expression signatures (hashes) to
          the temporary variable holding their computed result.
        """
        return {
            "copy_propagation_table": {},
            "expression_value_table": {},
        }

    def _get_active_scope(self):
        if not self.scope_stack:
            self.scope_stack.append(self._create_scope_table())
        return self.scope_stack[-1]

    def _enter_block_scope(self):
        self.scope_stack.append(self._create_scope_table())

    def _exit_block_scope(self):
        if self.scope_stack:
            self.scope_stack.pop()

    def _generate_temporary_variable(self):
        temp_name = f"__swiftpack_lvn_{self.temp_variable_counter}"
        self.temp_variable_counter += 1
        return temp_name

    def _is_terminal_value(self, node):
        """Checks if a node is an atomic value that does not require expression extraction."""
        return isinstance(node, (ast.Name, ast.Constant, ast.Tuple, ast.List, ast.Set, ast.Dict))

    # -------------------------------------------------------------------------
    # 1. Copy & Constant Propagation Phase
    # -------------------------------------------------------------------------

    def _resolve_copy_propagation(self, variable_name, scope):
        """Resolves recursive alias chains and propagated constants.

        For example: `a = b`, `b = 10` -> resolves `a` to `ast.Constant(value=10)`.
        """
        visited_aliases = set()
        current_name = variable_name
        copy_table = scope["copy_propagation_table"]

        while current_name in copy_table and current_name not in visited_aliases:
            visited_aliases.add(current_name)
            replacement = copy_table[current_name]
            if isinstance(replacement, ast.Name):
                current_name = replacement.id
            else:
                return replacement

        if current_name != variable_name:
            return ast.Name(id=current_name, ctx=ast.Load())
        return None

    # -------------------------------------------------------------------------
    # 2. Common Subexpression Elimination (CSE) Phase
    # -------------------------------------------------------------------------

    def _compute_expression_signature(self, expr_node):
        """Generates a canonical structural signature for an expression node."""
        return ast.dump(expr_node, include_attributes=False)

    def _eliminate_common_subexpression(self, expr_node, scope):
        """Performs Common Subexpression Elimination using the expression value table.

        If the expression was previously computed in the same basic block:
            Replaces the expression with the existing temporary variable.
        Else:
            Creates a temporary variable, emits an assignment statement, and updates the table.
        """
        if self._is_terminal_value(expr_node):
            return expr_node, []

        expr_signature = self._compute_expression_signature(expr_node)
        expr_table = scope["expression_value_table"]

        if expr_signature in expr_table:
            reused_temp_name = expr_table[expr_signature]
            return ast.copy_location(ast.Name(id=reused_temp_name, ctx=ast.Load()), expr_node), []

        new_temp_name = self._generate_temporary_variable()
        expr_table[expr_signature] = new_temp_name

        hoisted_assignment = ast.Assign(
            targets=[ast.copy_location(ast.Name(id=new_temp_name, ctx=ast.Store()), expr_node)],
            value=ast.copy_location(expr_node, expr_node),
        )
        return ast.copy_location(ast.Name(id=new_temp_name, ctx=ast.Load()), expr_node), [hoisted_assignment]

    # -------------------------------------------------------------------------
    # 3. Constant Folding Evaluation Helpers
    # -------------------------------------------------------------------------

    def _evaluate_binary_operation(self, operator, left_value, right_value):
        try:
            if isinstance(operator, ast.Add): return left_value + right_value
            if isinstance(operator, ast.Sub): return left_value - right_value
            if isinstance(operator, ast.Mult): return left_value * right_value
            if isinstance(operator, ast.Div): return left_value / right_value
            if isinstance(operator, ast.FloorDiv): return left_value // right_value
            if isinstance(operator, ast.Mod): return left_value % right_value
            if isinstance(operator, ast.Pow): return left_value ** right_value
            if isinstance(operator, ast.BitAnd): return left_value & right_value
            if isinstance(operator, ast.BitOr): return left_value | right_value
            if isinstance(operator, ast.BitXor): return left_value ^ right_value
            if isinstance(operator, ast.LShift): return left_value << right_value
            if isinstance(operator, ast.RShift): return left_value >> right_value
        except Exception:
            return None
        return None

    def _evaluate_unary_operation(self, operator, operand_value):
        try:
            if isinstance(operator, ast.UAdd): return +operand_value
            if isinstance(operator, ast.USub): return -operand_value
            if isinstance(operator, ast.Not): return not operand_value
            if isinstance(operator, ast.Invert): return ~operand_value
        except Exception:
            return None
        return None

    def _fold_boolean_operations(self, expr_node):
        if isinstance(expr_node, ast.BoolOp):
            evaluated_values = [self._optimize_expression(item, self._get_active_scope()) for item in expr_node.values]
            if all(isinstance(v, ast.Constant) for v in evaluated_values):
                if isinstance(expr_node.op, ast.And):
                    return ast.Constant(value=all(bool(v.value) for v in evaluated_values))
                if isinstance(expr_node.op, ast.Or):
                    return ast.Constant(value=any(bool(v.value) for v in evaluated_values))
        return expr_node

    # -------------------------------------------------------------------------
    # 4. Expression Simplification & Algebraic Optimization Pipeline
    # -------------------------------------------------------------------------

    def _simplify_and_fold_expression(self, node):
        """Applies compile-time constant folding and algebraic identity simplifications."""
        # --- Binary Operations ---
        if isinstance(node, ast.BinOp):
            left = self._optimize_expression(node.left, self._get_active_scope())
            right = self._optimize_expression(node.right, self._get_active_scope())
            node.left, node.right = left, right

            # Compile-time Constant Folding
            if isinstance(left, ast.Constant) and isinstance(right, ast.Constant):
                folded_value = self._evaluate_binary_operation(node.op, left.value, right.value)
                if folded_value is not None:
                    return ast.Constant(value=folded_value)

            # Algebraic Identity Simplifications
            if isinstance(node.op, ast.Add):
                if isinstance(right, ast.Constant) and right.value == 0: return left
                if isinstance(left, ast.Constant) and left.value == 0: return right

            if isinstance(node.op, ast.Sub):
                if isinstance(right, ast.Constant) and right.value == 0: return left

            if isinstance(node.op, ast.Mult):
                if isinstance(right, ast.Constant) and right.value == 1: return left
                if isinstance(left, ast.Constant) and left.value == 1: return right
                if isinstance(right, ast.Constant) and right.value == 0: return ast.Constant(value=0)
                if isinstance(left, ast.Constant) and left.value == 0: return ast.Constant(value=0)

            if isinstance(node.op, ast.Div):
                if isinstance(right, ast.Constant) and right.value == 1: return left

            if isinstance(node.op, ast.Pow):
                if isinstance(right, ast.Constant) and right.value == 1: return left
                if isinstance(right, ast.Constant) and right.value == 0: return ast.Constant(value=1)

            return node

        # --- Unary Operations ---
        if isinstance(node, ast.UnaryOp):
            operand = self._optimize_expression(node.operand, self._get_active_scope())
            node.operand = operand

            if isinstance(operand, ast.Constant):
                folded_value = self._evaluate_unary_operation(node.op, operand.value)
                if folded_value is not None:
                    return ast.Constant(value=folded_value)

            if isinstance(node.op, ast.USub):
                if isinstance(operand, ast.Constant) and operand.value == 0:
                    return ast.Constant(value=0)
            return node

        # --- Comparison Operations ---
        if isinstance(node, ast.Compare):
            left = self._optimize_expression(node.left, self._get_active_scope())
            comparators = [self._optimize_expression(c, self._get_active_scope()) for c in node.comparators]
            node.left, node.comparators = left, comparators

            if isinstance(left, ast.Constant) and all(isinstance(c, ast.Constant) for c in comparators):
                values = [left.value] + [c.value for c in comparators]
                comparison_result = True
                for i in range(len(node.ops)):
                    op = node.ops[i]
                    l_val, r_val = values[i], values[i + 1]
                    if isinstance(op, ast.Eq): ok = l_val == r_val
                    elif isinstance(op, ast.NotEq): ok = l_val != r_val
                    elif isinstance(op, ast.Lt): ok = l_val < r_val
                    elif isinstance(op, ast.LtE): ok = l_val <= r_val
                    elif isinstance(op, ast.Gt): ok = l_val > r_val
                    elif isinstance(op, ast.GtE): ok = l_val >= r_val
                    else: ok = True
                    comparison_result = comparison_result and ok
                return ast.Constant(value=comparison_result)
            return node

        return node

    def _optimize_expression(self, node, scope):
        """Main recursive expression optimizer."""
        if isinstance(node, ast.Name):
            propagated_alias = self._resolve_copy_propagation(node.id, scope)
            if propagated_alias is not None:
                return self._optimize_expression(propagated_alias, scope)
            return node

        for field_name, field_value in list(ast.iter_fields(node)):
            if isinstance(field_value, list):
                rewritten_list = []
                for item in field_value:
                    if isinstance(item, ast.AST):
                        rewritten_list.append(self._optimize_expression(item, scope))
                    else:
                        rewritten_list.append(item)
                setattr(node, field_name, rewritten_list)
            elif isinstance(field_value, ast.AST):
                setattr(node, field_name, self._optimize_expression(field_value, scope))

        return self._simplify_and_fold_expression(node)

    # -------------------------------------------------------------------------
    # Block Traversal Helpers
    # -------------------------------------------------------------------------

    def _process_statement_sequence(self, statement_body):
        processed_statements = []
        for stmt in statement_body:
            rewritten_stmt = self.visit(stmt)
            if rewritten_stmt is None:
                continue
            if isinstance(rewritten_stmt, list):
                processed_statements.extend(rewritten_stmt)
            else:
                processed_statements.append(rewritten_stmt)
        return processed_statements

    # -------------------------------------------------------------------------
    # AST Visitor Control Flow Drivers
    # -------------------------------------------------------------------------

    def visit_Module(self, node):
        self._enter_block_scope()
        node.body = self._process_statement_sequence(node.body)
        self._exit_block_scope()
        return node

    def visit_FunctionDef(self, node):
        self._enter_block_scope()
        node.body = self._process_statement_sequence(node.body)
        self._exit_block_scope()
        return node

    def visit_AsyncFunctionDef(self, node):
        self._enter_block_scope()
        node.body = self._process_statement_sequence(node.body)
        self._exit_block_scope()
        return node

    def visit_If(self, node):
        node.test = self._optimize_expression(node.test, self._get_active_scope())
        self._enter_block_scope()
        node.body = self._process_statement_sequence(node.body)
        self._exit_block_scope()

        self._enter_block_scope()
        node.orelse = self._process_statement_sequence(node.orelse)
        self._exit_block_scope()
        return node

    def visit_For(self, node):
        node.iter = self._optimize_expression(node.iter, self._get_active_scope())
        node.target = self._optimize_expression(node.target, self._get_active_scope())
        self._enter_block_scope()
        node.body = self._process_statement_sequence(node.body)
        self._exit_block_scope()

        self._enter_block_scope()
        node.orelse = self._process_statement_sequence(node.orelse)
        self._exit_block_scope()
        return node

    def visit_While(self, node):
        node.test = self._optimize_expression(node.test, self._get_active_scope())
        self._enter_block_scope()
        node.body = self._process_statement_sequence(node.body)
        self._exit_block_scope()

        self._enter_block_scope()
        node.orelse = self._process_statement_sequence(node.orelse)
        self._exit_block_scope()
        return node

    def visit_With(self, node):
        for item in node.items:
            if isinstance(item, ast.withitem):
                item.context_expr = self._optimize_expression(item.context_expr, self._get_active_scope())
                if item.optional_vars is not None:
                    item.optional_vars = self._optimize_expression(item.optional_vars, self._get_active_scope())
        self._enter_block_scope()
        node.body = self._process_statement_sequence(node.body)
        self._exit_block_scope()
        return node

    # -------------------------------------------------------------------------
    # AST Visitor Statement Nodes
    # -------------------------------------------------------------------------

    def visit_Assign(self, node):
        scope = self._get_active_scope()
        node.value = self._optimize_expression(node.value, scope)

        if not isinstance(node.value, (ast.Name, ast.Constant)):
            node.value, hoisted_prefix = self._eliminate_common_subexpression(node.value, scope)
            if hoisted_prefix:
                rewritten_targets = [self._optimize_expression(target, scope) for target in node.targets]
                hoisted_assignment = ast.copy_location(ast.Assign(targets=rewritten_targets, value=node.value), node)
                return hoisted_prefix + [hoisted_assignment]

        rewritten_targets = []
        for target in node.targets:
            rewritten_target = self._optimize_expression(target, scope)
            rewritten_targets.append(rewritten_target)
            if isinstance(rewritten_target, ast.Name) and isinstance(node.value, (ast.Name, ast.Constant)):
                # Record copy mapping: target_variable -> constant_or_alias
                scope["copy_propagation_table"][rewritten_target.id] = node.value

        node.targets = rewritten_targets
        return ast.copy_location(ast.Assign(targets=node.targets, value=node.value), node)

    def visit_AnnAssign(self, node):
        scope = self._get_active_scope()
        node.value = self._optimize_expression(node.value, scope) if node.value is not None else None
        if isinstance(node.target, ast.Name) and node.value is not None and isinstance(node.value, (ast.Name, ast.Constant)):
            scope["copy_propagation_table"][node.target.id] = node.value
        return node

    def visit_Return(self, node):
        if node.value is not None:
            node.value = self._optimize_expression(node.value, self._get_active_scope())
        return node

    def visit_Expr(self, node):
        node.value = self._optimize_expression(node.value, self._get_active_scope())
        return node

    def visit_AugAssign(self, node):
        scope = self._get_active_scope()
        node.target = self._optimize_expression(node.target, scope)
        node.value = self._optimize_expression(node.value, scope)
        return node

    def visit_Name(self, node):
        scope = self._get_active_scope()
        propagated_alias = self._resolve_copy_propagation(node.id, scope)
        if propagated_alias is not None:
            return self._optimize_expression(propagated_alias, scope)
        return node