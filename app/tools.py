
def calculator(expression: str) -> str:
    """计算两个数字的加、减、乘、除。"""
    import ast
    import operator

    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Div) and right == 0:
                raise ValueError("除数不能为零")

            return operators[type(node.op)](left, right)

        # 拒绝函数调用、变量、幂运算等其他表达式
        raise ValueError("只支持数字之间的加、减、乘、除")

    try:
        tree = ast.parse(expression, mode="eval")
        result = evaluate(tree)
        return str(result)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError):
        raise ValueError("表达式无效，请输入合法的四则运算表达式")
