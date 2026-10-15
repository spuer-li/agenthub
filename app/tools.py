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
    except SyntaxError:
        raise ValueError("表达式语法错误，请输入合法的四则运算表达式")
    except ValueError as exc:
        if str(exc) == "除数不能为零":
            raise
        raise ValueError("表达式无效，请输入合法的四则运算表达式") from exc   

def text_stats(text: str) -> str:
    """统计文本的字符数、非空白字符数和单词数。"""
    import json

    return json.dumps(
        {
            "character_count": len(text),
            "non_whitespace_count": len("".join(text.split())),
            "word_count": len(text.split()),
        },
        ensure_ascii=False,
    )


# 工具注册表：工具名称 -> Python 函数
TOOL_REGISTRY = {
    "calculator": {
        "function": calculator,
        "argument": "expression",
        "description": "计算数学四则运算。当用户需要计算时使用。",
        "parameter_description": "数学表达式，例如 125 * 36"
    },
    "text_stats": {
        "function": text_stats,
        "argument": "text",
        "description": (
            "统计一段文本的字符总数、非空白字符数和单词数。"
            "当用户要求是统计文本长度或单词数量时使用。"
        ),
        "parameter_description": "需要统计的文本内容"
    }
}


def execute_tool(tool_name: str, arguments: dict) -> str:
    """根据工具名称找到并执行对应的 Python 函数。"""
    if not isinstance(arguments, dict):
        raise ValueError("工具参数必须是字典")

    tool_config = TOOL_REGISTRY.get(tool_name)

    if tool_config is None:
        raise ValueError(f"不支持的工具：{tool_name}")
    
    argument_name = tool_config["argument"]
    argument_value = arguments.get(argument_name)

    if not isinstance(argument_value, str):
        raise ValueError(f"{argument_name}必须是字符串")

    tool = tool_config["function"]
    return tool(argument_value)
