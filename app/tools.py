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

def add_numbers(a: str, b: str) -> str:
    return str(a+b)

def repeat_text(text: str, times: int) -> str:
    if times < 0:
        raise ValueError("重复次数不能小于零")

    if times > 1000:
        raise ValueError("重复次数不能超过1000")

    return text * times

def format_text(text: str, uppercase: bool) -> str:
    if uppercase:
        return text.upper()
    return text

# 工具注册表：工具名称 -> Python 函数
TOOL_REGISTRY = {
    "calculator": {
        "function": calculator,
        "description": "计算数学四则运算。当用户需要计算时使用。",
        "parameters": {
            "expression": {
                "type": "string",
                "description": "数学表达式，例如 125 * 36",
                "required": True
            }
        },
    },
    "text_stats": {
        "function": text_stats,
        "description": (
            "统计一段文本的字符总数、非空白字符数和单词数。"
            "当用户要求是统计文本长度或单词数量时使用。"
        ),
        "parameters": {
            "text": {
                "type": "string",
                "description": "需要统计的文本内容",
                "required": True
            }
        },
    },
    "add_numbers": {
        "function": add_numbers,
        "description": "计算两个数字的和。当用户需要将两个数字相加时使用。",
        "parameters": {
            "a": {
                "type": "number",
                "description": "第一个数字，可以是整数或小数",
                "required": True
            },
            "b": {
                "type": "number",
                "description": "第二个数字，可以是整数或小数",
                "required": True
            }
        }
    },
    "repeat_text": {
        "function": repeat_text,
        "description": "将指定文本重复指定次数",
        "parameters": {
            "text": {
                "type": "string",
                "description": "需要重复的文本",
                "required": True
            },
            "times": {
                "type": "integer",
                "description": "重复次数，必须是整数",
                "required": True
            }
        }
    },
    "format_text": {
        "function": format_text,
        "description": "根据选项决定是否将文本转换为大写。",
        "parameters": {
            "text": {
                "type": "string",
                "description": "需要处理的文本",
                "required": True
            },
            "uppercase": {
                "type": "boolean",
                "description": "是否转换为大写",
                "required": True
            }
        }
    }
}


def execute_tool(tool_name: str, arguments: dict) -> str:
    """根据工具名称找到并执行对应的 Python 函数。"""
    if not isinstance(arguments, dict):
        raise ValueError("工具参数必须是字典")

    tool_config = TOOL_REGISTRY.get(tool_name)

    if tool_config is None:
        raise ValueError(f"不支持的工具：{tool_name}")

    parameter_configs = tool_config["parameters"]
    unexpected = set(arguments) - set(parameter_configs)
    if unexpected:
        raise ValueError(f"存在未定义的参数：{', '.join(unexpected)}")

    type_mapping = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool
    }

    for name, config in parameter_configs.items():
        if name not in arguments:
            if config.get("required", True):
                raise ValueError(f"缺少必填参数：{name}")
            continue

        value = arguments[name]
        argument_type = config["type"]
    
        if argument_type == "string":
            valid = isinstance(value, str)
        elif argument_type == "integer":
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif argument_type == "number":
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif argument_type == "boolean":
            valid = isinstance(value, bool)
        else:
            raise ValueError(f"不支持的参数类型：{argument_type}")

        if not valid:
            raise ValueError(f"{value}必须是{argument_type}")    

    tool = tool_config["function"]
    return tool(**arguments)
