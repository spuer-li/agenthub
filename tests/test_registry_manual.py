
from app.tools import execute_tool, TOOL_REGISTRY

print("已注册工具：", list(TOOL_REGISTRY.keys()))

result = execute_tool(
    "calculator",
    {"expression": "125 * 36"},
)
print("计算结果：", result)

try:
    execute_tool("unknown_tool", {})
except ValueError as e:
    print("未知工具校验成功：", e)
