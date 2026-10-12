
from app.tools import calculator

print("测试 1：", calculator("125 * 36"))
print("测试 2：", calculator("100 / 4"))
print("测试 3：", calculator("15 + 8 - 3"))

try:
    calculator("1 / 0")
except ValueError as e:
    print("测试 4：", e)

try:
    calculator("__import__('os')")
except ValueError as e:
    print("测试 5：", e)
