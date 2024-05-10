from collections import deque

flag = 0
# 初始化 trigger，长度为100，初始值都为0
trigger = deque([0] * 5, maxlen=5)

# 检查温度是否超过27度，如果超过则触发报警
temperature = 28

if temperature > 27:
    trigger.append(1)
else:
    trigger.append(0)

# 如果 trigger 中1的数量超过80，则触发报警
if sum(trigger) > 3:
    flag = 1
