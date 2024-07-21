import queue
import time

# 创建一个队列
queue_image = queue.Queue(maxsize=4)

# 示例：向队列中添加一些元素
for i in range(6):
    try:
        queue_image.put(f'image_{i}', block=False)
        print(f"Added image_{i} to the queue")
    except queue.Full:
        print("Queue is full in show_frame")
        continue
    except Exception as e:
        print(f"Error in show_frame: {e}")
        break
print('---------------------')

# 检查队列中有多少内容
num_elements = queue_image.qsize()
print(f"There are {num_elements} elements in the queue.")

# 示例：处理队列中的元素
# while not queue_image.empty():
#     image = queue_image.get()
#     print(f"Processing {image}")
#     time.sleep(1)  # 模拟处理时间

for i in range(4):
    image = queue_image.get()
    print(f"Processing {image}")
    time.sleep(1)  # 模拟处理时间

print('---------------------')
for i in range(6):
    try:
        queue_image.put(f'image_{i}', block=False)
        print(f"Added image_{i} to the queue")
    except queue.Full:
        print("Queue is full in show_frame")
        continue
    except Exception as e:
        print(f"Error in show_frame: {e}")
        continue
print('---------------------')
# 再次检查队列中的内容
num_elements = queue_image.qsize()
print(f"After processing, there are {num_elements} elements left in the queue.")


