# 1.程序主体更新过程
## 04程序更新（240618）
线程异常处理：在主线程中捕获子线程异常，这样可以更方便地调试和管理线程状态。
日志记录：使用 Python 的 logging 模块替代 print 语句，这样可以更好地管理日志级别和输出格式。
线程关闭：确保在 running_flag 被设置为 False 后，所有子线程都能优雅地退出。
代码结构优化：分离函数逻辑，提升代码的可读性和复用性。

1. 想通过判断Queue是否满进行

方案1：queue_image.put(f'image_{i}', block=False)

方案2:if queue_image.full():(终选方案)

## 06程序更新（240620）

添加queue_image.full()作为判断

使用func_initial2

flag_run[程序是否运行，是否需要手部检测]

## 07程序更新
### 240622
本程序主要任务：探索使用Class作为动作函数ABC的实现
### 240624:(initial7_2)
#### 问:1：如何解决需要box的事情
##### 方法：
重新获取keypoints
重新设计cap函数，对比直接放入results和放入提取后的keypoints
##### 结果：
直接放入results的速度无明显变化，但是为了保证手部的绘制，cap过程还是需要导出plot文件

最后想法是直接提取 box 和 keypoints
#### 问题2：keypoints提取的问题
之前是tolist，然后试一下
```
required_keypoints_indices = [0, 1, 2]  # indices for keypoints required for action 1
required_keypoints = [keypoints[i][:2] for i in required_keypoints_indices]
```
是不是并不需要这一步，直接通过keypoints索引读取就行了


# 2.封装函数更新过程
## func_initial2(240619)
更新func_frame_cap函数

拆分process_pose,process_hand

# 3.学习函数更新过程


## 短期任务
更新程序ver6

## 长期任务
封装不同动作


