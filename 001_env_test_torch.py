#输入库
import torch
import torch.backends.cudnn as cudnn
#查看版本
print('torch.__version__ =',torch.__version__)
#查看qpu是否可用
print('torch.cuda.is_available() =', torch.cuda.is_available())
#返回设备qpu个数
print('torch.cuda.device_count() =', torch.cuda.device_count())
#查看对应CUDA的版本号
print('torch.backends.cudnn.version() =',torch.backends.cudnn.version())
print('torch.version.cuda =',torch.version.cuda)


# print(' =',)