import numpy as np
import time
import datetime
import torch


# a = np.random.randint(0, 100, 100)
# a = 14.012354989
# b = "a={:.4f}".format(a)
# print(b)


# a = time.time()
# time_struct = time.localtime(a)
# print(time.localtime(a))
# time_str = "{}年{}月{}日{}:{}:{}".format(time_struct[0], time_struct.tm_mon, time_struct.tm_mday, time_struct.tm_hour,
#                                    time_struct.tm_min, time_struct.tm_sec)
# print(time_str)

t1 = torch.rand(1, 10)
print(t1)
t2 = torch.max(t1, 1)
print(t2)


