import tensorflow as tf

def MakeHighRange():
    HighRange = [11.,1.,10.,14.,13.5,4.5,4.,4.,12.5,-3.5,8.,12.]
    # print(len(HighRange))
    HighRange = tf.convert_to_tensor(HighRange)
    return HighRange

def MakeLowRange():
    LowRange = [5.5,-4.,1.,-7.,6.,1.,-3.,-5.,6.5,-10.,3.,3.]
    # print(len(LowRange))
    LowRange = tf.convert_to_tensor(LowRange)
    return LowRange
