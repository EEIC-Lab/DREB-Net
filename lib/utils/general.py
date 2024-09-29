import math


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def one_flat_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    #return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1
    return lambda x: ((1 - math.cos((x - (steps // 2)) * math.pi / (steps // 2))) / 2) * (y2 - y1) + y1 if (x > (steps // 2)) else y1

