from model.unet_plus_plus import UNetPlusPlus
from utils.model_props import *

def test():
    input_shape = (1, 3, 480, 480)
    model = UNetPlusPlus(1, bilinear=False, fix_weight=False)
    info(model, input_shape)
    inference_time(model, input_shape)
    throughout(model, input_shape)
    # * Output shape: torch.Size([1, 1, 224, 224])
    # * Flops: 91404214272
    #   36.2M
    # * Mean@1 89.931ms Std@5 0.323ms FPS@1 11.12
    # * Throughput: 11.080

if __name__ == '__main__':
    test()
