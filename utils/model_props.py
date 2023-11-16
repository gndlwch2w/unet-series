import torch
import numpy as np
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def info(model, input_shape=(1, 3, 224, 224), device="cuda:0"):
    model = model.to(device)
    dummy_input = torch.randn(*input_shape, dtype=torch.float32).to(device)
    flops = FlopCountAnalysis(model, (dummy_input,)).total()
    out = model(dummy_input)
    print(f' * Output shap', out.shape)
    print(f' * Flops: {flops}')
    print(parameter_count_table(model))

def inference_time(model, input_shape=(1, 3, 224, 224), repetitions=300, device='cuda:0'):
    model.to(device)
    dummy_input = torch.randn(*input_shape, dtype=torch.float32).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(
        mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))

def throughout(model, input_shape=(1, 3, 224, 224), repetitions=100, device='cuda:0'):
    model.to(device)
    dummy_input = torch.randn(*input_shape, dtype=torch.float32).to(device)
    total_time = 0
    with torch.no_grad():
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    throughput = (repetitions * input_shape[0]) / total_time
    print(f' * Throughput: {throughput:.3f}')
