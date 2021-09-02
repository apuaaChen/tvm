import torch

import tvm
from tvm import relay
import numpy as np
import tvm.testing
from tvm import auto_scheduler
from tvm.contrib import graph_executor


class Model(torch.nn.Module):
    def __init__(self, subscript):
        super(Model, self).__init__()
        self.subscript = subscript
    
    def forward(self, *input):
        return torch.einsum(self.subscript, input)


def test_einsum(subscripts, shapes):

    model = Model(subscripts).eval().cuda()
    ops = []
    shape_list = []
    data = []
    for idx, shape in enumerate(shapes):
        tmp = torch.randn(size=shape, dtype=torch.float32, device="cuda")
        ops.append(tmp,)
        shape_list.append(("x%d"%idx, shape))
        data.append(tmp.cpu().numpy())
    o = model(ops)

    # obtain the JIT scripted model
    scripted_model = torch.jit.trace(model, tuple(ops)).eval().cuda()
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    target = tvm.target.Target("cuda", host="llvm")
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target, include_simple_tasks=True)

    log_file = "./einsum_gpu.json"
    def run_tuning():
        # measure_ctx launches a different process for measurement to provide isolation
        # It protect the master process from GPU crashes
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=64,  # change this to 20000 to achieve the best performance
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            verbose=0
        )

        tuner.tune(tune_option)

    run_tuning()

    dev = tvm.device(str(target), 0)
    output_shape = o.size()

    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib_ansor = relay.build(mod, target=target, params=params)
    module_ansor = graph_executor.GraphModule(lib_ansor["default"](dev))
    for idx, d in enumerate(data):
        module_ansor.set_input("x%d"%idx, d)
    
    module_ansor.run()
    tvm_output = module_ansor.get_output(0, tvm.nd.empty(output_shape)).numpy()
    print("max error: %.4f" % np.max(tvm_output - o.cpu().numpy()))


def verify_einsum():
    test_einsum("ij,jk->ik", [(2, 3), (3, 4)])
    test_einsum("ij,jk,km->im", [(2, 3), (3, 4), (4, 5)])
    test_einsum("ii", [(5, 5)])
    test_einsum("ii->i", [(5, 5)])
    test_einsum("ij->i", [(5, 5)])
    test_einsum("...j->...", [(5, 5)])
    test_einsum("...j, j", [(5, 5), (5,)])
    test_einsum("..., ...", [(), (2, 3)])
    test_einsum("ijk, jil->kl", [(3, 4, 5), (4, 3, 2)])
    test_einsum("ij, ij -> i", [(1, 4), (2, 4)])
    test_einsum("...ij, ...jk -> ...ik", [(1, 4), (4, 2)])
    test_einsum("...ij, ...ik -> ...jk", [(1, 1, 1, 4), (1, 1, 1, 3)])
    
    

if __name__ == "__main__":
    verify_einsum()