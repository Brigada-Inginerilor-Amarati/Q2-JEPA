import sys

# CUDA-Q loads LLVM/MLIR with RTLD_GLOBAL; prevent torch._dynamo/torchvision from
# dlopen'ing libtriton.so (which links its own LLVM/MLIR and aborts on symbol collision).
sys.modules.setdefault("triton", None)
