## Run the Program

cd Lenet300100/deep-codegen

python train.py

make sure torch and torchvision is in the environment

## Coding

kernel.cu: matrix multiplication kernel (tiling, memory coalescing, avoiding bank conflict)\\
pytorch_apis.py: forward and backward of matrix multiplication\\
myLinear.py: Linear (using kaiming init)\\
myLenet.py: Lenet300100 from scratch\\
Code generation from https://github.com/the-data-lab/deep-codegen.git
