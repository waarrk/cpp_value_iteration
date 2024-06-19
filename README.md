# cpp_value_iteration

## CPU

```bash
g++ -std=c++11 main.cpp common.cpp obstacle.hpp -o main.exe
./main.exe
```

## GPU

```bash
nvcc -std=c++11 main_gpu.cu common.cpp obstacle.hpp -o main_gpu.exe
./main_gpu.exe
```

## Graph

```bash
Python3 decode.py
```
