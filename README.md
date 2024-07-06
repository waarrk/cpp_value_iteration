# cpp_value_iteration

## CPU

### Signle Thread

```bash
g++ -std=c++11 -o3 main.cpp common.cpp obstacle.cpp -o main.exe
./main.exe 128
```

### Multi Thread

```bash
g++ -std=c++11 -o3 main_mulch.cpp common.cpp obstacle.cpp -o main_mulch.exe
./main_mulch.exe 128 8
```

## GPU

```bash
nvcc -O3 main_gpu.cu common.cpp obstacle.cpp -o main_gpu.exe
./main_gpu.exe 128 8 8 8
```

## Graph

```bash
Python3 decode.py
```
