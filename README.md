# cpp_value_iteration

## CPU

### Signle Thread

```bash
g++ -std=c++11 main.cpp common.cpp obstacle.cpp -o main.exe
./main.exe
```

### Multi Thread

```bash
g++ -std=c++11 main_mulch.cpp common.cpp obstacle.cpp -o main_mulch.exe
./main_mulch.exe
```

## GPU

```bash
nvcc -std=c++11 main_gpu.cu common.cpp obstacle.cpp -o main_gpu.exe
./main_gpu.exe
```

## Graph

```bash
Python3 decode.py
```
