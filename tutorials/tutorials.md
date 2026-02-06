# Cutile (CUDA Tile DSL) 核心概念总结

## 简介

**Cutile**（CUDA Tile）是一个基于 Python 的 CUDA DSL（领域特定语言），用于编写高性能 GPU 算子。它让你用 Python 编写深度学习算子（如矩阵乘法、Softmax、LayerNorm 等），自动编译为高效的 CUDA 代码。

---

## 1. `ct.kernel` - Kernel 装饰器

### 作用
用于**定义 CUDA Kernel 函数**，将 Python 函数编译为在 GPU 上并行执行的代码。

### 示例
```python
@ct.kernel
def ct_sum(x: ct.Array, y: ct.Array, tile_size: ct.Constant):
    block_id = ct.bid(0)                              # 获取 block 索引
    tile_x = ct.load(x, index=(block_id, ), shape=(tile_size, ))
    tile_x = ct.sum(tile_x.astype(ct.float32))
    ct.atomic_add(y, (0, ), tile_x.astype(y.dtype))
```

### 特点
- 函数内的代码会在每个 CUDA block 上**并行执行**
- 使用 `ct.bid(0)` 获取 block 索引（对应 CUDA 的 `blockIdx.x`）
- 支持类型注解：
  - `ct.Array`：全局内存数组
  - `ct.Tile`：片上缓存（Shared Memory）
  - `ct.Constant`：编译时常量

### `@ct.kernel` vs `@ct.function`

| 装饰器 | CUDA 对应 | 作用 | 调用方式 |
|--------|-----------|------|----------|
| `@ct.kernel` | `__global__` 函数 | 可被 CPU 启动的 GPU 入口函数 | 通过 `ct.launch()` 启动 |
| `@ct.function` | `__device__` 函数 | 仅能被 kernel 调用的设备函数 | 在 kernel 内部调用 |

```python
@ct.function
def tile_norm(x: ct.Tile, tile_size: ct.Constant) -> ct.Tile:
    # device 函数：只能在 kernel 内部调用
    mean = ct.sum(x) / tile_size
    return x - mean

@ct.kernel
def ct_norm(x: ct.Array, y: ct.Array, tile_size: ct.Constant):
    # global 函数：可以被 ct.launch 启动
    tile_x = ct.load(x, (block_id, 0), (1, tile_size))
    tile_x = tile_norm(tile_x, tile_size)  # 调用 device 函数
    ct.store(y, (block_id, 0), tile_x)
```

---

## 1.5 `ct.load` / `ct.store` - 数据加载与存储

### 作用
`ct.load` 用于从 **Global Memory** 高效加载数据到 **SM 的片上内存**（Tile），`ct.store` 用于将计算结果从 Tile 写回 Global Memory。

**关键特性**：
- 自动优化内存访问模式，无需关心数据实际存储在 **L1 Cache** 还是 **Shared Memory**
- 支持 **TMA (Tensor Memory Accelerator)** 硬件加速传输
- 自动处理内存对齐和合并访问

### 语法
```python
tile = ct.load(
    array,           # 源数组 (ct.Array)
    index,           # 起始索引，如 (block_id, 0)
    shape,           # 加载的 tile 形状，如 (1, tile_size)
    allow_tma=True,  # 是否允许使用 TMA 硬件加速
    padding_mode=ct.PaddingMode.ZERO,  # 越界填充模式
    order="C"        # 内存布局 "C"(行优先) 或 "F"(列优先)
)

ct.store(
    array,           # 目标数组 (ct.Array)
    index,           # 起始索引
    tile,            # 要存储的 tile (ct.Tile)
    allow_tma=True   # 是否允许使用 TMA
)
```

### 示例
```python
@ct.kernel
def ct_matmul(
    A: ct.Array, B: ct.Array, o: ct.Array,
    tileM: ct.Constant, tileN: ct.Constant, tileK: ct.Constant
):
    block_x, block_y = ct.bid(0), ct.bid(1)
    num_tile_k = ct.cdiv(k, tileK)
    accumulator = ct.full((tileM, tileN), 0, dtype=ct.float32)
    
    for k_iter in range(num_tile_k):
        # 从 Global Memory 加载 tile 到 SM
        tileA = ct.load(
            A, (block_x, k_iter), (tileM, tileK), 
            padding_mode=ct.PaddingMode.ZERO, order="C"
        )
        tileB = ct.load(
            B, (k_iter, block_y), (tileK, tileN),
            padding_mode=ct.PaddingMode.ZERO, order="F"
        )
        # 在 SM 上执行矩阵乘法
        accumulator = ct.mma(tileA, tileB, accumulator)
    
    # 将结果写回 Global Memory
    ct.store(o, (block_x, block_y), accumulator)
```

### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `array` | `ct.Array` | 源/目标数组（Global Memory） |
| `index` | `tuple` | 在数组中的起始索引位置 |
| `shape` | `tuple` | 要加载/存储的 tile 形状 |
| `allow_tma` | `bool` | 是否使用 TMA 硬件加速（Hopper+架构） |
| `padding_mode` | `ct.PaddingMode` | 索引越界时的填充策略：`ZERO`（填0）、`NAN`（填NaN） |
| `order` | `str` | 内存布局：`"C"` 行优先（默认），`"F"` 列优先 |

---

## 2. `ct.launch` - 启动 Kernel

### 作用
**异步提交 Kernel 执行请求**到指定的 CUDA Stream。`ct.launch` 在 CPU (Host) 端执行，将被 `@ct.kernel` 装饰的 GPU 函数及其执行配置加入到 Stream 的命令队列中，GPU 随后按顺序执行。

### 示例
```python
ct.launch(
    torch.cuda.current_stream(),  # CUDA 流
    (num_blocks, ),               # Grid 维度（多少个 block）
    ct_sum,                       # Kernel 函数
    (x, y, tile_size)             # 传递给 kernel 的参数
)
# 注意：launch 会立即返回，不会等待 GPU 执行完成！
```

### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `stream` | `torch.cuda.Stream` | CUDA 流，控制异步执行 |
| `grid_dim` | `tuple` | Block 数量，如 `(num_blocks,)` 或 `(grid_x, grid_y)` |
| `kernel` | `callable` | 被 `@ct.kernel` 装饰的函数 |
| `args` | `tuple` | 传递给 kernel 的参数列表 |

### 关键特性：异步非阻塞

```python
ct.launch(stream, (num_blocks,), my_kernel, (x, y))
print("这行代码会立刻执行，不会等待 GPU 算完")
# 如需等待 GPU 完成，需显式同步：
# torch.cuda.synchronize()
```

---

## 3. `stream` - CUDA 流

### 作用
**CUDA Stream** 是 GPU 上的异步执行队列，用于控制 kernel 执行的顺序和并发。

### 示例
```python
stream = torch.cuda.current_stream()  # 获取 PyTorch 当前流
ct.launch(stream, (num_blocks, ), kernel, args)
```

### 关键特性

| 特性 | 说明 |
|------|------|
| **异步执行** | Kernel 启动后立即返回，不阻塞 CPU |
| **顺序保证** | 同一流中的操作按提交顺序串行执行 |
| **并发能力** | 不同流可同时执行多个 kernel，提升 GPU 利用率 |

### 特殊注意：默认流（0 号流）的同步行为

**默认流（Legacy Default Stream）** 是一个特殊的隐式流，它有一个重要的同步特性：**与所有其他流强制同步**。

**Legacy 默认流的行为：**

```
Stream 1:  [Kernel A]                          [Kernel C]
                 ↓                                  ↑
Default(0):      [========= Kernel B =========]
                 ↑                                  ↓
Stream 2:       [Kernel D]                        [Kernel E]

结果：所有流被 Kernel B 串行化，无法并发执行
```

当在 Legacy 默认流上执行任何操作（如 kernel 启动或同步）时：
1. 默认流**先等待**所有其他阻塞流中的操作完成
2. 执行默认流中的操作
3. 然后所有其他流**等待**默认流的操作完成

**这意味着：** 即使你有多个独立创建的流，只要中间插入了一个默认流的操作，所有流都会被强制串行化，**丧失并发能力**。

**解决方案：**
- 使用 CUDA 7+ 的 **per-thread default stream** 模式（编译时添加 `--default-stream per-thread`），使默认流变成普通流
- 创建流时使用 `cudaStreamNonBlocking` 标志创建非阻塞流，它们不会与 Legacy 默认流同步
- 避免在多流程序中混用默认流操作

**PyTorch 中的默认流：**
```python
# torch.cuda.current_stream() 返回的是当前线程的默认流
# 在 PyTorch 中通常是 per-thread 模式，但仍需注意与 Legacy 流的混用问题
stream = torch.cuda.current_stream()
```

---

## 三者关系

```
┌─────────────────────────────────────────────────────────────┐
│                     CPU (Host)                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │   @ct.kernel                                        │    │
│  │   def my_kernel(...):  ← __global__ 函数定义        │    │
│  │       # 定义 GPU 并行代码                            │    │
│  └─────────────────────────────────────────────────────┘    │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │   ct.launch(stream, grid, kernel, args)             │    │
│  │   ├─ 异步提交 kernel 到 Stream 队列                  │    │
│  │   └─ 立即返回（非阻塞）                              │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     GPU (Device)                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  CUDA Stream 队列                                    │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐             │    │
│  │  │kernel A │→│kernel B │→│kernel C │ → ...        │    │
│  │  └─────────┘  └─────────┘  └─────────┘             │    │
│  │       ↓              ↓              ↓              │    │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐         │    │
│  │  │Block 0-1│    │Block 0-1│    │Block 0-1│         │    │
│  │  │Block 2-3│    │Block 2-3│    │Block 2-3│         │    │
│  │  │   ...   │    │   ...   │    │   ...   │         │    │
│  │  └─────────┘    └─────────┘    └─────────┘         │    │
│  │      （每个 kernel 并行执行多个 block）               │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 与传统 CUDA C 的对比

| Cutile (Python) | CUDA C | 说明 |
|-----------------|--------|------|
| `@ct.kernel` | `__global__ void func()` | 定义可被 CPU 启动的 GPU 函数 |
| `@ct.function` | `__device__ void func()` | 定义仅能在 GPU 内部调用的设备函数 |
| `ct.launch(stream, grid, kernel, args)` | `func<<<grid, block>>>(args)` | 异步提交 kernel 到 Stream |
| `torch.cuda.current_stream()` | `cudaStream_t` | 异步执行流 |
| `ct.bid(0)` | `blockIdx.x` | Block 索引 |
| `ct.load/ct.store` | 手动内存管理 | Tile 级数据加载/存储 |

---

## 优势

1. **Python 原生**：无需编写 CUDA C，降低开发门槛
2. **Tile 级抽象**：通过 `ct.Tile` 高效利用 Shared Memory
3. **自动优化**：编译器自动生成高性能的线程级代码
4. **与 PyTorch 集成**：无缝使用 `torch.Tensor` 作为输入输出