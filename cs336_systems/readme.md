有用的文件只有:

三个实现
fa2_triton
fa2_torch
fa2_naive

两个测试
benchmark
test_precision


测试结果(三次平均):
```
n_heads = 32
sequence_length = 8576
d_head = 128
cuda:1
tile = 32
```

| triton allow_tf32 | torch medium | torch high           | torch highest(default)            |
|-------------------|--------------|----------------------|--------------------------|
| **true(default)**   | pass 4/8 time 92.5ms | pass 4/8 time 93ms | pass 5/8 time 190ms    |
| **false**         | pass: 7/8 time 149ms | pass 7/8 time 150ms| pass 8/8 time 224ms    |

我选择 triton allow_tf32 true && torch medium


```
n_heads = 32
sequence_length = 8576
d_head = 128
cuda:1
tile = 32
torch high
```

pass: 4/8
time: 93ms


```
tile_size 16
torch high
```

pass: 4/8
time: 105ms

```
tile_size 32
torch highest
```

pass: 5/8
time: 190ms

```
tile_size 32
triton allow_tf32 flase
torch high
```
pass: 7/8
time: 150ms

```
tile_size 32
triton allow_tf32 flase
torch highest
```
pass: 8/8
time: 224ms

```
tile_size 32
triton tf32 true
bwd use P = torch.softmax(S, dim=-1)
```
pass: 8/8
这招太狠了, 不是实验书上的实现, 并且无法扩展到triton, 但精度不错.

