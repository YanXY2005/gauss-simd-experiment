# 高斯消去SIMD优化实验代码
## 文件列表
- serial.cc：串行基准版本
- avx2.cc：AVX2非对齐向量化版本
- avx2_only_elim.cc：仅向量化行消去版本
- avx2_aligned.cc：AVX2对齐优化版本
- avx2_unroll.cc：循环展开优化版本

## 编译运行命令
```bash
# 串行版本
g++ -O2 serial.cc -o serial.exe
./serial.exe 2048

# AVX2非对齐版本
g++ -O2 -mavx2 -mfma avx2.cc -o avx2.exe
./avx2.exe 2048

# AVX2对齐版本
g++ -O2 -mavx2 -mfma avx2_aligned.cc -o avx2_aligned.exe
./avx2_aligned.exe 2048
