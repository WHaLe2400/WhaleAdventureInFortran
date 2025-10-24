"""
用 numpy 从 MNIST .gz(idx) 生成 .npy/.bin/.meta
用法：
  python3 DATA_Read.py /path/to/mnist_dir
输出（在同一目录）:
  train-images.npy  train-images.bin  train-images.meta
  train-labels.npy  train-labels.bin  train-labels.meta
"""
import sys, gzip, struct, os
import numpy as np

FILES = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'
]

DTYPE_MAP = {
    0x08: np.uint8,
    0x09: np.int8,
    0x0B: '>i2',
    0x0C: '>i4',
    0x0D: '>f4',
    0x0E: '>f8'
}

def read_idx_gz(path):
    with gzip.open(path, 'rb') as f:
        magic = f.read(4)
        if len(magic) < 4:
            raise ValueError("Invalid IDX header")
        _, _, dtype_code, ndim = struct.unpack('>BBBB', magic)
        dims = [struct.unpack('>I', f.read(4))[0] for _ in range(ndim)]
        data = f.read()
    return dtype_code, dims, data

def process(path):
    name = os.path.basename(path)
    dtype_code, dims, data = read_idx_gz(path)
    if dtype_code not in DTYPE_MAP:
        raise ValueError(f"Unsupported IDX dtype {dtype_code}")
    np_dtype = np.dtype(DTYPE_MAP[dtype_code])
    arr = np.frombuffer(data, dtype=np_dtype)
    arr = arr.reshape(tuple(dims))
    base = name.replace('.gz','').replace('-idx','').replace('ubyte','').replace('.','_')
    out_npy = os.path.join(os.path.dirname(path), base + '.npy')
    out_bin = os.path.join(os.path.dirname(path), base + '.bin')
    out_meta = os.path.join(os.path.dirname(path), base + '.meta')
    np.save(out_npy, arr)
    # .tofile 写原始二进制流（Fortran 用 stream/read 读取字节或相应类型）
    arr.tofile(out_bin)
    with open(out_meta, 'w') as m:
        m.write(f"dtype_code={dtype_code}\n")
        m.write(f"numpy_dtype={np_dtype.str}\n")
        m.write("dims=" + " ".join(str(d) for d in dims) + "\n")
    print("Wrote:", out_npy, out_bin, out_meta)

def main():
    # 主入口：检查命令行参数，期望一个目录路径
    if len(sys.argv) < 2:
        print("Usage: python3 DATA_Read.py /path/to/mnist_dir")
        sys.exit(1)

    # 要处理的目录（第一个参数）
    d = sys.argv[1]

    # 确认给定路径是一个目录，否则退出
    if not os.path.isdir(d):
        print("Not a directory:", d); sys.exit(1)

    # 遍历预定义的 FILES 列表，拼接到输入目录下形成完整路径
    for fn in FILES:
        p = os.path.join(d, fn)

        # 如果文件存在，调用 process 进行解析并保存；异常时打印错误但继续处理下一个文件
        if os.path.exists(p):
            try:
                process(p)
            except Exception as e:
                print("Failed:", p, e)
        else:
            # 文件不存在则跳过并提示
            print("Skip (not found):", p)

if __name__ == '__main__':
    main()