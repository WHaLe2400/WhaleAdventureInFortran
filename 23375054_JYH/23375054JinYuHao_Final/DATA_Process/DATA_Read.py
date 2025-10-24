"""
用 numpy 从 MNIST .gz(idx) 生成 .npy/.bin/.meta/.txt
用法：
    python3 DATA_Read.py /path/to/mnist_dir [--out /path/to/output_dir]
若不指定 --out，则输出到每个输入文件所在目录。
输出（在输出目录中）:
    train-images.npy  train-images.bin  train-images.meta  train-images.txt
    train-labels.npy  train-labels.bin  train-labels.meta  train-labels.txt
"""
import sys, gzip, struct, os
import numpy as np
import argparse

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

def process(path, out_dir=None):
    name = os.path.basename(path)
    dtype_code, dims, data = read_idx_gz(path)
    if dtype_code not in DTYPE_MAP:
        raise ValueError(f"Unsupported IDX dtype {dtype_code}")
    np_dtype = np.dtype(DTYPE_MAP[dtype_code])
    arr = np.frombuffer(data, dtype=np_dtype)
    arr = arr.reshape(tuple(dims))

    if out_dir is None:
        out_dir = os.path.dirname(path)
    os.makedirs(out_dir, exist_ok=True)

    base = name.replace('.gz','').replace('-idx','').replace('ubyte','').replace('.','_')
    out_npy = os.path.join(out_dir, base + '.npy')
    out_bin = os.path.join(out_dir, base + '.bin')
    out_meta = os.path.join(out_dir, base + '.meta')
    out_txt = os.path.join(out_dir, base + '.txt')

    np.save(out_npy, arr)
    arr.tofile(out_bin)
    with open(out_meta, 'w') as m:
        m.write(f"{dtype_code}\n")
        m.write(f"{np_dtype.str}\n")
        m.write(" ".join(str(d) for d in dims) + "\n")

    if arr.ndim == 1:
        fmt = '%d' if np.issubdtype(arr.dtype, np.integer) else '%.18e'
        np.savetxt(out_txt, arr, fmt=fmt)
    elif arr.ndim == 2:
        fmt = '%d' if np.issubdtype(arr.dtype, np.integer) else '%.18e'
        np.savetxt(out_txt, arr, fmt=fmt)
    else:
        samples = arr.shape[0]
        rest = int(np.prod(arr.shape[1:]))
        resh = arr.reshape(samples, rest)
        fmt = '%d' if np.issubdtype(arr.dtype, np.integer) else '%.18e'
        np.savetxt(out_txt, resh, fmt=fmt)

    print("Wrote:", out_npy, out_bin, out_meta, out_txt)

def main():
    parser = argparse.ArgumentParser(description="Convert MNIST IDX.gz to .npy/.bin/.meta/.txt")
    parser.add_argument('indir', nargs='?', default="23375054_JYH/23375054JinYuHao_Final/1_DATA",
                        help='输入目录，包含 IDX .gz 文件（默认示例目录）')
    parser.add_argument('-o', '--out', default="23375054_JYH/23375054JinYuHao_Final/1_DATA_Reread",
                        help='输出目录（可选），若指定则所有生成文件写入该目录；否则写入输入文件所在目录')
    args = parser.parse_args()

    d = args.indir
    out_dir = args.out

    if not os.path.isdir(d):
        print("Not a directory:", d)
        sys.exit(1)

    for fn in FILES:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            try:
                process(p, out_dir=out_dir)
            except Exception as e:
                print("Failed:", p, e)
        else:
            print("Skip (not found):", p)

if __name__ == '__main__':
    main()