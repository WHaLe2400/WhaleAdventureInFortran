import wget
import os
def download_mnist(save_dir):
   urls = [
       "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
       "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
       "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
       "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
   ]
   for url in urls:
       filename = os.path.join(save_dir, os.path.basename(url))
       if not os.path.exists(filename):
           wget.download(url, out=filename)
       else:
           print(f"{filename} 已存在")
download_mnist("/root/0_FoRemote/WhaleAdventureInFortran/1_DATA")