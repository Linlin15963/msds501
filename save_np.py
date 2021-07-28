import sys
import os
import numpy as np

script_path = sys.argv[1]
dirname = os.path.dirname(script_path)
dirname = "." if dirname.strip() == "" else dirname

ret_vocab_path = "{dir}/{f}".format(dir = dirname, f = "glove.42B.300d.vocab.txt")
ret_vectr_path = "{dir}/{f}".format(dir = dirname, f = "glove.42B.300d.npy")

vocab = []
mylistofvectors = []
with open(script_path, 'r', encoding='latin1') as f:
    for line in f.readlines():
        arr = line.strip().split(' ')
        if len(arr) != 301:
            print(arr)
        word, v = arr[0], arr[-300:]
        vocab.append(word)
        mylistofvectors.append(np.array(v, dtype=np.float32))

with open(ret_vectr_path, "wb") as f:
    np.save(f, np.array(mylistofvectors))

with open(ret_vocab_path, 'w') as f:
    for word in vocab:
        f.write("{v}\n".format(v=word))