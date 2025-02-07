import torch
import matplotlib.pyplot as plt

def makemore():
    words = open('datasets/names.txt', 'r').read().splitlines()
    N = torch.zeros((28, 28), dtype=torch.int32)
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i for i,s in enumerate(chars)}
    stoi['<S>'] = 26
    stoi['<E>'] = 27

    for w in words:
        chs = ['<S>'] + list(w) + ['<E>']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1
    
    plt.imshow(N)

makemore()