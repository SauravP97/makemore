import torch

def makemore():
    words = open('datasets/names.txt', 'r').read().splitlines()
    print(words[:10])
    print(len(words))
    b = {}

    for w in words:
        chs = ['<S>'] + list(w) + ['<E>']
        for ch1, ch2 in zip(chs, chs[1:]):
            bigram = (ch1, ch2)
            b[bigram] = b.get(bigram, 0) + 1
    
    print(sorted(b.items(), key = lambda kv: -kv[1]))

makemore()