# makemore - A character level Language Model :part_alternation_mark:

Makemore is an auto-regressive character-level language model. It can take a text file as an input, where each line is assumed to be one training example and it generates more examples like it. New and Unknown...

This repo covers the work of Andrej's [makemore](https://github.com/karpathy/makemore/) repo with a more detailed documentation of each implementations. :simle:

## Implementations covered:

1. [Bigram model](#bigram-model)


## Bigram model

A **Bigram** model performs next character prediction on the basis of the current character.

To implement a Bigram model, we will need a training dataset. We build our training dataset with the help of a collection of [names](/datasets/names.txt) and further plucking out the bigrams from it. Bigram here refers to the set of character pairs present in the **names** dataset.

***Goal The end goal is to generate / make more names similar to those present in the **names** dataset, but potentially never seen before in the dataset.***

Out current names dataset with approx `32K` examples looks somewhat like this.

```
emma
olivia
ava
isabella
sophia
charlotte
...
```

Now it's time to pluck out the bigrams from the above dataset. We have a special character `.` which will act as a starting and the ending character for names dataset.

Eg. `emma` will be interpreted as `.emma.`

And hence we can have following set of bigrams from this example:
`{ '.e', 'em', 'mm', 'ma', 'a.'}`

Since all the names are in lower case characters (no special characters), we can safely say that there can be `26 + 1` possible characters (including `.`) and `(26+1) * (26+1) = 729` possible bigrams.

We will maintain a map that will count the frequency of each bigrams in the training dataset. It can be visualized in this way.

![Bigram Map](/media/bigram_map.png)

We can see the above map stores the frequency of each possible bigrams. Now instead of storing frequency, let's store the probability of occurence of each bigram.

![probability distribution](/media/prob_dist.png)

We have used [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) to achieve this in our existing bigram count map.

***Note: We have initially added 1 to all the bigram counts to perform model smoothing i.e. to avoid the presence of 0 values in the map leading to infinite log values.***

```python
# Broadcasting the column level sum
# Add 1 to all numbers i.e. perform Model smoothing to avoid negative log likelihoods
P = (N+1).float()
P = P / P.sum(1, keepdim=True)
```

The updated map with probabilities look somewhat like this.

![Bigram map probabilities](/media/bigram_map_prob.png)

Now, we have the probabilities of the occurnece of the next character for each character. We can start sampling more characters based on the given probability distributions.
We will use Pytorch's [Multinomial](https://pytorch.org/docs/stable/generated/torch.multinomial.html) to sample next character based on the given probability distribution.

```python

# Sampling more words
g = torch.Generator().manual_seed(2147483647)

for i in range(40):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

```

The sampling process described below is repeated until we reach at the end character `.`.

![multinomial sampling](/media/sampling_multinomial.png)

We can now sample more words with the above approach.

```
cexze.
momasurailezitynn.
konimittain.
llayn.
ka.
da.
staiyaubrtthrigotai.
moliellavo.
...
```

## Running the project
### Virtual Environments in Python

Creating a Virtual Environment

```python
> virtualenv saurav_env

# Launching a Virtual Environment
> source saurav_env/bin/activate

# Installing dependencies in the virtual env
> pip install Django==1.9

# Deactivate Python Virtual Environment
> deactivate
```

Read more here: https://www.geeksforgeeks.org/python-virtual-environment/