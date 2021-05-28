# Word similarity and relationships

## Goal

In this project, you will leverage an important advance in natural language processing called [word2vec](http://arxiv.org/pdf/1301.3781.pdf) (or just *word vectors* / *word embeddings*) to study the similarity between words. As part of the project, you will learn to deal with large data files. We're going to use a "database" from [Stanford's GloVe project](https://nlp.stanford.edu/projects/glove/).  For example, given a single word, we can find the *n* closest words, at least according to the corpus from which the word vectors were derived:

```
Enter a word or 'x:y as z:'
> dog
dog is similar to {dogs puppy pet cat pup}
> cow
cow is similar to {pig sheep goat cattle bull}
> united
united is similar to {kingdom america country britain us}
> chinese
chinese is similar to {korean china vietnamese japanese thai}
> alien
alien is similar to {extraterrestrial spaceship evil planet creature}
> approach
approach is similar to {methodology understanding strategy rather perspective}
```

Given three words, we can also use word vectors to fill in the blank of partial analogies of the form "*x is to y as z is to _____*":

```
Enter a word or 'x:y as z:'
> king:queen as man:
king is to queen as man is to {woman girl lady wonder guy}
> apple:tree as seed:
apple is to tree as seed is to {tree leaf planting plant seedling}
> hammer:nail as comb: 
hammer is to nail as comb is to {nail manicure hair cuticle brush}
> dog:puppy as cat:
dog is to puppy as cat is to {kitten puppy pup kitty pug}
> like:love as dislike:
like is to love as dislike is to {love adore hate liking loathe}
```

Your main goal is to implement a simple interactive program that repeatedly accepts either a word or a partial analogy in the form "`x:y as z:`" for 3 words x, y, and z. I provide the main program for you that actually queries words or analogies from the user and then calls a set of functions you must implement.

## Description

Imagine trying to compare two documents for similarity. One document might be about "Installing Windows software" and another one might be about "Deinstalling Microsoft programs."  Because there are no words in common, at least for these titles, it's hard for a computer to tell these titles are related. A human, on the other hand, can easily equate Windows with Microsoft and software with programs etc., thus, finding the titles similar.

Until 2013, software could really only compare two words for exact match or a so-called *edit distance* (how many character edits to go from one word to the other). With word vector, we have a model for the "meaning" of a word in the form of a big vector of floats (usually 50 to 300 dimensional). These vectors are derived from a neural network that learns to map a word to an output vector such that neighboring words in some large corpus are close in 300-space. ("*The main intuition underlying the model is the simple observation that ratios of word-word co-occurrence probabilities have the potential for encoding some form of meaning.*" see [GloVe project](https://nlp.stanford.edu/projects/glove/)) For example, given the words ['petal','love','king', 'cat'], here is a two-dimensional projection of the vectors for those words and the 3 nearest to those words (there is some overlap):

<img src="figures/wordvec1.png" width=400>

The amazing thing about these vectors is that somehow they really encode the relationship between words. From the original paper, which you can also verify with this project code, the vector arithmetic `king - man + woman` is extremely close to the vector for `queen`!  The GloVe project at Stanford has a nice example showing the vector difference between various words:

<img src="https://nlp.stanford.edu/projects/glove/images/man_woman.jpg" width=400>

A good place to start this project is look at the provided main program as it identifies the key sequence of operations.

### Main program

To provide a little commandline interpreter where users can type in words or partial analogies, you will use this main program in your `wordsim.py` file (that you must create in the root directory of your depository):
 
```python
if __name__ == '__main__':
    glove_dirname = sys.argv[1]
    gloves = load_glove(glove_dirname)

    plot_words(gloves,['petal','love','king', 'cat'], 3)

    print("Enter a word or 'x:y as z:' (type 'exit' to quit)")
    cmd = ''
    while cmd!=None:
        cmd = input("> ")
        if cmd.strip()=='exit':
            break
        match = re.search(r'(\w+):(\w+) as (\w+):', cmd)
        if match is not None and len(match.groups())==3:
            x = match.group(1).lower()
            y = match.group(2).lower()
            z = match.group(3).lower()
            if x not in gloves:
                print(f"{x} is not a word that I know")
                continue
            if y not in gloves:
                print(f"{y} is not a word that I know")
                continue
            if z not in gloves:
                print(f"{z} is not a word that I know")
                continue
            words = analogies(gloves, x, y, z, 5)
            print("%s is to %s as %s is to {%s}" % (x,y,z,' '.join(words)))
        elif re.match(r'\w+', cmd) is not None:
            if cmd not in gloves:
                print(f"{cmd} is not a word that I know")
                continue
            words = closest_words(gloves, cmd.lower(), 5)
            print("%s is similar to {%s}" % (cmd,' '.join(words)))
        else:
            print("Enter a word or 'x:y as z:'")
```

where you just have to fill in the arguments to `analogies(...)` and `closest_words(...)` and write those functions, as described below.

To learn more about passing arguments from the command line to your Python program, see the bottom of our [bash intro](https://github.com/parrt/msds501/blob/master/notes/bash-intro.md).

Users can quit the program by typing "exit" instead of a word or word analogy (you can also kill the running program by using control-C or control-D (on unix), which means "end of file"). That makes the loop terminate and therefore the program.

### Getting word vector data

Your first task is preprocess the word vector "database" and store it in a more suitable and speedy format via NumPy. Download the (HUGE) [Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB download): glove.42B.300d.zip](http://nlp.stanford.edu/data/glove.42B.300d.zip) file from the [GloVe project](https://nlp.stanford.edu/projects/glove) and unzip it into a data directory. (I recommend directory `data` sitting at the root of your user account; e.g., mine is `/Users/parrt/data`.) The unzipped filename is `glove.42B.300d.txt`:

```bash
$ mkdir ~/data # a good place to store data
$ cd ~/data
$ unzip glove.42B.300d.zip
$ ls -l
...
-rw-rw-r--@    1 parrt  staff  5025028820 Oct 24  2015 glove.42B.300d.txt
...
```

(`~` from the command line means your user home directory, such as `/Users/parrt`.)

You will pass that directory name containing your glove data to the main `wordsim.py` program so that it knows where the data is (which will be different on your machine than mine, so we use a command line argument).

The space-separated format of the glove word vector file is extremely simple: it's just the token (usually a word) followed by the components of the word vector. For example:

```
, 0.18378 -0.12123 -0.11987 0.015227 ...
the -0.20838 -0.14932 -0.017528 -0.028432 ...
. 0.10876 0.0022438 0.22213 -0.12102 ...
and -0.09611 -0.25788 -0.3586 -0.32887 ...
...
```

One of the problems we have in data science is that files can be huge, as is the case here. 5,025,028,820 characters is 5 gigabytes (5G), which could expand to much more after loading it into memory.  This will start to get close to the amount of RAM you have in your laptop but you should be okay. Even with a fast machine with an SSD instead of a spinning disk, it takes a few minutes to load all of that text and convert it into floating-point numbers. That will be painfully slow as you try to develop your code because you must reload that data every time you start up `wordsim.py`.

#### Saving the word vectors in NumPy binary

To make development faster and easier, let's convert that text file into a binary format that is not only smaller but much faster to load.  The idea will be to write a small script to load in the text once and save it in binary into a different file.  Create a script called `save_np.py` that does this preprocessing step. Subsequent runs of your main program can load the faster version of the data file rather than the 5G text file. The goal of the script is to create two new files from the original text version:

* `glove.42B.300d.vocab.txt` A list of words from the original word vector file; one word per line
* `glove.42B.300d.npy` A matrix containing all of the word vectors as saved by NumPy's `save()` method. Each row of the matrix represents the word vector for a word.

Script `save_np.py` reads the original glove text file line by line using `f.readlines()`. If you try to load the entire thing with `f.read()` and do a `split('\n')` or similar, you will run out of memory or run into speed problems for sure. So, process the lines one by one, adding the associated word to a vocabulary list of strings and the word vector to a list of numpy arrays.  Given a line, `split(' ')` will give us a list containing the vocabulary word as the first element and the word vector as the remaining 300.  If those 300 strings, one per floating-point number, is in variable `v` then `np.array(v, dtype=np.float32)` will give a fast conversion to a numpy array.  From the list of arrays, we can make a matrix with `np.array(mylistofvectors)`. Save the list of vocabulary words, one per line, into the `glove.42B.300d.vocab.txt` file and use `np.save()` to save the matrix into `glove.42B.300d.npy`.  Store the generated files in the same data directory passed into your `save_np.py` script from the command line.

On my machine, it takes about 3 minutes 30 seconds to load the original text the data file and save the two new files in the current working directory. From the command line, you can time how long things take easily:

```bash
$ time python save_np.py ~/data
Loaded matrix of shape (1917494, 300)
Saved 1917494 words into vocabulary file
Saved matrix into .npy file

real	3m29.343s
user	1m51.068s
sys	0m28.134s
```

Note that your script must take a command line argument indicating the directory containing the word vector file so that I can test your code on my machine. (I'll have the data in a different location than you will.) Your script reads from that directory and writes files to the same data directory. 

The resulting binary numpy file is half the size:

```bash
$ ls -l ~/data/glove.42B.300d.*
-rw-r--r--  1 parrt  staff  2300992928 Apr  8 16:35 /Users/parrt/data/glove.42B.300d.npy
-rw-rw-r--@ 1 parrt  staff  5025028820 Oct 24  2015 /Users/parrt/data/glove.42B.300d.txt
-rw-r--r--  1 parrt  staff    17597168 Apr  8 16:35 /Users/parrt/data/glove.42B.300d.vocab.txt
```

The real benefit is that loading the matrix from the binary file is much faster than loading from a text file. For example, this code:

```python
import numpy as np
filename = "/Users/parrt/data/glove.42B.300d.npy"
vecs = np.load(filename)
print(f"Loaded matrix with shape {vecs.shape}")
```

executes in 1.5 seconds instead of 3 1/2 minutes.

Your `wordsim.py` script will load this optimized set of files and you only need to run `save_np.py` once.

#### Debugging word vector loading

For debugging purposes, as you try to load and save the glove files, you might want to grab the first 50 lines or so from the original text file and store that into a small file. This will take milliseconds to load and you can step through with the debugger to figure out why it is not loading properly or whatever.  The following command on the command line creates such a file for you.

```bash
head -50 glove.42B.300d.txt > glove50.txt
```

The first 50 tokens are:

```
, the . and to of a in " is for : i ) that ( you it on - with 's
this by are at as be from have was or your not ... we ! but ? all
will an my can they n't do he more if
```

When debugging, you can have your program load this file instead of the real `glove.42B.300d.txt` file.

#### Restricting the vocabulary to improve interaction speed

We have another issue with the size of our data set.  While we can load the 1,917,494 words and their vectors quickly, that does not mean we can search through them linearly (one by one) quickly to compute distances and so on.  When I run my solution for this project on the full 1.9M words, it takes about 30 seconds to find similar words and do word analogies.  That does not provide a very good user experience and, while we could fix this using parallel processing, we'll simply restrict the size of the vocabulary to the 235,886 words in the dictionary available on your Mac or Unix machine:

```bash
$ head /usr/share/dict/words 
A
a
aa
aal
aalii
aam
Aani
aardvark
aardwolf
Aaron
$ wc -l /usr/share/dict/words 
  235886 /usr/share/dict/words
```

Being able to choose a reasonable subset of your data for development or other purposes is a useful skill, so let's incorporate that into our final method contained within `wordsim.py` that loads the words and word vectors:

```python
def load_glove(dir:str) -> dict:
    """
    Given The name of the directory holding files glove.42B.300d.npy and
    glove.42B.300d.vocab.txt, this function returns a dictionary mapping
    a string word to numpy array with the associated 300 dimensional
    word vector. The words are restricted to those found within
    file /usr/share/dict/words (make sure to convert those words to
    lowercase for normalization purposes).
    """
    ...
```

### Computing similar words

Ok, now that you have the preprocessed data files ready and stored in your data directory, you can work on the actual word similarity and analogy parts. 

Given a word, *w*, the easy way to find the *n* nearest words is to exhaustively compute the distance from *w*'s vector to every other vector in the database. Sort by the distance and take the first *n* words. Here is the signature of the function you must implement in `wordsim.py` and a comment describing its implementation:

```python
def closest_words(gloves, word, n):
	"""
	Given a gloves dictionary of word:vector and a word return the n nearest words
	as a list of strings. The word is not considered his own nearest neighbor,
	so do not include that in the returned list.
	
	Compute the Euclidean distance between the vector for word and
	every other word's vector. Track the distances with a list of tuples
	of the form: (distance, word).  Sort the list by distance. Return a list
	of the first n words from the sorted list. Do not return the tuples, just the words. Return a python list of strings not numpy array.
	"""
	...
```

Given input `lizard`, your program should respond with:

```
lizard is similar to {snake iguana crocodile frog turtle}
```

Given input `russia`, your program should respond with:

```
russia is similar to {moscow russian soviet iran finland}
```

### Computing missing analogy words

Your final goal is to complete partial analogies given to you by the user. In other words, given input "`shoe:foot as glove:`", your program should respond with:

```bash
shoe is to foot as glove is to {foot hand finger thumb leg}
```

(As you can see it's not perfect, but it does get `hand` as the second closest.)

The key is to look at the relationship between words, which means vector difference. Take a look at the following 2D projection of some word vectors and the vector differences, such as `['madam','mister','niece', 'nephew', 'king', 'queen']`.

<img src="figures/male-female-vectors.png" width="250">

In 2D, the vector differences are fairly similar, which means that their meaning is somehow similar. In this case, the vector difference is reflecting gender in some way.

Here's how to use vector differences for word analogies `x:y as z:`. Compute the vector difference between the first two words and then look for similar vector differences. The simplest mechanism is to exhaustively compare the x-y vector difference to the vector difference from z to all other words in the table. If we sort by distance, the first *n* words will be the most appropriate words to finish the analogy. Here is your code template for the function.

```python
def analogies(gloves, x, y, z, n):
    """
    Given a gloves dictionary of word:vector and 3 words from
    "x is to y as z is to _____", return the n best words that fill in
    the blank to complete the analogy.

    Compute the vector difference between x and y then compute the
    vector difference between z and all vectors, v, in gloves database
    (ignore v=z).  You care about the distance between the xy vector
    and the zv vector for all vectors v. Track the distances with a
    list of tuples of the form: (distance, word).  Sort the list by
    distance. Return a list of the first n words from the sorted
    list. Do not return the tuples, just the words.
    """
    ...
```

## Using PCA to display word vectors

If you have some extra time, you can do a nice visualization of word vectors. The idea is to take the very large 300-dimensional vectors and project them onto just 2-dimensional space so that we can plot them. The key to such a compression is to perform *principal components analysis* (PCA) on a set of word vectors, which you might hear about in the linear algebra boot camp. This is how I drew the graph above for the words ['petal','love','king', 'cat'] (and 3 nearest neighbors). Here is some skeleton code for you to get started:

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_words(gloves, words, n):
    """
    Get a list of vectors for n similar words for each w in words.
    Flatten that into a single list of vectors, including the original
    words' vectors.  Compute a word vector for each of those words and
    put into a list. Use PCA to project the vectors onto two dimensions.
    Extra separate X and Y coordinate lists and pass to matplotlib's scatter
    function. Then, iterate through the expanded word list and plot the
    string using text() with, say, fontsize=9. call plt.show().
    """
    ...
```

For `plot_words(gloves,['petal','glove','computer'], 4)` you should get:

<img src="figures/wordvec2.png" width=300>

You might need to Google around a little bit but there are plenty of examples on the web.
 
## Deliverables

In your repository, you should submit the following files in the root directory.

* `save_np.py` Reads the word vector file and saves the 2D numpy matrix in file `glove.42B.300d.npy` in the current working directory. It also saves the vocabulary list in `glove.42B.300d.vocab.txt` in the current working directory.
* `wordsim.py` This embodies all of the nearest neighbor and word analogy functionality

*Do not add the word vector glove data to the repository!*

My test script will run `save_np.py` first to get the faster data files and then will run the test rig in `test_wordsim.py` (which I'll copy into your directory during grading).

We will only be testing the nearest neighbor and word analogy functionality, not the visualization and not the `save_np.py` script. I will read but not execute `save_np.py` because it takes so long. I will test your `wordsim.py` code using my version of the saved data set.

You can use numpy (e.g., `np.linalg.norm()`) but please do not refer to a bunch of random packages that I probably don't have installed on my test box. Your test will fail.

*Please do not leave a bunch of debugging print statements in your code.* The output of your program is part of your result so make sure you only emit what you are supposed to.

## Evaluation

We will run [test_wordsim.py](https://github.com/parrt/msds501/blob/master/projects/test_wordsim.py) from the command line as follows using the  vectors (where I have placed my datafiles in directory `~/data`):

```bash
$ python -m pytest -v test_wordsim.py ~/data
========================== test session starts ===========================
platform darwin -- Python 3.8.6, pytest-6.2.3, py-1.10.0, pluggy-0.13.1 -- /Users/parrt/opt/anaconda3/bin/python
cachedir: .pytest_cache
rootdir: /Users/parrt
plugins: anyio-2.0.2, xdist-2.2.1, repeat-0.9.1, forked-1.3.0
collected 11 items                                                       

test_wordsim.py::test_similar_words[0] PASSED                      [  9%]
test_wordsim.py::test_similar_words[1] PASSED                      [ 18%]
test_wordsim.py::test_similar_words[2] PASSED                      [ 27%]
test_wordsim.py::test_similar_words[3] PASSED                      [ 36%]
test_wordsim.py::test_similar_words[4] PASSED                      [ 45%]
test_wordsim.py::test_similar_words[5] PASSED                      [ 54%]
test_wordsim.py::test_analogies[0] PASSED                          [ 63%]
test_wordsim.py::test_analogies[1] PASSED                          [ 72%]
test_wordsim.py::test_analogies[2] PASSED                          [ 81%]
test_wordsim.py::test_analogies[3] PASSED                          [ 90%]
test_wordsim.py::test_analogies[4] PASSED                          [100%]

========================== 11 passed in 11.59s ===========================
```

**That test rig must run in under 30 seconds on my machine for you to get credit for the project.**