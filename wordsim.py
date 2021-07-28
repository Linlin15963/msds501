import sys
import re
import numpy as np

# glove.42B.300d.vocab.txt (I guess all words?)
# glove.42B.300d.npy       (I guess this is a list of list, like a matrix?)
def load_glove(dir:str) -> dict:
    """
    Given The name of the directory holding files glove.42B.300d.npy and
    glove.42B.300d.vocab.txt, this function returns a dictionary mapping
    a string word to numpy array with the associated 300 dimensional
    word vector. The words are restricted to those found within
    file /usr/share/dict/words (make sure to convert those words to
    lowercase for normalization purposes).
    """
    words_set = {}
    with open("/usr/share/dict/words", "r") as f:
        for line in f.readlines():
            word = line.strip().lower()
            if words_set.get(word) is None:
                words_set[word] = 1

    f_npy = "{dir}/glove.42B.300d.npy".format(dir=dir)
    mat = np.load(f_npy)

    f_vocab = "{dir}/glove.42B.300d.vocab.txt".format(dir=dir)
    wdict = {}
    with open(f_vocab, "r") as f:
        nrows = 0
        for word in f.readlines():
            w = word.strip()
            if words_set.get(w) is not None:
                wdict[w] = mat[nrows]
            nrows += 1
    return wdict

def closest_words(gloves, word, n):
    """
    Given a gloves dictionary of word:vector and a word return the n nearest words
    as a list of strings. The word is not considered his own nearest neighbor,
    so do not include that in the returned list.

    Compute the Euclidean distance between the vector for word and
    every other word's vector. Track the distances with a list of tuples
    of the form: (distance, word).  Sort the list by distance. Return a list
    of the first n words from the sorted list. Do not return the tuples, just
    the words. Return a python list of strings not numpy array.
    """
    vector = gloves[word]
    words_sim_ret = []
    for w, v in gloves.items():
        if word == w:
            continue
        #dist = np.sqrt(sum([(x-y)**2 for x, y in zip(vector, v)]))
        dist = np.sqrt(np.sum(np.square(vector - v)))
        words_sim_ret.append((dist, w))      # why not(w, dist)
    ret = sorted(words_sim_ret, key=lambda x: x[0], reverse=False)  
    words_list = [ word for dist, word in ret[:n] ]
    return words_list

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
    #xy = np.array([(m - n) for m, n in zip(gloves.get(x), gloves.get(y))])
    xy = gloves.get(x) - gloves.get(y)
    words_sim_ret = []
    v = gloves.get(z)
    for word, vector in gloves.items():
        if z == word:
            continue
        #zv = np.array([ (j - i) for i, j in zip(vector, v)])
        zv = v - vector
        #dist = np.sqrt(sum([ ( m - n )** 2 for m, n in zip(xy, zv)]))
        dist = np.sqrt(np.sum(np.square(xy - zv)))
        words_sim_ret.append((dist, word))

    ret = sorted(words_sim_ret, key=lambda x: x[0], reverse=False)
    words = [word for dist, word in ret[:n]]
    return words

#wdict = load_glove(script_path)
#print(closest_words(wdict, 'lizard', 5))
#print(closest_words(wdict, 'russia', 5))
#print(analogies(wdict, 'shoe', 'foot', 'glove', 5))

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
    words_list = []
    words_vector = []
    for word in words:
        words_list.append(word)
        words_vector.append(gloves.get(word))

        ws = closest_words(gloves, word, n)
        words_list.extend(ws)
        for w in ws:
            words_vector.append(gloves.get(w))

    pca = PCA(n_components=2)
    pca.fit(words_vector)
    cpt_2D = pca.transform(words_vector)
    plt.scatter(cpt_2D[:, 0], cpt_2D[:, 1])
    for idx, word in enumerate(words_list):
        x, y = cpt_2D[idx]
        plt.text(x, y, word, fontsize=9)
    plt.show()

#plot_words(wdict,['petal','glove','computer'], 4)
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
