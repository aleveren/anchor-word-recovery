from __future__ import division, print_function

import numpy as np
import scipy.sparse
import scipy.io
import sys

# Define basestr for python 2 vs 3 compatibility
try:
    basestr
except NameError as e:
    basestr = str

def main():
    if len(sys.argv) < 3:
        print("usage: input_matrix vocab_file cutoff")
        sys.exit()

    input_matrix = sys.argv[1]
    full_vocab = sys.argv[2]

    # cutoff for the number of distinct documents that each word appears in
    cutoff = int(sys.argv[3])

    M, trunc_vocab = truncate(
        input_matrix = input_matrix,
        input_vocab = full_vocab,
        stopwords = 'stopwords.txt',
        cutoff = cutoff)

    output_matrix = sys.argv[1]+".trunc"
    output_vocab = sys.argv[2]+".trunc"

    scipy.io.savemat(output_matrix, {'M' : M}, oned_as='column')

    print('New number of words is ', M.shape[0])
    print('New number of documents is ', M.shape[1])

    # Output the new vocabulary
    with open(output_vocab, 'w') as output:
        for word in trunc_vocab:
            print(word, file=output)

def truncate(input_matrix, input_vocab, stopwords, cutoff):
    if isinstance(input_matrix, basestr):
        # Load previously generated document
        M = scipy.io.loadmat(input_matrix)['M']
    else:
        M = input_matrix

    if isinstance(input_vocab, basestr):
        # Load the vocabulary
        vocab_list = []
        with open(input_vocab, 'r') as file:
            for line in file:
                vocab_list.append(line.rstrip())
    else:
        vocab_list = input_vocab

    table = {word: i for i, word in enumerate(vocab_list)}
    numwords = len(vocab_list)
    remove_word = [False]*numwords

    if isinstance(stopwords, basestr):
        # Read in the stopwords
        with open(stopwords, 'r') as file:
            for line in file:
                if line.rstrip() in table:
                    remove_word[table[line.rstrip()]] = True
    else:
        for word in enumerate(stopwords):
            if word in table:
                remove_word[table[word]] = True

    if M.shape[0] != numwords:
        print('Error: vocabulary file has different number of words', M.shape, numwords)
        sys.exit()
    print('Number of words is ', numwords)
    print('Number of documents is ', M.shape[1])

    M = M.tocsr()

    new_indptr = np.zeros(M.indptr.shape[0], dtype=np.int32)
    new_indices = np.zeros(M.indices.shape[0], dtype=np.int32)
    new_data = np.zeros(M.data.shape[0], dtype=np.float64)

    indptr_counter = 1
    data_counter = 0
    trunc_vocab = []

    for i in range(M.indptr.size - 1):
        if not remove_word[i]:
            trunc_vocab.append(vocab_list[i])

            # start and end indices for row i
            start = M.indptr[i]
            end = M.indptr[i + 1]

            # if number of distinct documents that this word appears in is >= cutoff
            if (end - start) >= cutoff:
                new_indptr[indptr_counter] = new_indptr[indptr_counter-1] + end - start
                new_data[new_indptr[indptr_counter-1]:new_indptr[indptr_counter]] = M.data[start:end]
                new_indices[new_indptr[indptr_counter-1]:new_indptr[indptr_counter]] = M.indices[start:end]
                indptr_counter += 1
            else:
                remove_word[i] = True

    new_indptr = new_indptr[0:indptr_counter]
    new_indices = new_indices[0:new_indptr[indptr_counter-1]]
    new_data = new_data[0:new_indptr[indptr_counter-1]]

    M = scipy.sparse.csr_matrix((new_data, new_indices, new_indptr))
    M = M.tocsc()

    return M, trunc_vocab

if __name__ == "__main__":
    main()
