from __future__ import division, print_function

import sys
from numpy.random import RandomState
import numpy as np
from fastRecover import do_recovery
from anchors import findAnchors
import scipy.sparse as sparse
import time
from Q_matrix import generate_Q_matrix 
import scipy.io
from collections import namedtuple
import argparse

_Params_fields = ["seed", "eps", "max_threads", "new_dim", "anchor_thresh",
    "top_words", "infile", "vocab_file", "K", "loss", "outfile"]

class Params(namedtuple("Params", _Params_fields)):
    @classmethod
    def from_dict(cls, d):
        kwargs = dict(
            outfile = None,
            seed = int(time.time()))
        kwargs.update(d)
        return cls(**kwargs)

    @classmethod
    def from_file_and_cmd_args(cls, filename, extra_args):
        kwargs = dict(
            outfile = None,
            seed = int(time.time()))

        with open(filename) as f:
            for l in f:
                if l == "\n" or l[0] == "#":
                    continue
                l = l.strip()
                l = l.split('=')
                if l[0] == "max_threads":
                    kwargs["max_threads"] = int(l[1])
                elif l[0] == "eps":
                    kwargs["eps"] = float(l[1])
                elif l[0] == "new_dim":
                    kwargs["new_dim"] = int(l[1])
                elif l[0] == "seed":
                    kwargs["seed"] = int(l[1])
                elif l[0] == "anchor_thresh":
                    kwargs["anchor_thresh"] = int(l[1])
                elif l[0] == "top_words":
                    kwargs["top_words"] = int(l[1])
                else:
                    raise ValueError("Unrecognized param: '{}'".format(l[0]))

        kwargs.update(extra_args)

        return cls(**kwargs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("settings_file")
    parser.add_argument("vocab_file")
    parser.add_argument("K", type=int)
    parser.add_argument("loss")
    parser.add_argument("outfile")
    args = parser.parse_args()

    settings_file = args.settings_file
    del args.settings_file

    params = Params.from_file_and_cmd_args(settings_file, extra_args = vars(args))
    analysis = Analysis(params)
    analysis.run()

TopWordsSummary = namedtuple("TopWordsSummary",
    ["topic_index", "anchor_word_index", "anchor_word",
    "top_word_indices", "top_words"])

# Define basestr for python 2 vs 3 compatibility
try:
    basestr
except NameError as e:
    basestr = str

class Analysis(object):
    def __init__(self, *args, **kwargs):
        # TODO: clean up how params are set (should just use named args?)
        if len(args) == 1:
            self.params = args[0]
        elif len(args) == 0:
            if "params" in kwargs:
                self.params = kwargs["params"]
            else:
                self.params = Params.from_dict(kwargs)
        else:
            raise ValueError("Unexpected number of positional args")

    def run(self):
        params = self.params

        if isinstance(params.infile, basestr):
            M = scipy.io.loadmat(params.infile)['M']
        else:
            M = params.infile
        assert sparse.isspmatrix_csc(M), "Must provide a sparse CSC matrix"

        print("Input matrix shape: {}".format(M.shape))

        if isinstance(params.vocab_file, basestr):
            with open(params.vocab_file) as f:
                vocab = f.read().strip().split()
        else:
            vocab = params.vocab_file
        assert np.iterable(vocab), "Must provide an iterable vocab"

        assert M.shape[0] == len(vocab), \
            "Number of rows must correspond to vocab size: {} rows vs {} vocab words" \
            .format(M.shape[0], len(vocab))

        #only accept anchors that appear in a significant number of docs
        print("identifying candidate anchors")
        candidate_anchors = []
        for i in range(M.shape[0]):
            if len(np.nonzero(M[i, :])[1]) > params.anchor_thresh:
                candidate_anchors.append(i)

        print(len(candidate_anchors), "candidates")

        #forms Q matrix from document-word matrix
        Q = generate_Q_matrix(M)

        # Save copy of unnormalized Q, before any normalizations happen
        self.Q_unnormalized = Q.copy()

        #check that Q sum is 1 or close to it
        print("Q sum is", Q.sum())
        V = Q.shape[0]
        print("done reading documents")

        #find anchors- this step uses a random projection
        #into low dimensional space
        anchors = findAnchors(Q, params, candidate_anchors)
        print("anchors are:")
        for i, a in enumerate(anchors):
            print(i, vocab[a])

        #recover topics
        A, topic_likelihoods = do_recovery(Q, anchors, params)
        print("done recovering")

        output_streams = [sys.stdout]
        output_file_handle = None
        if params.outfile is not None:
            np.savetxt(params.outfile+".A", A)
            np.savetxt(params.outfile+".topic_likelihoods", topic_likelihoods)
            output_file_handle = open(params.outfile+".topwords", 'w')
            output_streams.append(output_file_handle)

        def print_multiple(*args, **kwargs):
            # Print the same info to multiple output streams
            for f in output_streams:
                print(*args, file=f, **kwargs)

        # Display top words per topic
        all_topwords = []
        for k in range(params.K):
            topwords = np.argsort(A[:, k])[-params.top_words:][::-1]
            print_multiple(vocab[anchors[k]], ':', end=' ')
            for w in topwords:
                print_multiple(vocab[w], end=' ')
            print_multiple("")
            all_topwords.append(TopWordsSummary(
                topic_index = k,
                anchor_word_index = anchors[k],
                anchor_word = vocab[anchors[k]],
                top_word_indices = topwords,
                top_words = [vocab[w] for w in topwords]))

        if params.outfile is not None:
            output_file_handle.close()

        # make some results available as attributes of "self"
        self.Q = Q
        self.M = M
        self.A = A
        self._R = None
        self.topic_likelihoods = topic_likelihoods
        self.candidate_anchors = candidate_anchors
        self.anchors = anchors
        self.vocab = vocab
        self.all_topwords = all_topwords

    @property
    def R(self):
        if self._R is not None:
            return self._R

        Ap = np.linalg.pinv(self.A)
        self._R = Ap.dot(self.Q_unnormalized).dot(Ap.T)
        return self._R

if __name__ == "__main__":
    main()
