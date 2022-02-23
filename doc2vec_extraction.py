import os
import dill
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class Corpora:
    def __init__(self, corpus_paths: list):
        '''
        Args:
            corpus_paths(list): paths to corpus file
        '''

        # load sentences
        idx = dict()
        for i, p in enumerate(corpus_paths):
            lines = open(p, mode='r', encoding='utf-8').readlines()
            idx[i] = {j: line for j, line in enumerate(lines)}

        # tag sentences with Corpus_tag and Sentence_tag
        self._corpora = list()
        for i in sorted(idx.keys()):
            self._corpora.append([TaggedDocument(s.split(), [f'C{i}_{j}'])
                                  for j, s in sorted(idx[i].items(), key=lambda x: int(x[0]))])

        self.periods = np.arange(0, len(self._corpora), 1)

        # set up docvecs to nan
        self.docvecs = np.array([np.array([[np.nan]*100 for doc in corpus])
                                 for corpus in self._corpora], dtype=object)

    def _doc2vec(self, vector_size: int = 100, window: int = 6, min_count: int = 1, workers: int = 6, epochs: int = 10,
                 load: bool = False, filename: str = None, train_on_fail: bool = True) -> object:
        '''
        Args:
            vector_size(int, optional, default=100): Dimensionality of the feature vectors
            window(int, optional, default=6): The maximum distance between the current and predicted word within a sentence
            min_count(int, optional, default=1): Ignores all words with total frequency lower than this
            workers(int, optional, default=6): Use these many worker threads to train the model (=faster training with multicore machines)
            epochs(int, optional, default=10): Number of iterations (epochs) over the corpus
            load(bool, optional, default=False): Load a pre-trained doc2vec, if True and filename set
            filename(str, optional, default=None): Load/Store doc2vec from this file
            train_on_fail(bool, optional, default=True): Safe loading. If the loading of a pre-trained model fails, a new model is trained and stored in filename
        '''

        # try to load a pre-trained doc2vec
        if load:
            # load doc2vec lookup table if exists
            if os.path.exists(filename):
                self.model = Doc2Vec.load(filename)
            # train a new doc2vec model
            elif train_on_fail:
                self.model = None
            # file not found
            else:
                raise FileNotFoundError

        # train a new doc2vec model
        if not load or (train_on_fail and self.model is None):
            self.model = Doc2Vec([doc for corpus in self._corpora for doc in corpus],
                                 vector_size=vector_size,
                                 window=window,
                                 min_count=min_count,
                                 workers=workers,
                                 epochs=epochs)

            # store model if filename is set
            if filename is not None:
                dir = os.path.dirname(filename)
                if dir:
                    os.makedirs(dir, exist_ok=True)
                self.model.save(filename)

        # keep doc vectors in memory
        self.docvecs = np.array([np.array([self.model.dv[doc.tags[0]] for doc in corpus])
                                 for corpus in self._corpora], dtype=object)

        return self.docvecs

    def train_model(self, mode: str = 'doc2vec', **params) -> object:
        '''Represents sentences from different corpora in a vector space

        Args:
            mode(str, optional, default='doc2vec'): the mode to transform docs to vectors
            params: additional parameters of the called function
        '''
        return getattr(self, f'_{mode}')(**params)

    def getWordCorpus(self, time_tag: int, word: str) -> object:
        '''Returns a WordCorpus, that is a collection of text sequences in which the target word occurs.

        Args:
            time_tag(int): the i-th period
            word(str): a word of interest
        Returns:
            WordCorpus
        '''
        return WordCorpus(word, time_tag, self._corpora[time_tag], self.docvecs[time_tag])

    def __getitem__(self, time_tag):
        return self._corpora[time_tag]


class WordCorpus:
    '''Corpus containing only sentences related to specific words'''

    def __init__(self, word: str, time_tag: int, corpus:object, docvecs:object):
        '''
        Args:
            word(str): a word of intereset
            time_tag(int): the i-th time period
            corpus(object): collection of text sequences from time period i-th
            docvecs(object): doc-vectors for the text sequences
        '''

        self.word = word
        self.time_tag = time_tag

        # index of sentences containing the word of intereset
        mask = np.array([word in d.words for d in corpus], dtype=bool)

        # sentences containing the word of intereset
        self.word_corpus = np.array(corpus, dtype=object)[mask, :]

        # docvecs for such sentences
        self.word_docvecs = docvecs[mask, :]

        # set up wordvecs to nan
        self.wordvecs = np.array([np.nan]*self.word_corpus.shape[0], dtype=object)

    def get_text(self, sequence:object) -> str:
        '''Returns text of a target sequence

        Args:
            sentence: (int or np.array(TaggedDocument))). Sequence id if int. Else encapsuled Tagged Document
        Returns:
            Text of target sequence
        '''

        # sequence is a TaggedDocument
        if not isinstance(sequence, int):
            words = sequence[0]  # sequence.words
        else:
            # isinstance(sequence, int) --> sequence id
            try:
                words = self.word_corpus[sequence].words
            except IndexError:
                raise Exception('Unavailable document')

        return " ".join(words)

    def get_texts(self, sequence_ids:list) -> list:
        '''Returns text of target sequences'''
        return [self.get_text(i) for i in sequence_ids]

    def get_vectors(self, sequence_ids:list) -> object:
        '''Returns docvecs of target sequences'''
        return self.word_docvecs[sequence_ids, :]

    def get_centroid(self, sequence_ids:list) -> object:
        '''Returns centroid for target docvecs'''
        return self.get_vectors(sequence_ids).mean(axis=0)

    def get_cluster_stats(self, sequence_ids) -> tuple:
        '''Returns stats for a cluster of target docvecs'''
        centroid = self.get_centroid(sequence_ids)
        delta = cosine_distances(centroid.reshape(1, -1), self.get_vectors(sequence_ids))
        return delta.mean(), delta.std(), delta.min(), delta.max()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_paths", default='data/corpora/processed_ccoha1.txt;data/corpora/processed_ccoha2.txt', type=str,
                        help="Paths to all corpus time slices separated by ';'.")
    parser.add_argument("--vector_size", default=100,
                        type=int,
                        help="Dimensionality of the feature vectors.")
    parser.add_argument("--window", default=10,
                        type=int,
                        help="The maximum distance between the current and predicted word within a sentence.")
    parser.add_argument("--workers", default=10,
                        type=int,
                        help="Use these many worker threads to train the model (=faster training with multicore machines).")
    parser.add_argument("--epochs", default=15,
                        type=int,
                        help="Number of iterations (epochs) over the corpus.")
    parser.add_argument("--model_path", default='doc2vec_english.gnsm', type=str,
                        help="Path to output file containing doc2vec model.")
    parser.add_argument("--load", default=False, type=bool,
                        help="Load a pre-trained doc2vec model (stored in model_path).")
    parser.add_argument("--train_on_fail", default=True, type=bool,
                        help="Safe loading. If the load of a pre-trained model fails, a new model is trained and stored.")
    parser.add_argument("--target_path", default='data/targets/en_targets.txt', type=str,
                        help="Path to target file")
    parser.add_argument("--language", default='english', const='all', nargs='?',
                        help="Choose a language", choices=['english', 'latin'])
    parser.add_argument("--embeddings_path", default='embeddings_doc2vec_english.pickle', type=str,
                        help="Path to output pickle file containing embeddings.")
    args = parser.parse_args()

    C = Corpora(args.corpus_paths.split(';'))
    C.train_model(mode='doc2vec',
                  vector_size=args.vector_size,
                  window=args.window,
                  workers=args.workers,
                  epochs=args.epochs,
                  filename=args.model_path,
                  load=True,
                  train_on_fail=True)

    topickle = dict()

    print('Language:', args.language.upper())
    targets = pd.read_csv(args.target_path, sep='\t', names=['word'])
    for target in tqdm(targets.word.values):
        topickle[target] = dict()

        # WordCorpus period 0, # WordCorpus period 1
        wc0 = C.getWordCorpus(time_tag=0, word=target)
        wc1 = C.getWordCorpus(time_tag=1, word=target)

        # load wordvectors (if available else extract them)

        topickle[target]['t1_text'] = wc0.get_texts(wc0.word_corpus)
        topickle[target]['t2_text'] = wc1.get_texts(wc1.word_corpus)

        topickle[target]['t1'] = [v for v in wc0.word_docvecs]
        topickle[target]['t2'] = [v for v in wc1.word_docvecs]

    dir = os.path.dirname(args.output_corr_file)
    if dir:
        os.makedirs(dir, exist_ok=True)
    dill.dump(topickle, open(args.embeddings_path, mode='wb'))