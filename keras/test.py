import logging
import codecs
import pickle

import tables

logger = logging.getLogger(__name__)
vocabfile = "./data/full_github/vocab.desc.pkl"
vocab_tokens = pickle.load(open("./data/full_github/vocab.tokens.pkl", 'rb'))
vocab_apiseq = pickle.load(open("./data/full_github/vocab.apiseq.pkl", 'rb'))
vocab_desc = pickle.load(open("./data/full_github/vocab.desc.pkl", 'rb'))
vocab_methname = pickle.load(open("./data/full_github/vocab.methname.pkl", 'rb'))

test_methname = "./data/full_github/javaTest/test.java.methname.txt"
test_apiseq = "./data/full_github/javaTest/test.java.apiseq.txt"
test_tokens = "./data/full_github/javaTest/test.java.tokens.txt"
test_desc = "./data/full_github/javaTest/test.java.desc.txt"

voca = pickle.load(open(vocabfile, 'rb'))


# load javadoc query results for test
def load_javadoc_data( chunk_size):
    logger.debug('Loading a chunk of validation data..')
    logger.debug('methname')
    chunk_methnames = load_txt_file(test_methname, chunk_size)
    logger.debug('apiseq')
    chunk_apiseqs = load_txt_file(test_apiseq, chunk_size)
    logger.debug('tokens')
    chunk_tokens = load_txt_file(test_tokens, chunk_size)
    logger.debug('desc')
    chunk_descs = load_txt_file(test_desc, chunk_size)
    chunk_tokens = [convert(vocab_tokens, i) for i in chunk_tokens]
    chunk_apiseqs = [convert(vocab_apiseq, i) for i in chunk_apiseqs]
    chunk_methnames = [convert(vocab_methname, i) for i in chunk_methnames]
    chunk_descs = [convert(vocab_desc, i) for i in chunk_descs]
    return chunk_methnames, chunk_apiseqs, chunk_tokens, chunk_descs

def load_txt_file( filename, chunk_size):
    logger.info('Loading txt file (size={})..'.format(chunk_size))
    #with codecs.open(filename, encoding='utf8', errors='replace').readlines() as codes:
    with open(filename, 'r')as codefile:
        codes = codefile.readlines()
        txt = []
        for i in codes:
            txt.append(i)
        return txt

def convert( vocab, words):
    """convert words into indices"""
    if type(words) == str:
        words = words.strip().lower().split(' ')
    return [vocab.get(w, 0) for w in words]


def revert( vocab, indices):
    """revert indices into words"""
    ivocab = dict((v, k) for k, v in vocab.items())
    return [ivocab.get(i, 'UNK') for i in indices]

# fk = load_hdf5("./data/full_github/test.desc.h5", 0, 10)
# for i in fk:
#     ff = revert(voca, i)
#     print(ff)


methnames, apiseqs, tokens, descs = load_javadoc_data(-1)
print([revert(vocab_methname, i) for i in methnames])
print([revert(vocab_tokens, i) for i in tokens])
print([revert(vocab_apiseq, i) for i in apiseqs])
print([revert(vocab_desc, i) for i in descs])

