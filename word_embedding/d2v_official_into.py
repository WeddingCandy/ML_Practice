import gensim
import os
import collections
import smart_open
import random
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec

"""
https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb
"""


test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])#os.path :the  seperation label in present os
lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
lee_test_file = test_data_dir + os.sep + 'lee.cor'
"""
Define a Function to Read and Preprocess Text
Below, we define a function to open the train/test file (with latin encoding), 
read the file line-by-line, pre-process each line using a simple gensim pre-processing tool 
(i.e., tokenize text into individual words, remove punctuation, set to lowercase, etc), 
and return a list of words. Note that, for a given file (aka corpus), 
each continuous line constitutes a single document and the length of each line (i.e., document) can vary. 
Also, to train the model, we'll need to associate a tag/number with each document of the training corpus. 
In our case, the tag is simply the zero-based line number.
"""
def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(simple_preprocess(line), [i])

train_corpus = list(read_corpus(lee_train_file))
print(train_corpus)
test_corpus = list(read_corpus(lee_test_file, tokens_only=True))

model = Doc2Vec(size=50, min_count=2,workers=5)

model.build_vocab(train_corpus)

model.train(train_corpus, total_examples=model.corpus_count, epochs=10)

ranks = []
second_ranks = []
for doc_id in range(len(train_corpus)): #len(train_corpus)
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)

    second_ranks.append(sims[1]) #the second one
collections.Counter(ranks)


print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))


doc_id = random.randint(0, len(train_corpus) - 1)
print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
sim_id = second_ranks[doc_id]
print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_corpus[sim_id[0]].words)))


## Testing
# Pick a random document from the test corpus and infer a vector from the model
doc_id = random.randint(0, len(test_corpus) - 1)
inferred_vector = model.infer_vector(test_corpus[doc_id])
sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

# Compare and print the most/median/least similar documents from the train corpus
print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

