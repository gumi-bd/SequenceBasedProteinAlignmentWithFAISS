#import sys
#sys.path.append("/Users/dhanushkalva/anaconda3/envs/compbio/lib/python3.10/site-packages/faiss")
import faiss
import annoy
import pandas as pd

class faissKNN:
    def __init__(self, vectors, labels) -> None:
        self.vectors = vectors.astype('float32')
        self.labels = labels
    def build(self, nlist = 100):
        self.indexFlat = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIVFFlat(self.indexFlat, self.dimension, nlist)
        assert not self.index.is_trained
        self.index.train(self.vectors)
        assert self.index.is_trained
        self.index.add(self.vectors)
        
    def query(self, vectors, k=10, nprobe = 1):
        distances, indices = self.index.search(vectors, k) 
        res = []
        for j in range(len(indices)):
            res.append((sum([self.labels[i] for i in indices[j]]))/k)
        return res
    
class approxNeighbour:
    def __init__(self, vectors, labels) -> None:
        self.dimension = vectors.shape[1]
        self.vectors = vectors.astype('float32')
        self.labels = labels    
   
    def build(self, number_of_trees=5):
        self.index = annoy.AnnoyIndex(self.dimension)
        for i, vec in enumerate(self.vectors):
            self.index.add_item(i, vec.tolist())
        self.index.build(number_of_trees)
        
    def query(self, vector, k=10):
        indices = self.index.get_nns_by_vector(vector.tolist(), k, search_k=3)                                           
        return ([self.labels[i] for i in indices].sum())/k
    
class batchApproxNeighbour():
    def __init__(self) -> None:
        pass
 