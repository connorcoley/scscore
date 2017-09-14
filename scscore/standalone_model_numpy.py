'''
This is a standalone, importable SCScorer model. It does not have tensorflow as a
dependency and is a more attractive option for deployment. The calculations are 
fast enough that there is no real reason to use GPUs (via tf) instead of CPUs (via np)
'''

import math, sys, random, os
import numpy as np
import time
import rdkit.Chem as Chem 
import rdkit.Chem.AllChem as AllChem

import os 
project_root = os.path.dirname(os.path.dirname(__file__))

score_scale = 5.0
min_separation = 0.25

FP_len = 1024
FP_rad = 2

def mol_to_fp(mol, radius=FP_rad, nBits=FP_len):
    if mol is None:
        return np.zeros((nBits,), dtype=np.float32)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, 
        useChirality=True), dtype=np.bool)

def smi_to_fp(smi, radius=FP_rad, nBits=FP_len):
    if not smi:
        return np.zeros((nBits,), dtype=np.float32)
    return mol_to_fp(Chem.MolFromSmiles(smi), radius, nBits)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class SCScorer():
    def __init__(self, score_scale=score_scale):
        self.vars = []
        self.score_scale = score_scale

    def restore(self, weight_path):
        import cPickle as pickle
        with open(weight_path, 'rb') as fid:
            self.vars = pickle.load(fid)
        print('Restored variables from {}'.format(weight_path))
        return self

    def apply(self, x):
        # Each pair of vars is a weight and bias term
        for i in range(0, len(self.vars), 2):
            last_layer = (i == len(self.vars)-2)
            W = self.vars[i] 
            b = self.vars[i+1]
            x = np.matmul(x, W) + b
            if not last_layer:
                x = x * (x > 0) # ReLU
        x = 1 + (score_scale - 1) * sigmoid(x)
        return x

    def get_score_from_smi(self, smi='', v=False):
        if not smi:
            return ('', 0.)
        fp = np.array((smi_to_fp(smi)), dtype=np.float32)
        if sum(fp) == 0:
            if v: print('Could not get fingerprint?')
            cur_score = 0.
        else:
            # Run
            cur_score = self.apply(fp)
            if v: print('Score: {}'.format(cur_score))
        mol = Chem.MolFromSmiles(smi)
        if mol:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True)
        else:
            smi = ''
        return (smi, cur_score)

if __name__ == '__main__':
    model = SCScorer()    
    model.restore(os.path.join(project_root, 'models', 'full_reaxys_model', 'model.ckpt-10654.as_numpy.pickle'))
    smis = ['CCCOCCC', 'CCCNc1ccccc1']
    for smi in smis:
        (smi, sco) = model.get_score_from_smi(smi)
        print('%.4f <--- %s' % (sco, smi))