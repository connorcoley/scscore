'''
This script is meant to save the list of IDs used in the Reaxys set, because 
we cannot make the actual data available, unfortunately
'''
import os 
project_root = os.path.dirname(os.path.dirname(__file__))

import random
path = raw_input('Enter path to Reaxys .txt file: ')
data = []
with open(path, 'r') as f:
    for line in f:
        rex, n, _id = line.strip("\r\n").split(' ')
        r,p = rex.split('>>')
        if ('.' in p) or (not p):
            continue # do not allow multiple products or none
        n = int(n)
        for r_splt in r.split('.'):
            if r_splt:
                data.append((_id, n, r_splt, p))

random.seed(123)
random.shuffle(data)
data_len = len(data)

print('After splitting, %i total data entries' % data_len)
print('...slicing data and saving IDs')
train_data = data[:int(0.8 * data_len)]
val_data = data[int(0.8 * data_len):int(0.9 * data_len)]
test_data = data[int(0.9 * data_len):]

import cPickle as pickle
with open(os.path.join(project_root, 'data', 'reaxys_ids_train.pickle'), 'wb') as fid:
	pickle.dump([int(x[0]) for x in train_data], fid, -1)
with open(os.path.join(project_root, 'data', 'reaxys_ids_valid.pickle'), 'wb') as fid:	
	pickle.dump([int(x[0]) for x in val_data], fid, -1)
with open(os.path.join(project_root, 'data', 'reaxys_ids_test.pickle'), 'wb') as fid:
	pickle.dump([int(x[0]) for x in test_data], fid, -1)