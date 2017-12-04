# scscore

### Description
The SCScore model assigns a synthetic complexity score between 1 and 5 to a molecule. The score is based on the premise that published reactions, overall, should exhibit an increase in synthetic complexity. The model has been trained on 12M reactions from Reaxys.

### Usage
The standalone numpy model is defined in ```scscore/standalone_model_numpy.py```

### Dependencies if you want to use the final model
- RDKit (most versions should be fine)
- numpy

### Dpendencies if you want to retrain on your own data
- RDKit (most versions should be fine)
- tensorflow (r0.12.0)
- h5py
- numpy
