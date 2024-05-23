Finding minimum width of the layers of the network

SOLVER-wide.py    :
train and find minimum layer size with cross-validation, C=3:
 python SOLVER-wide.py trainsform1 300      - for transform1 layer, initial layer width=300

train only with cross-validation, C=3:
 python SOLVER-wide.py none 300      - initial layer width=300


SOLVER-opt.py    :
train with minimum layer sizes of transform1 and transform2 layers:
 python SOLVER-opt.py

CompareModels.py:
 compare two ensembles of the models, created by SOLVER-wide.py and SOLVER-opt.py

