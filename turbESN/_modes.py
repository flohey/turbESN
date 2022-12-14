import torch 

_DTYPE = torch.float32              # dtype of all torch tensors used with the ESN
_DEVICE = torch.device('cpu')       # device on which ESN runs, cpu: stable, gpu: untested
_ESN_MODES = ('auto', 'teacher', 'semi-teacher')           # prediction modes 
_WEIGTH_GENERATION = ('uniform', 'normal')                 # random weight generation
_EXTENDED_STATE_STYLES = ('default', 'square')             # layout of extended state 
_FIT_METHODS = ('tikhonov', 'pinv')                        # method of how Wout is computed
_LOGGING_FORMAT = '%(asctime)s %(threadName)s %(levelname)s: %(message)s'
_ID_PRINT = 4        # if esn.id % _ID_PRINT == 0: log ESN run (used in study.py only)
_LOSS_DEFAULT = 1e6  # default value for the loss, if nan values are encounterd during ESN run (used in run_turbESN in util.py) 