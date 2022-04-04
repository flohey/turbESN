#turbESN
from .core import (_DTYPE, _DEVICE, _LOGGING_FORMAT, ESN)

#Backends
import torch
import numpy as np

#Misc. 
from typing import Union, Tuple, List
import logging

#Data structure
import h5py

class CrossValidation():

    def __init__(self, 
                    esn: ESN,
                    n_folds: int,
                    fold_length: int=None, 
                    n_training_folds: int=2,
                    n_testing_folds: int=1, 
                    n_validation_folds: int=1,
                    data: torch.Tensor = None):

        logging.basicConfig(format=_LOGGING_FORMAT, level= esn.logging_level)

        self.n_folds = n_folds
        self.fold_length = fold_length
        self.n_training_folds = n_training_folds
        self.n_testing_folds = n_testing_folds
        self.n_validation_folds = n_validation_folds


        if data is not None:
            self.data_folded_u, self.data_folded_y = self.prepare_k_fold_walk_forward_validation_auto_data(data,
                                                                                                  esn_start=esn.esn_start,
                                                                                                  esn_end=esn.esn_end,
                                                                                                  n_folds=self.n_folds,
                                                                                                  fold_length=self.fold_length)

            if self.fold_length is None:
                self.fold_length = data_folded_u.shape[1]

    #--------------------------------------------------------------------------
    #FH 30.03.2022: added prepare_k_fold_walk_forward_validation_auto_data
    def prepare_k_fold_walk_forward_validation_auto_data(self, 
                                                        data: Union[np.ndarray, torch.Tensor], 
                                                        esn_start: int,
                                                        esn_end: int,
                                                        n_folds: int, 
                                                        fold_length: int = None):
        '''Prepares the data for k fold walk forward validation, see Lukosevicius et al. (2021). Divides the data into n_folds w. length fold_length.
        The cross validation will use the folds to create three parts: training (incl. transient), validation, testing
        For this method n_output = n_input must be valid. 

        INPUT:
            data           - data to use for ESN. Shape: (datatimesteps, n_input/n_output)
            esn_start      - Index of the original data, at which the training will begin. 
            esn_end        - Index of the original data, at which the testing will end.
            n_folds        - no. folds into which data will be divided into
            fold_length    - length of each fold, same for all n_folds folds
        
        RETURN:
            data_folded - data of shape (n_folds, fold_length, n_input/n_output)
        '''

        n_input = data.shape[1]
        data_esn = data[esn_start-1:esn_end,:]

        if fold_length is None:
            fold_length = int((esn_end - esn_start)/n_folds)     # choose fold_length according to data length

        assert esn_end - esn_start  >= n_folds*fold_length, f"Error: Chosen n_folds {n_folds} and fold_length {fold_length} do not fit in specified interval {esn_end - esn_start} steps"

        data_folded_u = data_esn[0:int(n_folds*fold_length),:].reshape(n_folds,fold_length,n_input)
        data_folded_y = data_esn[1:int(n_folds*fold_length)+1,:].reshape(n_folds,fold_length,n_input)
        
        return torch.as_tensor(data_folded_u,dtype=_DTYPE), torch.as_tensor(data_folded_y,dtype=_DTYPE)


    def save(self, filepath, f=None):
        ''' Saves the reservoir training, testing and validation data from CrossValidation class object into a hdf5 file.
            Data is saved as folded input and output data. ESN.save() should have been done before.
    
        INPUT:
            filepath   - path to which the hdf5 file of the ESN study was saved to
        '''

        if f is None:
            f = h5py.File(filepath, 'a')
            toClose = True
        
        logging.warn('Saving CrossValidation parameters to Hdf5 file {0}'.format(filepath))

        G_hp = f.get('Hyperparameters')
        G_hp.attrs['n_training_folds'] = self.n_training_folds
        G_hp.attrs['n_testing_folds'] = self.n_testing_folds
        G_hp.attrs['n_validation_folds'] = self.n_validation_folds

        G_data = f.get("Data")
        if G_data is None:
            G_data = f.create_group('Data')

        G_data.create_dataset('data_folded_u',   data = self.data_folded_u, compression = 'gzip', compression_opts = 9)
        G_data.create_dataset('data_folded_y',   data = self.data_folded_y, compression = 'gzip', compression_opts = 9)

        
        if toClose:
            f.close()




#--------------------------------------------------------------------------     
#--------------------------------------------------------------------------
#  CLASS METHODS
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
    @classmethod
    def read(cls, filepath: str, esn: ESN):


        try:
            with h5py.File(filepath,'r') as f:
                pass
        except:
            logging.debug('Error: file {0} not found.'.format(filepath))
        
        
        with h5py.File(filepath,'r') as f:

            G_hp = f.get('Hyperparameters')
            n_training_folds = G_hp.attrs['n_training_folds']
            n_testing_folds = G_hp.attrs['n_testing_folds']
            n_validation_folds = G_hp.attrs['n_validation_folds']

            G_data = f.get('Data')
            if 'data_folded_u' in G_data:
                data_folded_u = np.array(G_data.get('data_folded_u'))
                n_folds, fold_length, _ = data_folded_u.shape
            else:
                data_folded_u = None

            if 'data_folded_y' in G_data:
                data_folded_y = np.array(G_data.get('data_folded_y'))                                    
            else:
                data_folded_y = None

            assert data_folded_u is not None, "Error: couldnt find data_folded_u"
            assert data_folded_y is not None, "Error: couldnt find data_folded_y"
            
            cv = CrossValidation(esn =esn,
                                n_folds=n_folds,
                                fold_length=fold_length, 
                                n_training_folds=n_training_folds,
                                n_testing_folds=n_testing_folds, 
                                n_validation_folds=n_validation_folds)

            cv.data_folded_u = data_folded_u
            cv.data_folded_y = data_folded_y
        return cv
