import torch
import numpy as np
import networkx as nx
import sys
import h5py 
from typing import Union, Tuple, List
import logging


_DTYPE = torch.float32
_DEVICE = torch.device('cpu')                      
_ESN_MODES = ('auto', 'teacher', 'semi-teacher')   # prediction modes 
_WEIGTH_GENERATION = ('uniform', 'normal')         # random weight generation
_EXTENDED_STATE_STYLES = ('default', 'square')    # layout of extended state 
_LOGGING_FORMAT = '%(asctime)s %(threadName)s %(levelname)s: %(message)s'

class ESN:

    _ID_COUNTER = 0

    def __init__(self, 
                randomSeed: int,
                esn_start: int, 
                esn_end: int,
                trainingLength: int, 
                testingLength: int,
                validationLength: int,
                data_timesteps: int,
                n_input: int,
                n_output: int,
                n_reservoir: int = 256,
                leakingRate: float = 1.0, 
                spectralRadius: float = 0.95,
                reservoirDensity: float = 0.2,
                regressionParameter: float = 5e-2,
                bias_in: float = 1.0,
                bias_out: float = 1.0,
                outputInputScaling: float = 1.0,
                inputScaling: float = 1.0, 
                inputDensity: float = 1.0, 
                noiseLevel_in: float = 0.0,
                noiseLevel_out: float = 0.0,
                mode: str = "auto",
                weightGeneration: str = "uniform",
                extendedStateStyle: str = "default",
                transientTime: int = 50,
                use_watts_strogatz_reservoir: bool = False,
                ws_p: float = 0.2,
                verbose: bool = False):


        self.device = _DEVICE

        if verbose:
            self.logging_level = logging.DEBUG
        else:
            self.logging_level = logging.WARNING

        logging.basicConfig(format=_LOGGING_FORMAT, level= self.logging_level)
        
        assert mode in _ESN_MODES,'Error: unkown mode {0}. Choices {1}'.format(mode, _ESN_MODES)
        self.mode = mode                                             # prediction mode (auto - autonmous prediction, teacher - teacher forced)
        logging.info('ESN mode is ' + self.mode + '\n')

        assert extendedStateStyle in _EXTENDED_STATE_STYLES,'Error: unkown extended state style {0}. Choices {1}'.format(extendedStateStyle, _EXTENDED_STATE_STYLES)
        self.extendedStateStyle = extendedStateStyle        
        
        ESN._ID_COUNTER +=1
        self.id = ESN._ID_COUNTER                                    #reservoir instance ID. Used for logging to keep track of multiple processes.

        self.randomSeed = randomSeed                                 #RNG seed for random reservoir initalization
        self.esn_start = esn_start                                   #Index of the original data, at which the training output y_train will begin. 
                                                                     #Note that the training input u_train will therefore begin at the index esn_start-1. Therefore esn_start can not be 0!
        self.esn_end = esn_end                                       #Index of the original data, at which the testing/validation output y_test will end.
        self.trainingLength = trainingLength                         #no. time steps for the training data set
        self.testingLength = testingLength                           #no. time stes for the testing data set
        self.validationLength = validationLength                     #no. time stes for the validation data set
        self.data_timesteps = data_timesteps                         #no. time steps the orignal data has/should have
        self.esn_timesteps = trainingLength + testingLength          #no. total resulting time steps for the esn 
        self.n_input = int(n_input)                                  #input data dimensions
        self.n_output = int(n_output)                                #output data dimensions
        self.n_reservoir = int(n_reservoir)                          #dimensions of reservoir state and W with shape (n_reservoir, n_reservoir)
        
        
        if self.extendedStateStyle == _EXTENDED_STATE_STYLES[0]:
            self.xrows = int(1+n_reservoir+n_input)                  #dim of extended reservoir state: (xrows,timesteps). xrows = bias + n_reservoir + n_reservoir
        else:
            self.xrows = int(1+2*n_reservoir)                        #dim of extended reservoir state: (xrows,timesteps). xrows = bias + n_reservoir + n_reservoir

        self.leakingRate = leakingRate                               #factor controlling the leaky integrator formulation (1 -> fully nonlinear, 0 -> fully linear)
        self.spectralRadius = spectralRadius                         #maximum absolute eigenvalue of W
        self.reservoirDensity = reservoirDensity                     #fraction of non-zero elements of W
        self.regressionParameter = regressionParameter               #ridge regression/ penalty parameter of ridge regression
        self.bias_in = bias_in                                       #input bias in the input mapping: Win*[1;u]
        self.bias_out = bias_out                                     #output bias in the final output mapping:  y = Wout*[outputbias; outputInputScaling*u; s]
        self.outputInputScaling = outputInputScaling                 #factor by which the input data should be scaled by in the final output mapping: y = Wout*[outputbias; outputInputScaling*u; s]
        self.inputScaling = inputScaling                             #scaling of the columns of the input matrix Win
        self.inputDensity = inputDensity                             #fraction of non-zero elements of Win
        self.noiseLevel_in = noiseLevel_in                           #amplitude of the gaussian noise term inside the activation function
        self.noiseLevel_out = noiseLevel_out                         #amplitude of the gaussian noise term outside the activation function

        assert weightGeneration in _WEIGTH_GENERATION,'Error: unkown weightGeneration {0}. Choices {1}'.format(weightGeneration, _WEIGTH_GENERATION)
        self.weightGeneration = weightGeneration                     #method the random weights Win, W should be initialized.
        self.use_watts_strogatz_reservoir = use_watts_strogatz_reservoir   #whether reservoir matrix is given by Watts-Strogatz network (for small ws_p --> small world network)
        self.ws_p = ws_p                                             #rewiring probability for the Watts-Strogatz reservoir 
        self.transientTime = transientTime                           #washout length for reservoir states


        self.y_train = None           
        self.u_train = None
        self.y_test  = None
        self.u_test  = None
        self.y_val  = None
        self.u_val  = None
        
        self.pred_init_input = None
        self.val_init_input = None


#--------------------------------------------------------------------------     
#--------------------------------------------------------------------------
#  ESN IMPLEMENTATION
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
    def createInputMatrix(self):
        '''
        Random initialization of the input weights. The weights are drawn from U[-0.5,0.5] or N[0,1] and subsequently scaled by the input scaling.

        RETURN:
            Win - reservoir input weight matrix
        '''
        
        logging.debug('Building input matrix')

        if self.weightGeneration == 'uniform':
            self.Win = torch.rand(self.n_reservoir, 1 + self.n_input, device = self.device, dtype = _DTYPE) - 0.5
        elif self.weightGeneration == 'normal':
            self.Win = torch.randn(self.n_reservoir, 1 + self.n_input, device = self.device, dtype = _DTYPE)
        else:
            self.Win = torch.rand(self.n_reservoir, 1 + self.n_input, device = self.device, dtype = _DTYPE) - 0.5


        if self.inputDensity >= 1e-6:
            _mask = torch.rand(self.n_reservoir, 1 + self.n_input, dtype = _DTYPE, device = self.device) > self.inputDensity
            self.Win[_mask] = 0.0
        else:
            self.Win = torch.zeros(self.n_reservoir, 1 + self.n_input, device = self.device, dtype = _DTYPE)

#--------------------------------------------------------------------------
    def createReservoirMatrix(self):
        '''
        Random initialization of the reservoir weights. The weights are drawn from U[-0.5,0.5]. The reservoir density is set.
        Finally, the largest absolute eigenvalue is computed, by which Wres is normalized. Then Wres is scaled by the spectral radius.

        RETURN:
            Wres - reservoir weight matrix
        '''
        
        logging.debug('Building reservoir matrix')

        if self.weightGeneration == 'uniform':
            self.Wres = torch.as_tensor(torch.rand(self.n_reservoir, self.n_reservoir, dtype = _DTYPE, device = self.device) - 0.5, dtype = _DTYPE, device = self.device)
        elif self.weightGeneration == 'normal':
            self.Wres = torch.as_tensor(torch.randn(self.n_reservoir, self.n_reservoir, dtype = _DTYPE, device = self.device) , dtype = _DTYPE, device = self.device)
        else:
            self.Wres = torch.as_tensor(torch.rand(self.n_reservoir, self.n_reservoir, dtype = _DTYPE, device = self.device) - 0.5, dtype = _DTYPE, device = self.device)
 
        #prevent all-zero eigenvals 
        if self.reservoirDensity >= 1e-6 and self.spectralRadius != 0.0:

            #adjust reservoir density
            if self.use_watts_strogatz_reservoir:
                ws_k = int(self.reservoirDensity*self.n_reservoir)
                _mask = torch.tensor(~nx.to_numpy_array(nx.watts_strogatz_graph(self.n_reservoir,ws_k,self.ws_p)).astype(bool),dtype = bool)
            else:
                _mask = torch.rand(self.n_reservoir, self.n_reservoir, dtype = _DTYPE, device = self.device) > self.reservoirDensity
        
            self.Wres[_mask] = 0.0
            _eig_norm = torch.abs(torch.linalg.eigvals(self.Wres))
            self.Wres *= self.spectralRadius / torch.max(_eig_norm)
        else:
            self.Wres = torch.zeros((self.n_reservoir, self.n_reservoir), dtype = _DTYPE, device = self.device)

#--------------------------------------------------------------------------
    def createWeightMatrices(self):
        '''
        Reservoir initialization. The (fixed) weight matrices Win and Wres are created.
        '''
        self.createInputMatrix()
        self.createReservoirMatrix()

#--------------------------------------------------------------------------
    def calculateLinearNetworkTransmissions(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ''' 
        Computes input and state contributions to the activation function.
        
        INPUT:
            u - current reservoir input
            x - last reservoir state
        RETURN: Win@u + Wres@[bias;x]
            
        '''
        
        uu = u.reshape(self.n_input, 1)

        if self.inputScaling is None:
            self.inputScaling = 1
        
        if np.isscalar(self.inputScaling):
            _extendedinputScaling = torch.ones(self.n_input, device = self.device, dtype = _DTYPE) * self.inputScaling 
        elif len(self.inputScaling) != self.n_input:
            logging.critical(' inputScaling dimension ({0}) does not match input dimension ({1})\n'.format(len(self.inputScaling), self.n_input))
            raise RuntimeError
        else:
            _extendedinputScaling = self.inputScaling

        _extendedinputScaling = torch.vstack((torch.tensor(1, device = self.device, dtype = _DTYPE), _extendedinputScaling.reshape(-1,1))).flatten()

        return torch.matmul(self.Win * _extendedinputScaling, torch.vstack((torch.tensor(self.bias_in, device = self.device, dtype = _DTYPE), uu))) + torch.matmul(self.Wres, x)

#--------------------------------------------------------------------------
    def update(self, u: torch.Tensor, x: torch.Tensor):
        '''
        Single update step of the reservoir state.

        INPUT:
            u - current reservoir input
            x - last reservoir state (will be updated here)
        '''

        transmission = self.calculateLinearNetworkTransmissions(u, x)
        x *= (1.0 - self.leakingRate)

        x += self.leakingRate * (torch.tanh( transmission 
                                             + self.noiseLevel_in  * (torch.rand((self.n_reservoir,1), device = self.device, dtype = _DTYPE) - 0.5)
                                           ) + self.noiseLevel_out * (torch.rand((self.n_reservoir,1), device = self.device, dtype = _DTYPE) - 0.5)
                                )

#--------------------------------------------------------------------------
    def propagate(self, u: torch.Tensor, x: torch.Tensor = None, transientTime: int = 0) -> torch.Tensor:
        ''' 
        Propagates the reservoir state x according to the reservoir dynamics. The number of time iterations is assumed from the 
        specified input u. The transien time specifies how many iterations are discarded for the reservoir washout. 
        These transient reservoir states will not be saved.

        INPUT:
            u            - sequence of reservoir inputs
            x            - starting reservoir state
            transienTime - reservoir washout length

        RETURN:
            X - reservoir state matrix
        '''
        #TO DO: dirty fix for logger in prediction phase
        if transientTime != 0:
            logging.debug('Propagating states')

        assert u is not None,'No reservoir input for propagation has been provided!\n'

        if u.dtype is not _DTYPE:
            u = torch.as_tensor(u, device = self.device, dtype = _DTYPE)

        #default intial reservoir state.
        if x is None:
            x = torch.zeros((self.n_reservoir, 1), device = self.device, dtype = _DTYPE)
        else:
            x = x.reshape(self.n_reservoir,1)

        input_timesteps = u.shape[0]

        # state matrix
        X = torch.zeros((self.xrows, input_timesteps - transientTime), device = self.device, dtype = _DTYPE)
        
        for t in range(input_timesteps):
            self.update(u[t], x = x)
            
            if t >= transientTime:
                
                if self.extendedStateStyle == _EXTENDED_STATE_STYLES[0]:
                    X[:, t - transientTime] = torch.vstack(
                        (torch.tensor(self.bias_out, device = self.device, dtype = _DTYPE), self.outputInputScaling * u[t].reshape(self.n_input,1), x)
                    )[:, 0]
                else:
                    X[:, t - transientTime] = torch.vstack(
                        (torch.tensor(self.bias_out, device = self.device, dtype = _DTYPE), x, x**2)
                    )[:, 0]
        return X
#--------------------------------------------------------------------------
    def verifyReservoirConvergence(self, u: torch.Tensor, transientTimeCalculationEpsilon: float = 1e-3, transientTimeCalculationLength: int = 20 ):
        ''' Computes the convergence of two independent states: -1 and 1, when encountering the data u_train.
            INPUT:
                u - training input data (with which the reservoir is beeing forced)
                transientTimeCalculationEpsilon - proximity distance of the two initially independent states
                transientTimeCalculationLength  - proximity length of the two initially independent states

            RETURN:
                transientTime - no. time steps needed for two states -1 and 1, given the current reservoir and input u, to converge to a 'similar state'.
        '''
        input_timesteps = u.shape[0]

        x_init    =  torch.empty((2,self.n_reservoir, 1), device = self.device, dtype = _DTYPE)
        x_init[0] =  torch.ones((self.n_reservoir, 1), device = self.device, dtype = _DTYPE)
        x_init[1] = -torch.ones((self.n_reservoir, 1), device = self.device, dtype = _DTYPE)
        

        steps = 0
        for it in range(input_timesteps):
            
            #peak to peak distance
            ptp = x_init.max(dim = 0)[0] - x_init.min(dim = 0)[0]

            #states are close
            if torch.max(ptp) < transientTimeCalculationEpsilon:
                if steps >= transientTimeCalculationLength:
                    return it - transientTimeCalculationLength
                else: 
                    steps +=1
            
            #states are not close
            else:
                steps = 0

            #update reservoir states
            for ii in range(x_init.shape[0]):
                self.update(u[it], x_init[ii])
                
        logging.error('Reservoir states did not converge.')
#--------------------------------------------------------------------------
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        '''
        Reservoir output weights Wout is computed by L2 penalized linear regression.

        INPUT:
            X - reservoir state matrix
            y - target training output
        '''

        logging.debug('Fitting output matrix')
        assert X.shape[1] == y.shape[0], "Time dimension of X ({0}) does not match time dimension of y ({1}).\nDid you forget to exclude the transientTime of y? ".format(X.shape[1], y.shape[0])
        
        I = torch.eye(self.xrows, device = self.device)
        I[0,0] = 0  #do not include bias term in regularization, see Lukosevicius et al. Cogn. Comp. (2021) p.2

        self.Wout = torch.matmul(torch.matmul(y.T,X.T), torch.inverse(X@X.T + self.regressionParameter*I))

#--------------------------------------------------------------------------
    def fetch_state(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Fetches reservoir state from extended reserovir state, depending on the extended reservoir state layout.

        INPUT:
            X - extended reservoir state

        RETURN:
            reservoir state

        '''
        if self.extendedStateStyle == _EXTENDED_STATE_STYLES[0]:
            return X[int(1+self.n_input):].reshape(self.n_reservoir,1)
        else:
            return X[1:int(self.n_reservoir+1)].reshape(self.n_reservoir,1)

#--------------------------------------------------------------------------
    def predict(self, X: torch.Tensor, testingLength: int, init_input:torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Used in mode = auto. 

        Autonomous prediction mode of the reservoir. Starting from the last state of the training state matrix X and specified
        initial input init_input. If init_input not specified, rely on self.pred_init_input. The reservoir feeds its last output back to the input layer.

        INPUT:
            X             - state matrix (from which the last state will be taken as starting point) 
            testingLength - number of iterations of the prediction phase
            init_input    - initial input to the ESN from which autonom. prediction will start. 

        RETURN:
            y_pred - reservoir output (predictions)
            X_pred - reservoir state matrix (prediction phase)
        '''
        
        logging.debug('Predicting output')

        if init_input is None:
            if self.pred_init_input is None:
                logging.error('Error in predict: Initial prediction input is not defined! Returning default values for (y_pred, X_pred).')
                return torch.zeros((testingLength, self.n_output)), torch.zeros((self.xrows, testingLength))

        y_pred = torch.zeros([self.n_output,testingLength], device = self.device, dtype = _DTYPE) 
        X_pred = torch.zeros([self.xrows, testingLength], device = self.device, dtype = _DTYPE)
        pred_input = self.pred_init_input
        x = X[:,-1].reshape(self.xrows,)

        for it in range(testingLength):
            x_in = self.fetch_state(x)
            x = self.propagate(pred_input.reshape(1,self.n_input),x = x_in)      #state at time it
            
            pred_output = self.Wout@x                               #reservoir output at time it    

            y_pred[:,it] = pred_output.reshape(self.n_output,)
            X_pred[:,it] = x.reshape(self.xrows,)
            
            pred_input = pred_output                             #new input is the current output


        return y_pred.T, X_pred

#--------------------------------------------------------------------------
    def teacherforce(self, X: torch.Tensor, testingLength: int, u: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Use in mode = teacher.

        Teacher forcing mode of the reservoir. Starting from the last state of the training state matrix X. Inputs are given by self.u_test.
        The reservoir receives an external input u_test[it] each time step. 

        INPUT:
            X             - state matrix (from which the last state will be taken as starting point) 
            testingLength - number of iterations of the prediction phase
            u             - teacher signal (if None, self.u_test is used)

        RETURN:
            y_pred - reservoir output (predictions)
            X_pred - reservoir state matrix (prediction phase)
        '''
        
        logging.debug('Predicting output')
            
        y_pred = torch.zeros((self.n_output,testingLength), device = self.device, dtype = _DTYPE) 
        X_pred = torch.zeros((self.xrows, testingLength), device = self.device, dtype = _DTYPE)
        x = X[:,-1].reshape(self.xrows,)

        if u is None:
            u = self.u_test

        #compute reservoir states
        for it in range(testingLength):
            x_in = self.fetch_state(x)
            x = self.propagate(u[it].reshape(1,self.n_input),x = x_in)      #state at time it

            X_pred[:,it] = x.reshape(self.xrows,)

        #compute reservoir outputs
        y_pred = self.Wout@X_pred                                                     
    
        return y_pred.T, X_pred

#--------------------------------------------------------------------------
    def semiteacherforce(self, X: torch.Tensor, testingLength: int, index_list_auto: list, index_list_teacher: list, u_teacher: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Use in mode: semi-teacher.

        Semi-teacher forcing mode of the reservoir. Starting from the last state of the training state matrix X. 
        Inputs are partially given by self.u_test and autonomous predictions.
        The reservoir receives an external input u_test[it] each time step.
        For now restricted to cases, where n_input = n_output.

        INPUT:
            X                  - state matrix (from which the last state will be taken as starting point) 
            testingLength      - number of iterations of the prediction phase
            index_list_auto    - indices of the modes which are passed back as new input
            index_list_teacher - indices of the modes which are supplied by a teacher signal

        RETURN:
            y_pred - reservoir output (predictions)
            X_pred - reservoir state matrix (prediction phase)
        '''
        
        logging.debug('Predicting output')

        y_pred = torch.zeros([self.n_output,testingLength], device = self.device, dtype = _DTYPE) 
        X_pred = torch.zeros([self.xrows, testingLength], device = self.device, dtype = _DTYPE)
        x = X[:,-1].reshape(self.xrows,)


        def construct_input(index_list_auto, index_list_teacher, u_auto, u_teacher):
            '''Construct initial input from autonomous signal and teacher signal.'''

            u = torch.zeros(self.n_input)

            for aa in range(len(index_list_auto)):
                index = index_list_auto[aa]
                u[index] = u_auto[aa]
            for tt in range(len(index_list_teacher)):
                index = index_list_teacher[tt]
                u[index] = u_teacher[tt]

            return u

        pred_input = self.pred_init_input

        for it in range(testingLength):

            u_auto = pred_input[index_list_auto]     # autonomous part of prediction: used as part of new input
            u_test = self.u_test[it]                 # teacher part of prediction: used as part of new input
            u_merged = construct_input(index_list_auto, index_list_teacher, u_auto, u_test)
        
            x_in = self.fetch_state(x)
            x = self.propagate(u_merged.reshape(1,self.n_input),x = x_in)      #state at time it

            pred_output = self.Wout@x  
            y_pred[:,it] = pred_output.reshape(self.n_output,)
            X_pred[:,it] = x.reshape(self.xrows,)

            pred_input = pred_output


        return y_pred.T, X_pred


#--------------------------------------------------------------------------     
#--------------------------------------------------------------------------
#  HYPERPARAMETER SETTERS
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
    
    #FH 01/02/2021: Added SetTrainingData & SetTestingData
    #--------------------------------------------------------------------------
    def SetTrainingData(self, u_train: torch.Tensor, y_train: torch.Tensor):
        
        assert u_train.shape[0] == y_train.shape[0],'Training input dimension ({0}) does not match training output time dimension ({1}).\n'.format(u_train.shape[0], y_train.shape[0])
        assert u_train.shape[1] == self.n_input,'Training input dimension ({0}) does not match ESN n_input ({1}).\n'.format(u_train.shape[1], self.n_input)
        assert y_train.shape[1] == self.n_output, 'Training output dimension ({0}) does not match ESN n_output ({1}).\n'.format(u_train.shape[1], self.n_output)
        
        self.u_train = u_train
        self.y_train = y_train
    
    #--------------------------------------------------------------------------
    def SetTestingData(self, y_test: torch.Tensor, pred_init_input: torch.Tensor = None, u_test: torch.Tensor = None):       
        
        assert y_test.shape[1] == self.n_output,'Testing output dimension ({0}) does not match ESN n_output ({1}).\n'.format(y_test.shape[1], self.n_output)
    
        self.y_test = y_test
        self.u_test = u_test

        if self.mode == 'auto':

            if self.y_train is None:
                assert pred_init_input is not None, 'Initial testing input and self.y_train not specified.\n'

            if self.y_train is not None and pred_init_input is None:
                #Initial input is last training input. Then first prediction aligns with the first entry of y_test
                logging.debug('Initial testing input not specified. Using last target training output.')
                self.pred_init_input = self.y_train[-1:,:]    #initial input the trained ESN receives for the beginning of the testing phase
           
            elif pred_init_input is not None:
                self.pred_init_input = pred_init_input

        elif self.mode == 'teacher':
            assert u_test is not None, 'Teacher mode requires non empty u_test!\n'

        if self.mode == 'semi-teacher':

            if pred_init_input is not None:
                self.pred_init_input = pred_init_input

                #TO DO: pred_init_input has len n_input. The teacher part of it is not used (see self.semiteacherforce).

    #--------------------------------------------------------------------------
    # FH 30.03.2022: Added Validation Datset (auto mode!)
    def SetValidationData(self, y_val: torch.Tensor, u_val: torch.Tensor = None, val_init_input: torch.Tensor = None,):   

        assert y_val.shape[1] == self.n_output,'Validation output dimension ({0}) does not match ESN n_output ({1}).\n'.format(y_val.shape[1], self.n_output)
    
        self.u_val = u_val
        self.y_val = y_val

        if self.mode == 'auto':

            if self.y_test is not None and val_init_input is None:
                #Initial input is last testing input. Then first prediction aligns with the first entry of y_val
                logging.debug('Initial validation input not specified. Using last target test output.')
                self.val_init_input = self.y_test[-1:,:]    #initial input the trained ESN receives for the beginning of the validation phase

            elif val_init_input is not None:
                self.val_init_input = val_init_input

    #--------------------------------------------------------------------------
    #FH 14/11/2021: Added logging 
    def SetLoggingLevel(self, logging_level):
    
        logging.warn(f'Setting logging level to {logging_level}')
        logging.getLogger().setLevel(logging_level)
    
    #--------------------------------------------------------------------------
    def SetID(self, esn_id):
    
        logging.debug(f'Setting reservoir ID to {esn_id}')
        self.id = esn_id

    #--------------------------------------------------------------------------
    def SetNInputOutput(self,n_input: int, n_output: int = None, study_dict: dict = {}):
        
        logging.debug(f'Setting n_input {n_input}')

        self.n_input = n_input
        if n_output is not None:
            logging.debug(f'Setting n_output {n_output}')
            self.n_output = n_output
        else:
            self.n_output = self.n_input 
        
        #adjust xrows as according to changed n_input
        if self.extendedStateStyle == _EXNTENDED_STATE_STYLES[0]:
            self.xrows = int(1+self.n_reservoir+self.n_input)          
        
        study_dict['n_input'] = n_input
     #--------------------------------------------------------------------------
    def SetNReservoir(self,n_reservoir: int, study_dict: dict = {}):

        logging.debug(f'Setting n_reservoir {n_reservoir}')
        
        self.n_reservoir = n_reservoir

        #adjust xrows as according to changed n_reservoir
        if self.extendedStateStyle == _EXNTENDED_STATE_STYLES[0]:
            self.xrows = int(1+self.n_reservoir+self.n_input)           
        else:
            self.xrows = int(1+2*self.n_reservoir)

        study_dict['n_reservoir'] = n_reservoir

    #--------------------------------------------------------------------------
    def SetSpectralRadius(self, spectralRadius: float, study_dict: dict = {}):
        
        logging.debug(f'Setting spectralRadius {spectralRadius}')
        
        self.spectralRadius = spectralRadius
        study_dict['spectralRadius'] = spectralRadius
    #--------------------------------------------------------------------------
    def SetReservoirDensity(self, reservoirDensity: float, study_dict: dict = {}):
        
        logging.debug(f'Setting reservoirDensity {reservoirDensity}')
        self.reservoirDensity = reservoirDensity
        study_dict['reservoirDensity'] = reservoirDensity
    #--------------------------------------------------------------------------
    def SetLeakingRate(self, leakingRate: float, study_dict: dict = {}):
       
        logging.debug(f'Setting leakingRate {leakingRate}')
        self.leakingRate = leakingRate
        study_dict['leakingRate'] = leakingRate
    #--------------------------------------------------------------------------
    def SetRegressionParameter(self, regressionParameter: float, study_dict: dict = {}):
        
        logging.debug(f'Setting regressionParameter {regressionParameter}')
        self.regressionParameter = regressionParameter
        study_dict['regressionParameter'] = regressionParameter
    #--------------------------------------------------------------------------
    def SetBiasIn(self, bias_in: float, study_dict: dict = {}):

        logging.debug(f'Setting bias_in {bias_in}')
        self.bias_in = bias_in
        study_dict['bias_in'] = bias_in
    #--------------------------------------------------------------------------
    def SetBiasOut(self, bias_out: float, study_dict: dict = {}):

        logging.debug(f'Setting bias_out {bias_out}')
        self.bias_out = bias_out
        study_dict['bias_out'] = bias_out
    #--------------------------------------------------------------------------
    def SetOutputInputScaling(self, outputInputScaling: float, study_dict: dict = {}):
        
        logging.debug(f'Setting outputInputScaling {outputInputScaling}')
        self.outputInputScaling = outputInputScaling
        study_dict['outputInputScaling'] = outputInputScaling
    #--------------------------------------------------------------------------
    def SetInputScaling(self, inputScaling: float, study_dict: dict = {}):
        
        logging.debug(f'Setting inputScaling {inputScaling}')
        self.inputScaling = inputScaling
        study_dict['inputScaling'] = inputScaling
    #--------------------------------------------------------------------------
    def SetInputDensity(self, inputDensity: float, study_dict: dict = {}):
        
        logging.debug(f'Setting inputDensity {inputDensity}')
        self.inputDensity = inputDensity
        study_dict['inputDensity'] = inputDensity
    #--------------------------------------------------------------------------
    def SetNoiseLevelIn(self, noiseLevel_in: float, study_dict: dict = {}):
        
        logging.debug(f'Setting noiseLevel_in {noiseLevel_in}')
        self.noiseLevel_in = noiseLevel_in
        study_dict["noiseLevel_in"] = noiseLevel_in
    #--------------------------------------------------------------------------
    def SetNoiseLevelOut(self, noiseLevel_out: float, study_dict: dict = {}):
        
        logging.debug(f'Setting noiseLevel_out {noiseLeve_out}')
        self.noiseLevel_out = noiseLevel_out
        study_dict["noiseLevel_out"] = noiseLevel_out

    #--------------------------------------------------------------------------
    def SetTransientTime(self, transientTime: int, study_dict: dict = {}):
        
        logging.debug(f'Setting transientTime {transientTime}')
        self.transientTime = transientTime
        study_dict["transientTime"] = transientTime

    #--------------------------------------------------------------------------
    def SetDataScaling(self, dataScaling: float, study_dict: dict = {}):
        
        logging.debug(f'Setting dataScaling {dataScaling}')
        self.u_train *=dataScaling
        self.y_train *=dataScaling
        if self.u_test is not None:
            self.u_test  *=dataScaling
        self.y_test  *=dataScaling
        study_dict['dataScaling'] = dataScaling
    #--------------------------------------------------------------------------
    #FH 21/03/2021: added Seed Setter method
    def SetRandomSeed(self, randomSeed: int):
        
        logging.debug(f'Setting randomSeed {randomSeed}')
        self.randomSeed = randomSeed

        # TO DO: 
        # I guess this has no effect. The seed must be set in the script where the computations are done.
        if self.randomSeed is not None:
            torch.manual_seed(self.randomSeed)

    #--------------------------------------------------------------------------
    #FH 22/03/2021:  added automatic Setter method 
    def SetStudyParameters(self, config_istudy: Union[np.ndarray, torch.Tensor], study_tuple: tuple) -> dict:
        '''
        Sets the hyperparameters, specified in study_tuple, to values specified in config_istudy.
        Returns the changed setting in form of a dictionary. 

        Hyperparameters which are not included with their setter method, can not be studies by the grid/random search routine!
            
        INPUT:
            config_istudy  - array specifing values of the hyperparameters to which the ESN should be set
            study_tuple    - tuple of strings, specifying which hyperparameters are studied (order must be same as config_istudy)
        
        RETURN:
            study_dict - dict specifying the new hyperparameter setting of the ESN 
        '''
        
        assert len(study_tuple) > 0,'study_tuple is empty. Did you forget to specify the study HP?'

        study_dict  ={}

        for iparam, parameter in enumerate(study_tuple):

            if parameter == "n_reservoir":
                self.SetNReservoir(config_istudy[iparam], study_dict)

            elif parameter == "spectralRadius":
                self.SetSpectralRadius(config_istudy[iparam], study_dict)

            elif parameter == "reservoirDensity":
                self.SetReservoirDensity(config_istudy[iparam], study_dict)

            elif parameter == "leakingRate":
                self.SetLeakingRate(config_istudy[iparam], study_dict)

            elif parameter == "regressionParameter":
                self.SetRegressionParameter(config_istudy[iparam], study_dict)
            
            elif parameter == "bias_in":
                self.SetBiasIn(config_istudy[iparam], study_dict)
            
            elif parameter == "bias_out":
                self.SetBiasOut(config_istudy[iparam], study_dict)

            elif parameter == "outputInputScaling":
                self.SetOutputInputScaling(config_istudy[iparam], study_dict)

            elif parameter == "inputScaling":
                self.SetInputScaling(config_istudy[iparam], study_dict)
            
            elif parameter == "inputDensity":
                self.SetInputDensity(config_istudy[iparam], study_dict)

            elif parameter == "dataScaling":
                self.SetDataScaling(config_istudy[iparam], study_dict)

            elif parameter == "inputDensity":
                self.SetInputDensity(config_istudy[iparam], study_dict)
            
            elif parameter == "noiseLevel_in":
                self.SetNoiseLevelIn(config_istudy[iparam], study_dict)
            
            elif parameter == "noiseLevel_out":
                self.SetNoiseLevelOut(config_istudy[iparam], study_dict)
            
            elif parameter == "transientTime":
                self.SetTransientTime(config_istudy[iparam], study_dict)

            else:
                logging.critical('Specified parameter {0} unknown.'.format(parameter))
                raise KeyError
                
        return study_dict

#--------------------------------------------------------------------------     
#--------------------------------------------------------------------------
#  MISC. IMPLEMENTATIONS
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
    def toTorch(self):
        '''
        Moves the ESNParams object to python or torch types. Torch tensors are automatically shifted to device.
        This will stop numpy vs. torch clashes from happening.
        '''
        
        logging.debug('Moving class arrays to {0} with dtype {1}'.format(self.device, _DTYPE))

        allmembers = [m for m in dir(self) if m[:2] != '__' and m[-2:] != '__']
        ifaddmember = [not callable(getattr(self, name)) for name in allmembers ]
        
        #Cast from numpy to python/ torch types
        for ii,member in enumerate(allmembers):
            if ifaddmember[ii]:
                value = getattr(self,member)
                
                if isinstance(value, np.float32) or isinstance(value, np.float64):         #np.float   -> float
                    new_value = float(value)      

                elif isinstance(value, np.int32) or isinstance(value, np.int64):           #np.int   -> int
                    new_value = int(value) 
                
                elif isinstance(value, np.ndarray):         #array    -> tensor
                    new_value = torch.as_tensor(value, device = self.device, dtype = _DTYPE)
                
                elif isinstance(value, torch.Tensor):
                    if value.dtype != _DTYPE or value.device != self.device:
                        new_value = torch.as_tensor(value, dtype= _DTYPE, device = self.device)
                    else:
                        new_value = value
                else:
                    new_value = value
                    
                setattr(self, member, new_value)       
            
#--------------------------------------------------------------------------
    def setDevice(self, device: str):
        ''' Set globael ESNParams device (cpu or cuda)'''
        
        logging.debug('Setting device: {0}'.format(device))
        self.device = device
        
#---------------------------------------------------------------------------
    def save(self,filepath: str, f = None):
        ''' Saves the reservoir parameters, training and validation/testing data from ESNParams class object into a hdf5 file
    
        INPUT:
            filepath   - path to which the hdf5 file of the ESN study is saved to
        '''
        
        toClose = False
        logging.warn('Saving ESN parameters to Hdf5 file {0}'.format(filepath))

        if f is None:
            f = h5py.File(filepath, 'w')
            toClose = True
        
        G_hp = f.create_group('Hyperparameters')
        
        if self.randomSeed is None:
            self.randomSeed = False

        #Model
        G_hp.attrs['data_timesteps'] = self.data_timesteps
        G_hp.attrs['trainingLength'] = self.trainingLength
        G_hp.attrs['testingLength'] = self.testingLength
        G_hp.attrs['validationLength'] = self.validationLength
        G_hp.attrs['n_input'] = self.n_input
        G_hp.attrs['n_output']= self.n_output
        G_hp.attrs['n_reservoir'] = self.n_reservoir
        G_hp.attrs['leakingRate']= self.leakingRate
        G_hp.attrs['spectralRadius']= self.spectralRadius
        G_hp.attrs['regressionParameter'] = self.regressionParameter
        G_hp.attrs['reservoirDensity'] = self.reservoirDensity
        G_hp.attrs['noiseLevel_in'] = self.noiseLevel_in
        G_hp.attrs['noiseLevel_out'] = self.noiseLevel_out
        G_hp.attrs['inputScaling'] = self.inputScaling               
        G_hp.attrs['inputDensity'] = self.inputDensity
        G_hp.attrs['weightGeneration'] = self.weightGeneration  
        G_hp.attrs['extendedStateStyle'] = self.extendedStateStyle  
        G_hp.attrs['bias_in'] = self.bias_in
        G_hp.attrs['bias_out'] = self.bias_out
        G_hp.attrs['outputInputScaling'] = self.outputInputScaling
        G_hp.attrs['esn_start'] = self.esn_start
        G_hp.attrs['esn_end'] = self.esn_end
        G_hp.attrs['randomSeed'] = self.randomSeed
        G_hp.attrs['mode'] = self.mode
        G_hp.attrs['transientTime'] = self.transientTime
        G_hp.attrs['ws_p']  = self.ws_p
        G_hp.attrs['use_watts_strogatz_reservoir']= float(self.use_watts_strogatz_reservoir)
        
        
        G_data = f.create_group('Data')
        #Datasets
        if self.u_train is not None:
            G_data.create_dataset('u_train',   data = self.u_train, compression = 'gzip', compression_opts = 9)
        if self.y_train is not None:
            G_data.create_dataset('y_train',   data = self.y_train, compression = 'gzip', compression_opts = 9)
        if self.u_test is not None:
            G_data.create_dataset('u_test',   data = self.u_test, compression = 'gzip', compression_opts = 9)
        if self.y_test is not None:
            G_data.create_dataset('y_test',   data = self.y_test, compression = 'gzip', compression_opts = 9)
        if self.u_val is not None:
            G_data.create_dataset('u_val',   data = self.u_val, compression = 'gzip', compression_opts = 9)
        if self.y_val is not None:
            G_data.create_dataset('y_val',   data = self.y_val, compression = 'gzip', compression_opts = 9)

        if toClose:
            f.close()

#--------------------------------------------------------------------------     
#--------------------------------------------------------------------------
#  DUNDER METHODS
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
    def __repr__(self):

        repr = "--------------------------\n"
        repr += "ESN Setting:\n"
        repr += "--------------------------\n"
        if self.randomSeed is not None:
            repr += "randomSeed = {0}\n".format(self.randomSeed)
        else:
            repr += "randomSeed = None\n"
            
        repr += "n_input"+ " = {0:.0f}\n".format(self.n_input)
        repr += "n_output"+ " = {0:.0f}\n".format(self.n_output)
        
        repr += "n_reservoir" + " = {0:.0f}\n".format(self.n_reservoir)
        repr += "reservoir density = {0:.3f}\n".format(self.reservoirDensity)
        repr += "spectral radius = " + "{0:.3f}\n".format(self.spectralRadius)
        repr += "leaking rate = "+"{0:.3f}\n".format(self.leakingRate)
        repr += "regression parameter = "+"{0:.2e}\n".format(self.regressionParameter)
        
        
        repr += "training length = " + "{0:.0f}\n".format(self.trainingLength)
        repr += "testing length = " + "{0:.0f}\n".format(self.testingLength)
        repr += "esn start = " + "{0:.0f}\n".format(self.esn_start)
        repr += "esn end = " + "{0:.0f}\n".format(self.esn_end)
        repr += "data length = " + "{0:.0f}\n".format(self.data_timesteps)

        repr += "input bias = " + "{0:.0f}\n".format(self.bias_in)
        repr += "output bias = " + "{0:.0f}\n".format(self.bias_out)

        if np.isscalar(self.inputScaling):
            repr += "input scaling = " + "{0:.0f}\n".format(self.inputScaling)
        else:
            repr += "input scaling is array\n"

        repr += "output input scaling = " + "{0:.0f}\n".format(self.outputInputScaling)
        
        repr += "noise level inside activation function = " + "{0:.2e}\n".format(self.noiseLevel_in)
        repr += "noise level outside activation function = " + "{0:.2e}\n".format(self.noiseLevel_out)

        repr += "transientTime = " + "{0}".format(self.transientTime)
        
        return repr

#--------------------------------------------------------------------------     
#--------------------------------------------------------------------------
#  CLASS METHODS
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
    @classmethod
    def read(cls, filepath: str):
        '''
        Creates a ESNParams object from a saved hdf5 ESN file.

        INPUT:
            filepath - path to the saved ESN 

        RETURN:
            esn - ESNParams object deduced from saved ESN
        '''

        try:
            with h5py.File(filepath,'r') as f:
                pass
        except:
            logging.debug('Error: file {0} not found.'.format(filepath))


        with h5py.File(filepath,'r') as f:

            G_hp = f.get('Hyperparameters')
            trainingLength = G_hp.attrs['trainingLength']
            testingLength  = G_hp.attrs['testingLength']
            validationLength  = G_hp.attrs['validationLength']
        
            data_timesteps = G_hp.attrs['data_timesteps']
            n_input        = G_hp.attrs['n_input']
            n_output       = G_hp.attrs['n_output']
            n_reservoir    = G_hp.attrs['n_reservoir']
            leakingRate    = G_hp.attrs['leakingRate']
            spectralRadius  = G_hp.attrs['spectralRadius']
            regressionParameter    = G_hp.attrs['regressionParameter']
            reservoirDensity        = G_hp.attrs['reservoirDensity']
            noiseLevel_in              = G_hp.attrs['noiseLevel_in']
            noiseLevel_out              = G_hp.attrs['noiseLevel_out']
            
            inputScaling            = G_hp.attrs['inputScaling']
            inputDensity            = G_hp.attrs['inputDensity']
            randomSeed              = G_hp.attrs['randomSeed']
            weightGeneration        = G_hp.attrs['weightGeneration']
            extendedStateStyle      = G_hp.attrs['extendedStateStyle']
            
            bias_in                 = G_hp.attrs['bias_in']
            bias_out                = G_hp.attrs['bias_out']
            outputInputScaling                 = G_hp.attrs['outputInputScaling']
            esn_start    = G_hp.attrs['esn_start']
            esn_end      = G_hp.attrs['esn_end']
            mode = G_hp.attrs['mode']
            transientTime = G_hp.attrs['transientTime']
            use_watts_strogatz_reservoir = bool(G_hp.attrs['use_watts_strogatz_reservoir'])
            ws_p = G_hp.attrs['ws_p']

            
            G_data = f.get('Data')
            if 'y_train' in G_data:
                y_train = np.array(G_data.get('y_train'))
            else:
                y_train = None

            if 'u_train' in G_data:
                u_train = np.array(G_data.get('u_train'))
            else:
                u_train = None
            
            if 'y_test' in G_data:
                y_test = np.array(G_data.get('y_test'))
            else:
                y_test = None

            if 'u_test' in G_data:
                u_test = np.array(G_data.get('u_test'))
            else:
                u_test = None

            if 'y_val' in G_data:
                y_val = np.array(G_data.get('y_val'))
            else:
                y_val = None

            if 'u_val' in G_data:
                u_val = np.array(G_data.get('u_val'))
            else:
                u_val = None





        esn = ESN(  randomSeed = randomSeed,
                    esn_start = esn_start,
                    esn_end = esn_end,
                    trainingLength = trainingLength,
                    testingLength = testingLength,
                    validationLength = validationLength,
                    data_timesteps = data_timesteps,
                    n_input = n_input,
                    n_output =  n_output,
                    n_reservoir = n_reservoir,
                    leakingRate = leakingRate,
                    spectralRadius = spectralRadius,
                    reservoirDensity = reservoirDensity,
                    regressionParameter = regressionParameter,
                    bias_in = bias_in,
                    bias_out = bias_out,
                    outputInputScaling = outputInputScaling,
                    inputScaling = inputScaling,
                    inputDensity = inputDensity,
                    noiseLevel_in = noiseLevel_in,
                    noiseLevel_out = noiseLevel_out,
                    mode = mode,
                    weightGeneration = weightGeneration,
                    transientTime = transientTime,
                    extendedStateStyle = extendedStateStyle,
                    use_watts_strogatz_reservoir=use_watts_strogatz_reservoir,
                    ws_p=ws_p,
                    verbose = False
                )
            
        if u_train is not None and y_train is not None:
            esn.SetTrainingData(u_train = u_train, y_train = y_train)
        if y_test is not None:
            esn.SetTestingData(y_test = y_test, u_test = u_test)
        if u_val is not None and y_val is not None:
            esn.SetValidationData(y_val=y_val, u_val=u_val)     
    

        esn.setDevice(_DEVICE)

        return esn

#--------------------------------------------------------------------------
    @classmethod
    def hyperparameter_intervals(cls):
        '''
            Returns valid invtervals for hyperparameters. Might be used for random searches in these intervals.

            RETURN:
                HP_range_dict - hyperparameter interval dict
        '''

        HP_range_dict = {}

        HP_range_dict['n_reservoir'] = (1e2,5e3)
        HP_range_dict['leakingRate'] = (1e-2,1e0)
        HP_range_dict['spectralRadius'] = (1e-3,2e0)
        HP_range_dict['reservoirDensity'] = (1e-2,1e0)
        HP_range_dict['dataScaling'] = (1e-2,5e0)
        HP_range_dict['regressionParameter'] = (1e-6,1e1)
        HP_range_dict['bias_in'] = (1e-3,1e0)
        HP_range_dict['bias_out'] = (1e-3,1e0)
        HP_range_dict['outputInputScaling'] = (1e-3,1e0)
        HP_range_dict['inputScaling'] = (1e-2,1e1)
        HP_range_dict['inputDensity'] = (1e-2,1e0)
        HP_range_dict['noiseLevel_in'] = (1e-6,1e0)
        HP_range_dict['noiseLevel_out'] = (1e-6,1e0)
        
        return HP_range_dict 
 
#--------------------------------------------------------------------------     
#--------------------------------------------------------------------------
#  RESERVOIRS
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
    @classmethod
    def ml4ablReservoir(cls, 
                        data_timesteps: int = 2000, 
                        trainingLength: int = 700, 
                        testingLength: int = 500, 
                        validationLength: int = 500,
                        mode: str = 'auto', 
                        verbose: bool = False):
        '''
            Machine learning for atmospheric boundary layer project. Data should be the first 300 POD time coefficients
            of 2D RBC data with Neumann boundary conditions for the buoyancy.

        '''

        assert mode in _ESN_MODES,'Error: unkown mode {0}. Choices {1}'.format(mode, _ESN_MODES)

        esn_timesteps = trainingLength + testingLength + validationLength
        esn_start = data_timesteps - esn_timesteps
        esn_end = data_timesteps

        esn = cls(  randomSeed = 0,
                    esn_start = esn_start, 
                    esn_end = esn_end,
                    trainingLength =trainingLength, 
                    testingLength = testingLength,
                    validationLength = validationLength,
                    data_timesteps = data_timesteps,
                    n_input = 300,
                    n_output = 300,
                    n_reservoir = 4096,
                    leakingRate = 1.0, 
                    spectralRadius = 0.95,
                    reservoirDensity = 0.84,
                    regressionParameter = 5e-2,
                    bias_in = 1.0,
                    bias_out = 1.0,
                    outputInputScaling = 1.0,
                    inputScaling = 1.0, 
                    inputDensity = 1.0, 
                    noiseLevel_in  = 0.0,
                    noiseLevel_out = 0.0,
                    mode = mode,
                    weightGeneration = "uniform",
                    extendedStateStyle = 'default',
                    transientTime = 50,
                    use_watts_strogatz_reservoir = False,
                    verbose = verbose
                )
            
        return esn

#--------------------------------------------------------------------------
    @classmethod
    def L63Reservoir(cls, 
                     data_timesteps: int = 5000, 
                     trainingLength: int = 2059, 
                     testingLength: int = 1444, 
                     validationLength: int = 1444,
                     mode: str = 'auto', 
                     verbose: bool = False):
        '''
            Lorenz 63' standard reservoir. 
        '''

        assert mode in _ESN_MODES,'Error: unkown mode {0}. Choices {1}'.format(mode, _ESN_MODES)

        esn_timesteps = trainingLength + testingLength
        esn_start = data_timesteps - esn_timesteps
        esn_end = data_timesteps

        esn = cls(  randomSeed = 0,
                    esn_start = esn_start, 
                    esn_end = esn_end,
                    trainingLength =trainingLength, 
                    testingLength = testingLength,
                    validationLength = validationLength,
                    data_timesteps = data_timesteps,
                    n_input = 300,
                    n_output = 300,
                    n_reservoir = 4096,
                    leakingRate = 1.0, 
                    spectralRadius = 0.95,
                    reservoirDensity = 0.84,
                    regressionParameter = 5e-2,
                    bias_in = 1.0,
                    bias_out = 1.0,
                    outputInputScaling = 1.0,
                    inputScaling = 1.0, 
                    inputDensity = 1.0, 
                    noiseLevel_in  = 0.0,
                    noiseLevel_out = 0.0,
                    mode = mode,
                    weightGeneration = "uniform",
                    extendedStateStyle = 'default',
                    transientTime = 44,
                    use_watts_strogatz_reservoir = False,
                    verbose = verbose
                )

        return esn

#--------------------------------------------------------------------------     
#--------------------------------------------------------------------------
#  EXPERIMENTAL
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
    def activation_arg_dist(self, bins_input: torch.Tensor, bins_res: torch.Tensor, X_pred: torch.Tensor = None):
        '''
        Gives information about the distribution of values inside the activation function. Useful for assment of activation saturation.

        INPUT:
            bins_input  - bins to use for computing the distribution of Win@u (inputs)
            bins_res    - bins to use for computing the distribution of Wres@x (reservoir states)
            X_pred      - reservoir state matrix (prediction phase)
        RETURN:
            hist_total  - historam of Win@u + Wres@x
            hist_input  - histogram of Win@u
            hist_res    - histogram of Wres@x
        '''

        #TO DO: adapt range of bins to data, scaling, and spectral radius
        #TO DO: s_pred with nans could happen before
        
        arg_res = self.Wres@X_pred[int(1+self.n_input):,:]
        arg_input = self.Win@torch.cat((torch.ones((1,self.xrows), device = self.device, dtype = _DTYPE),self.Wout ),dim = 0)@X_pred
           
        hist_res, _ = np.histogram(arg_res.flatten(), bins = bins_res)
        hist_input, _  = np.histogram(arg_input.flatten(), bins = bins_input)
        hist_total, _  = np.histogram((arg_res+arg_input).flatten(), bins = bins_input)
        
        return hist_total/len((arg_res + arg_input).flatten()), hist_input/len(arg_input.flatten()), hist_res/len(arg_res.flatten())

#--------------------------------------------------------------------------
