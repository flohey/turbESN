 
## Release 0.1.8.5
------------------
### 29.03.2022
- changed identity matrix in fitting procedure to not include bias terms

### 30.03.2022
- added k fold forward walk validation data preparation in util.py
- added k fold forward walk validation procedure in util.py
- added option to run ESN w. third (validation) dataset (util.py & core.py)
- added function which checks user input data (prev. in RunturbESN)
- changed no. return variables in RunturbESN: now returns mse_train, mse_test, mse_val, y_pred_test, y_val
- fixed problem in assert in RunturbESN in util.py

### 31.03.2022
- adapted SaveStudy & ReadStudy to save mse_val & y_pred_val in util.py
- renamed LaunchThreads to launch_process_RunturbESN in study.py
- renamed LaunchSingleThread to launch_thread_RunturbESN in study.py
- adapted launch_thread_RunturbESN & Callback in study.py
- added launch_process_forward_validate_turbESN in study.py
- added launch_thread_forward_validate_turbESN in study.py
- added validationLength as ESN object value in core.py
- added validationLength to L63Reservoir in core.py
- added cross_validation.py to turbESN

### 01.04.2022
- moved k fold forward walk validation data preparation to cross_validation.py
- added option for single esn_id in CreateHDF5Groups in util.py
- adapted in ESN.save for single esn_id in core.py
- added CrossValidation.save in cross_validation.py
- changed k fold forward walk validation return value to torch.tensors in util.py
- added validationLength to ml4ablReservoir in core.py

## Release 0.1.8.8
------------------
### 04.04.2022
- adapted read method in core.py
- adapted save method in core.py
- implemented read method in cross_validation.py
- slight change to __init__ of CrossValidation in cross_validation.py
- adapted save of CrossValidation in cross_validation.py

## Release 0.0.1.8.9
--------------------
- corrected error in SetTrainingData method in core.py
- added: take val_init_input from self.y_test[-1:,:] in SetValidationData method in core.py
- pred_init_input = self.y_train[-1:,:] (note shape) in SetTestingData in core.py

### 11.04.2022
- corrected mistake in init of Wres when using Watts Strogatz Small World Reservoir in createReservoirMatrix method in core.py

### 02.05.2022
- added manual teacher signal in teacherforce method in core.py (useful for validation for example)
- added validation phase for teacherforcing mode in RunturbESN method in util.py

## Release 0.0.1.8.9.1
--------------------
### 13.05.2022
- added validation data option to PrepareTeacherData in util.py

## Release 0.1.8.9.2
------------------
### 13.05.2022
- corrected error in ReadStudy in util.py
-corrected error in RunTurbESn in util.py

## Release 0.0.1.8.9.4
--------------------
### 18.05.2022
- adapted InitRandomSearchStudyOrder in util.py to uniform & log-uniform grid
- changed HP range in hyperparameter_intervals in core.py

### 23.05.2022
- fixed error in ReadStudy in util.py

### 25.05.2022
- added minmax_scaling method to util.py 
- adapted random search, s.t. one can define the serch intervals oneself

## Release 0.0.1.8.9.5
--------------------
- pypi version is now read from __version__.py by setup.py
- corrected error in SetNReservoir method in core.py


## Release 0.0.1.8.9.6
---------------------
- fixed bug in ReadStudy method in util.py

## Release 0.0.1.8.9.7
---------------------
### 27.06.2022
- slightly modified verifyReservoirConvergence method in core.py

### 01.07.2022
- corrected PyDoc in RunturbESN method in util.py


## Release 0.0.1.9.0.1
---------------------
### 19.08.2022
- `__init__.py`: import `__version__`
- removed small world reservoir matrix initialization via networkx
- check for neuron specific leaking rate in __init__ method in core.py

### 22.08.2022
- fixed logging error in SetTrainingData in core.py
- **implemented feedback weights, where last ESN output is passed to the reservoir**

## Release 0.0.1.9.1.2
---------------------
### 22.08.2022
- fixed error in verifyReservoirConvergence method in core.py

### 24.08.2022
- renamed study_parameters to study_tuple in several methods in util.py

### 02.09.2022
- added _modes.py to root dir (.py files read constants & mode values from here now)
- added hyperparameters.json to root dir
- added get_HP_info method in core.py 
- updated save method in core.py to work with hyperparameters.json
- updated InitRandomSearchStudyOrder method in util.py to work with hyperparameters.json

- changed variable names (stay more consistent in naming convention): 
    - pred_init_input -> test_init_input (core.py)
    - x_fit   -> x_train (util.py)
    - x_pred  -> x_test  (util.py)
- changed launch_thread_RunturbESN, launch_process_RunturbESN in study.py to assess whethe weight matrices Wres,Win,Wfb must be recomputed or can be computed once per random seed
- adapted RunturbESN in util.py accordingly

### 03.09.2022
- added get_size method in core.py


### 05.09.2022
- fixed bug, that provided prediction input would not be used in predidct method in util.py
- added undo_minmax_scaling method in util.py

### 13.09.2022
- added ReadESNOutput, ReadMSE to util.py (separated the ReadStudy method)
- edited ReadStudy method in util.py

### 26.09.2022
- added new fit option: pseudo-inverse

### 27.09.2022
- refined user_study_w_config.py and changed name to run_gs.py
- refined esn_config.yaml


### 24.10.2022
- changed name CreateHDF5Groups to create_hdf5_groups in util.py
- changed name InitRandomStudyOrder to init_random_search in util.py
- added init_grid_search to util.py
- refined esn_config.yaml
- refined run_gs.py

### 03.11.2022
- renamed doRandomSearch parameter in esn_config.yaml
- fixed bug in predict method in core.py
- renamed verifyReservoirConvergence to verify_echo_state_property in core.py
- added verify_echo_state_property to run_gs.py
- added fit_method & mode to hyperparameters.json
- fixed bug in read in core.py

### 04.11.2022
- adapted read method in core.py to read all data from hdf5
- renamed  activation_arg_dist method to compute_activation_arg_distribution in core.py
- adapted compute_activation_arg_distribution method to feedback argument + added option to specify ESN prediction phase
- changed logging state in verify_echo_state_property method from error to warn in core.py
- added plot_activation_arg_distribution to util.py
- use importlib.resources for finding path to hyperparameters.json
- added plot_esn_predictions to util.py


## Release 0.0.1.9.2.0
----------------------
- fixed minor issues in run_gs.py
- changed syntax in verify_echo_state_property method in util.py
- fixed bug in plot_esn_predictions method in util.py

### 14.11.2022
- renamed launch_process_RunturbESN to parallelize_seeds in study.py
- renamed Callback to callback_seeds in study.py
- renamed start_thread_RunturbESN to thread_RunturbESN in study.py
- renamed nstudy to nsettings in util.py, core.py & study.py
- renamed istudy to isetting in util.py, core.py & study.py
- renamed CreateStudyConfigArray to create_study_config_list in util.py
- fixed bug in SaveStudy in util.py, renamed some variables
- added parallelize_settings & callback_setting methods in study.py 
- adapted descriptions and small details in parallelize_seeds & callback_seeds in util.py
- adapted create_hdf5_groups method in util.py to only take ints as args
- adapted run_gs.py, amongst other: choose between parallelization
- new convention: esn.id is only used to keep track of different es instances among different subprocesses, it is only used in study.py!
    - adapted in SaveStudy, ReadStudy, ReadMSE, ReadESNOutput
- renamed SaveStudy, ReadStudy, ReadMSE, ReadESNOutput to snake case equivalents

### 22.11.2022
- added default values to style_dict in plot_esn_predictions & plot_activation_arg_distribution in util.py

### 25.11.2022
- added type hints in thread_RunturbESN in study.py
- added callback_seeds_postprocess method to study.py

### 12.12.2022
- fixed error in read method in core.py, where self.xrows was not adapted correctly
- added member variable u_pre_val to ESN object in core.py
- added option to set u_pre_val via SetValidationData in core.py
- use self.u_pre_val (if not None) in RunturbESN to initialize reservoir state before validation phase (make it more comparable to test phase, which is also preceeded by state from GT data)
- self.val_init_input is now taken from self.u_pre_val (if val_init_input is None and u_pre_val is not None)


### 13.12.2022
- argument seeds in create_hdf5_structure method in util.py can be int or list, range now
- fixed wrong default mse shape in RunturbESN mehtod in util.py
- renamed ComputeWassersteinDistance to compute_wasserstein_distance in util.py
- renamed ComputeMSE to compute_mse in util.py
- renamed ComputeR2 to compute_r2 in util.py
- renamed RunturbESN to run_turbESN in util.py
- renamed thread_RunturbESN to thread_run_turbESN in study.py
- renamed setDevice to set_device in core.py
- renamed toTorch to to_torch in core.py
- added compute_nrmse method to util.py



### 14.12.2022
- changed computation of mse in run_turbESN to more generic way: user uses dict esn.loss_func to specify which loss functions should be computed, results are returned in loss_dict
    - added None init. for esn.loss_func
    - adapted save_study, read_study in util.py 
    - adapted read in core.py
    - renamed read_mse to read_loss in util.py
    - adapted read_loss in util.py
    - adapted thread_run_turbESN, callback_settings, callback_seeds in study.py
    - adapted forward_validate_auto_ESN in util.py (not tested)

- removed iseed argument in read_loss, read_study, read_esn_output in util.py
- renamed _MSE_DEFAULT to _LOSS_DEFAULT in _modes.py
- adpated basic_tour.ipynb notebook to new turbESN 


## Release 0.0.1.9.3.0
----------------------
### 16.11.2022
- removed reading the loss_func from read method in core.py 

### 04.01.2023
- run_gs.py checks now whether filepath_esn already exists and does not delete old data

### 05.01.2023
- adapted to check whether groups already exist in hdf5 file in create_hdf5_groups method in util.py
- adapted get_size method in core.py to reservoir state matrices

### 07.01.2023
- use logger for logging, read config from new file: logging_config.json
- removed iseed arg in save_study, now only take one randomSeed arg
- adapted parallelize_seeds and parallelize_settings in study.py
- this now makes it easier to store several grid searches with different seeds (yet same settings) in the same .hdf5 file
- renamed PrepareTeacherData to prepare_teacher_data in util.py
- renamed PreparePredictorData to prepare_auto_data in util.py
- renamed Recursion to recursion in util.py
- moved recursion inside init_study_order in util.py
- renmaed calculateLinearNetworkTransmission to calculate_activation_argument in core.py

## Release 0.0.1.9.3.1
----------------------
- fixed open mode in save method in core.py (w->a)

## Release 0.0.1.9.4.0
----------------------
### 01.03.2023
- added KLD as compute_KLD method in util.py
- added kwarg bins in compute_wasserstein_distance in util.py
- added get_file_name method to util.py
- added read_yaml method to core.py
- fixed error in save_study in util.py (scalar lossed can now be saved)
