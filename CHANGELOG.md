 
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
