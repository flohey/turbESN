# turbESN
Echo State Network code (used in parts of the DeepTurb project of Carl-Zeiss-Stiftung). Relies on a pytorch backend. Inspired by the easyesn package (https://github.com/kalekiu/easyesn).

Supports fully autonomous and teacher forced ESN predictions. A hybrid of both methods, called semi-techer forcing, is also implemented (restricted use).

## Installation
In order to install the package just use `pip3 install turbESN`. Here you can find the current PyPi release https://pypi.org/project/turbESN/.

## Requirements
Stable for
- `python `>=  3.6.0
- `torch`  >= 1.10.0
- `numpy`  >= 1.20.1
- `h5py`   >= 2.10.0 

## Manual
A first basic introduction can be found in the notbeook`basic_tour.ipynb`.  This example uses data from the Lorenz '63 system (https://en.wikipedia.org/wiki/Lorenz_system).

## About Echo State Networks/ Reservoir Computing
Classical Reservoir Computing  (RC) makes use of a nonlinear dynamical expansion of the information to be processed. This information is lifted to a high-dimensional reservoir state. A linear readout after each reservoir update yields the reservoir output. The training scheme of RC is concerned with finding the right hyperplane of within the reservoir space, s.t. the reservoir output matches the expected output.Therefore only the linear readout is trained, usually to minimize the mean square error between output and target. Therefore the readout can be computed by a simple linear regression (often with Tikhonov regularization/ L2 penalty). This, always converging learning procedure, makes RC an extremely computationally efficient method.
RC is more then just a method, it stands for new computing paradigm (in contrast to the Turing-von Neumann architecture).

The Echo State Network (ESN) is a certain implementation of RC. Its architecture follows a Recurrent Neural Network (RNN) structure, where only the output weights are trained by linear regression. This is also a simple way of circumventing the known exploding/ vanishing gradient problem which RNN suffer from.

## Applications:
- F. Heyder and J. Schumacher - Phys. Rev. E **103**, 053107 (2021):
  "Echo state network for two-dimensional turbulent moist Rayleigh-Bénard convection"

- P. Pfeffer, F. Heyder and J. Schumacher - Phys Rev. Res. **4**, 033176 (2022): 
  "Hybrid quantum-classical reservoir computing of thermal convection flow"

- F. Heyder and J. P. Mellado and J. Schumacher - Phys. Rev. E **106**, 055303 (2022): 
  "Generalizability of reservoir computing for flux-driven two-dimensional convection"

- M. S. Ghazijahani, F. Heyder, J. Schumacher and C. Cierpka, Meas. Sci. Technol. **34** (2023) 014002 
  "On the benefits and limitations of Echo State Networks for turbulent flow prediction"

