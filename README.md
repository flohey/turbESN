# turbESN
Echo State Network code (used in parts of the DeepTurb project of Carl-Zeiss-Stiftung). Relies on a pytorch backend. Inspired by the easyesn package (https://github.com/kalekiu/easyesn).

Supports fully autonomous and teacher forced ESN predictions. A hybrid of both methods, called semi-techer forcing, is also implemented (restricted use).

## Installation
In order to install the package just use `pip3 install turbESN`. Here you can find the current PyPi release https://pypi.org/project/turbESN/.

## Requirements
Stable for
- `python `>=  3.6.0
- `torch`  >= 1.8.1
- `numpy`  >= 1.20.1
- `h5py`   >= 2.10.0 
- `networkx`   >= 2.7.0 


## Manual
A first basic introduction can be found in the notbeook`basic_tour.ipynb`.  This example uses data from the Lorenz '63 system (https://en.wikipedia.org/wiki/Lorenz_system).

## About Echo State Networks/ Reservoir Computing
TO DO

## Applications:
- F. Heyder and J. Schumacher - Phys. Rev. E **103**, 053107:
  "Echo state network for two-dimensional turbulent moist Rayleigh-Bénard convection"
  
- F. Heyder and J. P. Mellado and J. Schumacher - arXiv:2108.06195: 
  "Echo state network models of a two-dimensional convective boundary layer "

## To Do
...
