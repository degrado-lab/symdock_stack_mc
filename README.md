# SymDOCK Stack MC

Monte Carlo sampling of optimal ligand stack conformers with the ANI force field.

### Requirements and Installation

Make sure you have [Anaconda] (https://www.anaconda.com/download) or [Miniconda] 
(https://www.anaconda.com/docs/getting-started/miniconda/install) set up on your 
machine. Create the environment necessary for running the code as follows:
```bash
conda create -f environment.yml
conda activate env_torchani
```

Then follow the instructions at the [TorchANI GitHub repository] 
(https://github.com/aiqm/torchani) to install torchani with pip.

### Use
The primary script for running MC on stacks is `monte_carlo_stack.py`. 
The two necessary arguments for this script are a mol2 file containing *three* 
copies of the input ligand in a stack and a path to the output CSV file where 
the energies, RMSDs, and associated affine transformations of the MC-optimized 
stacks will be output. A number of additional, optional arguments can be 
viewed using `python monte_carlo_stack.py -h`. An example mol2 file has been 
supplied in the `examples` directory.
