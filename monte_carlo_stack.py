import os
import sys
import argparse
import warnings
from time import time
from copy import copy,deepcopy

import numpy as np
from scipy.optimize import root
from scipy.spatial.distance import cdist

import torchani
from ase import units, Atoms

from matplotlib import pyplot as plt

class IG_SO3:
    def __init__(self, mean, var, l_max=2000):
        """Initialize an isotropic Gaussian distribution over SO(3) rotations.

        Parameters
        ----------
        mean : np.array [3 x 3]
            Mean rotation matrix of the distribution.
        var : float
            Scalar variance of the distribution.
        l_max : float
            Maximum value of l for use in PDF and CDF calculations.
        """
        self.mean = mean
        self.var = var
        self.l = np.arange(l_max)

    def angle_pdf(self, omega):
        """Compute the angle PDF for the isotropic SO(3) Gaussian distribution.

        Parameters
        ----------
        omega : float
            Rotation angle for which to compute the PDF.

        Returns
        -------
        f_omega : float
            The probability density for the angle omega.
        """
        if omega == 0.:
            return np.zeros(1)
        return (1. - np.cos(omega)) / np.pi * \
            np.sum((2. * self.l + 1) * 
                   np.exp(-self.l * (self.l + 1.) * self.var / 2.) * 
                   np.sin((self.l + 0.5) * omega) / np.sin(0.5 * omega))
    
    def angle_cdf(self, omega):
        """Compute the angle CDF for the isotropic SO(3) Gaussian distribution.

        Parameters
        ----------
        omega : float
            Rotation angle for which to compute the CDF.

        Returns
        -------
        F_omega : float
            The angle CDF for the rotation angle omega.
        """
        return omega / np.pi * \
            np.sum((2. * self.l + 1.) * 
                   np.exp(-self.l * (self.l + 1.) * self.var / 2.) * 
                   (np.sinc(self.l * omega / np.pi) - 
                    np.sinc((self.l + 1.) * omega / np.pi)))

    def rotmat_pdf(self, R):
        """Compute the rotation matrix PDF for the isotropic SO(3) Gaussian.

        Parameters
        ----------
        R : np.array [3 x 3]
            Rotation matrix for which to compute the probability density.

        Returns
        -------
        f_R : float
            The probability density for the rotation matrix R.
        """
        omega = np.arccos(0.5 * (np.trace(np.dot(mean.T, R)) - 1.))
        return self.angle_pdf(omega)

    def sample(self, method='lm'):
        """Sample a rotation from the isotropic SO(3) Gaussian distribution.

        Returns
        -------
        R : np.array [3 x 3]
            Rotation matrix sampled from the isotropic SO(3) Gaussian 
            distribution.
        method : str
            Method for the root finder to determine omega. 
        """
        u_phi, u_theta, u_omega = np.random.random(3)
        phi = 2. * np.pi * u_phi
        theta = np.arccos(1. - 2. * u_theta)
        ax = np.array([np.cos(phi) * np.sin(theta), 
                       np.sin(phi) * np.sin(theta), 
                       np.cos(theta)])
        cpm = np.cross(ax, -np.eye(3)) # cross product matrix of ax
        omega = root(lambda omega: self.angle_cdf(omega) - u_omega, 
                     u_omega, method=method, jac=self.angle_pdf).x
        omega = np.clip(omega, 0., np.pi)
        dR =  np.cos(omega) * np.eye(3) + np.sin(omega) * cpm + \
              (1. - np.cos(omega)) * np.outer(ax, ax)
        return np.dot(dR, self.mean)


def kabsch(P, Q):
    """Align two sets of points using the Kabsch algorithm.

    Parameters
    ----------
    P : np.array [N x 3]
        Array of coordinates to align with Q.
    Q : np.array [N x 3]
        Array of coordinates against which to align P.

    Returns
    -------
    R : np.array [3 x 3]
        Rotation matrix of the optimal transform of P to Q.
    t : np.array [3]
        Translation vector of the optimal transform of P to Q.
    P_aligned : np.array [N x 3]
        P after alignment, i.e P x R + t.
    rmsd : float
        RMSD between P_aligned and Q.
    """
    Pbar = np.mean(P, axis=0)
    Qbar = np.mean(Q, axis=0)
    H = np.dot((P - Pbar).T, (Q - Qbar))
    U, S, Vt = np.linalg.svd(H, full_matrices=True)
    d = np.linalg.det(np.dot(U, Vt))
    R = np.dot(U, np.dot(np.diag([1, 1, d]), Vt))
    # define t such that X1 = np.dot(X0, R) + t
    t = Qbar - np.dot(Pbar, R)
    P_aligned = np.dot(P, R) + t
    rmsd = np.sqrt(np.sum((P_aligned - Q) ** 2) / len(P))
    return R, t, P_aligned, rmsd

def update_stacks(atoms, two_stack, three_stack, R, t):
    """Apply a rotation and transformation to a molecule to stack it.

    Parameters
    ----------
    atoms : ase.Atoms
        ase Atoms object for a single molecule.
    two_stack : ase.Atoms
        ase Atoms object for a stack of two molecules.
    three_stack : ase.Atoms
        ase Atoms object for a stack of three molecules.

    Returns
    -------
    flag : bool
        If True, clashes exist within the first 10 images or neighboring 
        images are too far to have an ANI interaction energy.
    """
    mol_n_atoms = len(atoms)
    positions = np.vstack([atoms.positions] +
                          [np.zeros_like(atoms.positions)] * 9)
    for i in range(9):
        positions[(i+1)*mol_n_atoms:(i+2)*mol_n_atoms] = \
            np.dot(positions[i*mol_n_atoms:(i+1)*mol_n_atoms], R) + t
    two_stack.set_positions(positions[:2*mol_n_atoms])
    three_stack.set_positions(positions[:3*mol_n_atoms])
    clashing = cdist(atoms.positions, positions[mol_n_atoms:]).min() < 1.2
    distant = cdist(atoms.positions, 
                    positions[mol_n_atoms:2*mol_n_atoms]).min() > 5.1
    return clashing or distant

def per_monomer_energy(E_mono, two_stack, three_stack):
    """Compute the per-monomer energy of a stack using ANI.

    Parameters
    ----------
    E_mono : float
        Energy (in eV) of one monomer.
    two_stack : ase.Atoms
        ase Atoms object for a stack of two molecules. Must have a 
        calculator set.
    three_stack : ase.Atoms
        ase Atoms object for a stack of three molecules. Must have a 
        calculator set.

    Returns
    -------
    E : float
        Per monomer energy (in kcal/mol) of the stack.
    """
    mol_n_atoms = len(two_stack) // 2
    # check whether a molecular trimer calculation is justified
    x0 = three_stack[:mol_n_atoms].positions
    x1 = three_stack[mol_n_atoms:2 * mol_n_atoms].positions
    x2 = three_stack[2 * mol_n_atoms:].positions
    mindists10 = cdist(x1, x0, metric='sqeuclidean').min(axis=1)
    mindists12 = cdist(x1, x2, metric='sqeuclidean').min(axis=1)
    # check whether atoms from monomers 0 and 2 lie within 5.1 Angstroms of 
    # an atom in monomer 1, thus giving rise to a three-body contribution
    trimer_flag = np.any(np.logical_and(mindists10 < 26.01, 
                                        mindists12 < 26.01))
    if trimer_flag:
        # compute three-body contribution by subtracting the energies of 
        # three monomers from the total trimer energy, then divide by three 
        # to get the per-monomer energy
        E = (three_stack.get_potential_energy() - 3. * E_mono) / 3.
    else:
        # compute two-body contribution by subtracting the energies of two 
        # monomers from the total dimer energy, then divide by two to get 
        # the per-monomer energy
        E = (two_stack.get_potential_energy() - 2. * E_mono) / 2.
    E *= 23.0609 # convert from eV to kcal/mol
    return E

def mol2_to_Atoms(mol2):
    """Read mol2 file and output molecule names and ase Atoms objects.

    Parameters
    ----------
    mol2 : str
        Path to mol2 file to read.

    Returns
    -------
    atoms_list : list
        List of ase Atoms objects for each molecule in the mol2 file.
    names_list : list
        List of str-valued names for each molecule in the mol2 file.
    """
    with open(mol2, 'r') as f:
        lines = f.read().split('\n')

    names_list = []
    atoms_list = []

    symbols = []
    coords = []

    is_name = False
    is_atom = False
    for line in lines:
        if is_name:
            names_list.append(line.split()[0])
        if line == "@<TRIPOS>MOLECULE":
            is_name = True
        else:
            is_name = False
        if line == "@<TRIPOS>ATOM":
            is_atom = True
            if len(symbols):
                atoms_list.append(Atoms(symbols, np.array(coords)))
            symbols = []
            coords = []
        elif len(line) == 0 or line[:9] == "@<TRIPOS>":
            is_atom = False
        if line != "@<TRIPOS>ATOM" and is_atom:
            tokens = line.split()
            if tokens[1][:2] == 'Cl':
                symbols.append('Cl')
            elif tokens[1][:2] == 'Br':
                symbols.append('Cl') # replace Br with Cl
            elif tokens[1][:1] == 'I':
                symbols.append('Cl') # replace I with Cl
            else:
                symbols.append(tokens[1][0])
            coords.append([float(val) for val in tokens[2:5]])
            if symbols[-1] not in ["H", "C", "N", "O", "F", "S", "Cl"]:
                print('Warning: Atom {} in molecule {} not supported by '
                      'ANI force field.'.format(symbols[-1], names_list[-1]))
    if len(symbols):
        atoms_list.append(Atoms(symbols, np.array(coords)))
    return atoms_list, names_list

def parse_args():
    argp = argparse.ArgumentParser()
    argp.add_argument("in_mol2", type=os.path.realpath, 
                      help="Path to input mol2 file.")
    argp.add_argument("out_csv", type=os.path.realpath, 
                      help="Path at which to output the CSV file containing "
                      "energies, RMSDs, and affine transformations of the "
                      "MC-optimized stacks.")
    argp.add_argument("--cc", action="store_true", 
                      help="Use ANI1ccx instead of ANI2x.")
    argp.add_argument("-n", "--n_sweep", type=int, default=1000, 
                      help="Number of MC sweeps to do per ligand.")
    argp.add_argument("-b", "--beta", type=float, default=1., 
                      help="Inverse temperature for Metropolis-Hastings MCMC.")
    argp.add_argument("-r", "--rot_stdev", type=float, default=0.1, 
                      help="Standard deviation of the SO(3) Isotropic "
                      "Gaussian MC moves on the rotation matrix.")
    argp.add_argument("-t", "--trans_stdev", type=float, default=0.1, 
                      help="Standard deviation of the Gaussian MC moves on "
                      "the translation vector.")
    argp.add_argument("--n-mono", type=int, default=2, 
                      help="Number of monomers in the input mol2 file.")
    argp.add_argument("-v", "--verbose", action="store_true", 
                      help="Print runtime and acceptance probability over "
                      "all sweeps for each molecule.")
    return argp.parse_args()

def main():
    args = parse_args()

    if args.cc:
        calculator = torchani.models.ANI1ccx().ase()
    else:
        calculator = torchani.models.ANI2x().ase()

    if not os.path.exists(args.out_csv):            
        with open(args.out_csv, "a") as f:
            f.write(('name,pre_opt_energy (kcal/mol),'
                     'post_opt_energy (kcal/mol),'
                     'pre_post_opt_rmsd (Angstrom),'
                     'rot_mat_row_1,rot_mat_row_2,'
                     'rot_mat_row_3,transl_vec') + '\n')

    ig = IG_SO3(np.eye(3), args.rot_stdev ** 2, 
                l_max=int(np.floor(8.8 / args.rot_stdev)))
    
    atoms_list, names_list = mol2_to_Atoms(args.in_mol2)

    for atoms, name in zip(atoms_list, names_list):
        mol_n_atoms = len(atoms) // args.n_mono # REMOVE
        atoms_mate = atoms[mol_n_atoms:2*mol_n_atoms] # REMOVE
        atoms = atoms[:mol_n_atoms] # REMOVE
        R0, t0_raw, _, _ = kabsch(atoms.positions, atoms_mate.positions)
        mu = atoms.positions.mean(axis=0)
        atoms.set_positions(atoms.positions - mu)
        t0 = t0_raw + np.dot(mu, R0) - mu
        ig.mean = R0

        start = time()
        two_stack = deepcopy(atoms) + deepcopy(atoms)
        three_stack = deepcopy(two_stack) + deepcopy(atoms)
        flag = update_stacks(atoms, two_stack, three_stack, R0, t0)
        if flag:
            print(('WARNING: molecule {} features dimers that clash or have a '
                   'minimum interatomic distance greater than 5.1 Angstroms, '
                   'and thus are unsuitable for ANI sampling.').format(name))
            continue
        '''
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        colors = ['r', 'g', 'b']
        for k, color in enumerate(colors):
            ax.scatter(three_stack.positions[k*mol_n_atoms:(k+1)*mol_n_atoms, 0], 
                       three_stack.positions[k*mol_n_atoms:(k+1)*mol_n_atoms, 1],
                       three_stack.positions[k*mol_n_atoms:(k+1)*mol_n_atoms, 2], 
                       color=color)
        plt.show()
        '''
        atoms.set_calculator(calculator)
        two_stack.set_calculator(calculator)
        three_stack.set_calculator(calculator)
        E_mono = atoms.get_potential_energy()
        E0 = per_monomer_energy(E_mono, two_stack, three_stack)
        E_orig, X_orig = E0, copy(two_stack.positions)
        sweep_min, E_min, X_min, R_min, t_min = \
            -1, E0, copy(X_orig), copy(R0), copy(t0)

        accepted = [False] * args.n_sweep
        for sweep in range(args.n_sweep): # Metropolis-Hastings MCMC
            R = ig.sample()
            t = t0 + args.trans_stdev * np.random.randn(3)
            if update_stacks(atoms, two_stack, three_stack, R, t):
                continue
            E = per_monomer_energy(E_mono, two_stack, three_stack)
            # accept or reject moves; forward and reverse generation 
            # probabilities are equal and so need not be included
            if E0 > E or np.exp(args.beta * (E0 - E)) >= np.random.random():
                accepted[sweep] = True
                E0, R0, t0 = E, copy(R), copy(t)
                ig.mean = R0
                if E0 < E_min:
                    sweep_min, E_min, X_min, R_min, t_min = \
                        sweep, E0, copy(two_stack.positions), R0, \
                        t0 + mu - np.dot(mu, R0)
       
        _, _, _, rmsd = kabsch(X_orig, X_min)

        atoms.set_positions(atoms.positions + mu)
        update_stacks(atoms, two_stack, three_stack, R_min, t_min)
        '''
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        colors = ['r', 'g', 'b']
        for k, color in enumerate(colors):
            ax.scatter(three_stack.positions[k*mol_n_atoms:(k+1)*mol_n_atoms, 0], 
                       three_stack.positions[k*mol_n_atoms:(k+1)*mol_n_atoms, 1],
                       three_stack.positions[k*mol_n_atoms:(k+1)*mol_n_atoms, 2], 
                       color=color)
        plt.show()
        '''

        with open(args.out_csv, "a") as f:
            f.write(name + ',' + str(E_orig) + ',' + 
                    str(E_min) + ',' + str(rmsd) + ',' + 
                    ' '.join([str(val) for val in R_min[0]]) + ',' + 
                    ' '.join([str(val) for val in R_min[1]]) + ',' + 
                    ' '.join([str(val) for val in R_min[2]]) + ',' + 
                    ' '.join([str(val) for val in t_min]) + '\n')

        if args.verbose:
            print('Molecule : {}'.format(name))
            print('\tOptimized at MC sweep {}'.format(sweep_min))
            print('\tRuntime: {} seconds'.format(time() - start))
            print('\tAcceptance probability: {}'.format(np.mean(accepted)))

if __name__ == "__main__":
    main()
