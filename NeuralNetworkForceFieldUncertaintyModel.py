import torch
import torch.nn as nn
import pickle
from BPDescriptor import BPDescriptor, torchpi
from NeuralNetworkForceFieldModel import AtomicSubNetwork, NeuralNetworkForceFieldModel

torch.set_default_tensor_type(torch.FloatTensor)

class AtomicUncertaintySubNetwork(AtomicSubNetwork):
    def __init__(self, input_size = None, hidden_layer_size = None, globaltype = None, activation = 'silu'):
        super().__init__(input_size, hidden_layer_size, globaltype, activation_type)
        
    def assign_hidden_layer(self, hidden_layer_size):
        self.hidden_layer_size_ = [self.input_size] + hidden_layer_size + [2]

class NeuralNetworkForceFieldUncertaintyModel(NeuralNetworkForceFieldModel):
    def __init__(self, pair_table,\
                       hidden_layer_size = [], atomic_subnetwork = [], bp_descriptor = None, \
                       Rc = 6.0, Rs_radial_step = 0.5, Rs_angular_step = 1.5, As_angular_step = 30, eta = 1.0, zeta = 1.0,\
                       device = 'cpu'):
        super().__init__(pair_table,\
                         hidden_layer_size, atomic_subnetwork, bp_descriptor, \
                         Rc, Rs_radial_step, Rs_angular_step, As_angular_step, eta, zeta,\
                         device)

    def evaluate_energy(self, atom_position, lattice_vector, inverse_lattice_vector, pair_table):
        # atom_position: (N*Dim)
        descriptor = self.bp_descriptor_.compute_descriptor(atom_position, lattice_vector, inverse_lattice_vector, pair_table)
        energy = 0.
        energy_variance = 0.
        for j_type in range(self.number_of_localtypes_):
            atomic_energy = self.atomic_subnetwork_[j_type].forward(descriptor[j_type])
            energy += atomic_energy[:,0].sum()
            energy_variance += atomic_energy[:,1].sum()
        return energy, energy_variance
    
    def evaluate_force(self, atom_position, lattice_vector, inverse_lattice_vector, pair_table, create_graph = False):
        atom_position.requires_grad = True
        if atom_position.grad: atom_position.grad.zero_()
        energy, energy_variance = self.evaluate_energy(atom_position,lattice_vector,inverse_lattice_vector, pair_table)
        force = -torch.autograd.grad(energy, atom_position, create_graph = create_graph)[0]
        force_variance = -torch.autograd.grad(energy_variance, atom_position, create_graph = create_graph)[0]
        return force, force_variance, energy, energy_variance

    def evaluate_virial(self, atom_position, lattice_vector, inverse_lattice_vector, pair_table):
        if self.unit == 0:
            prefactor = 4184/6.02214076e-7/101325 # kcal/mole/A^3 -> atm
        else:
            prefactor = 1.602176565e+6 # eV/A^3 -> bar
        inverse_volume = 1.0/torch.linalg.det(lattice_vector)
        prefactor *= inverse_volume
        lattice_vector.requires_grad = True
        if lattice_vector.grad: lattice_vector.grad.zero_()
        force, _, energy, _ = self.evaluate_force(atom_position, lattice_vector, inverse_lattice_vector, pair_table)
        lattice_virial = torch.matmul(torch.autograd.grad(energy, lattice_vector)[0].transpose(-2,-1), lattice_vector)*prefactor
        force_virial = -torch.matmul(force.transpose(-2,-1), atom_position)*prefactor
        virial = -(force_virial+lattice_virial)
        virial = (virial.transpose(0,1) + virial)*0.5
        pressure = virial.diag().mean()
        return pressure, virial, force, energy
    