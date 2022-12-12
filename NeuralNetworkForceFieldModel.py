import torch
import torch.nn as nn
# import pickle
import dill
from BPDescriptor import BPDescriptor, torchpi

torch.set_default_tensor_type(torch.FloatTensor)

class NeuralNetworkForceFieldModelLoader:
    def __init__(self, model_name, pair_table, device):
        try:
            fin = open(model_name,"rb")
            self.nnff_model = dill.load(fin)
            fin.close()
        except:
            print("Error: %s not found or not readable."%(model_name))
            exit()
        self.nnff_model.elambda = None
        self.nnff_model.bp_descriptor_.update_descriptor(device, pair_table)
        self.nnff_model.to(device)
        # print("NNFF model:")
        # self.nnff_model.bp_descriptor_.print_details()
        # print("\tStructures of the Atomic Sub-networks:")
        # for i_localtype in range(self.nnff_model.number_of_localtypes_):
        #     i_globaltype = self.nnff_model.atomic_subnetwork_[i_localtype].globaltype_
        #     print("\t\ttype%s: "%(i_globaltype),end="")
        #     hidden_layer_size = self.nnff_model.atomic_subnetwork_[i_localtype].get_hidden_layer_size()
        #     for a_layer in hidden_layer_size:
        #         print("%d"%(a_layer),end="\t")
        #     print("")
        # print("")
    
    def get_model(self):
        return self.nnff_model

class AtomicSubNetwork(nn.Module):
    def __init__(self, input_size = None, hidden_layer_size = None, globaltype = None, activation_type = 'silu'):
        super().__init__()
        if input_size is None: return None
        self.globaltype_ = globaltype
        self.input_size = input_size
        self.assign_activation(activation_type)
        self.assign_hidden_layer(hidden_layer_size)
        self.linear_layer = nn.ModuleList()
        for layer_index in range(len(self.hidden_layer_size_)-1):
            self.linear_layer.append(nn.Linear(self.hidden_layer_size_[layer_index],self.hidden_layer_size_[layer_index+1]))
        
    def assign_activation(self, activation_type):
        if activation_type == 'tanh': self.activate = nn.Tanh()
        elif activation_type == 'relu': self.activate = nn.ReLU()
        elif activation_type == 'silu': self.activate = nn.SiLU()
        elif activation_type == 'sin': self.activate = lambda x: x.sin()

    def assign_hidden_layer(self, hidden_layer_size):
        self.hidden_layer_size_ = [self.input_size] + hidden_layer_size + [1]
        
    def forward(self, descriptor):
        atomic_energy = descriptor
        for layer_index in range(len(self.hidden_layer_size_)-2):
            atomic_energy = self.linear_layer[layer_index](atomic_energy)
            atomic_energy = self.activate(atomic_energy)
        atomic_energy = self.linear_layer[len(self.hidden_layer_size_)-2](atomic_energy)
        return atomic_energy

    def get_hidden_layer_size(self):
        return self.hidden_layer_size_

class AtomicSinSubNetwork(AtomicSubNetwork):
    def __init__(self, input_size = None, hidden_layer_size = None, globaltype = None, activation_type = 'silu'):
        super().__init__(input_size, hidden_layer_size, globaltype, activation_type)
    
    def forward(self, descriptor):
        atomic_energy = descriptor
        
        atomic_energy = self.linear_layer[0](atomic_energy)
        atomic_energy = atomic_energy.sin()
        for layer_index in range(1, len(self.hidden_layer_size_)-2):
            atomic_energy = self.linear_layer[layer_index](atomic_energy)
            atomic_energy = self.activate(atomic_energy)
        atomic_energy = self.linear_layer[len(self.hidden_layer_size_)-2](atomic_energy)
        return atomic_energy 

class NeuralNetworkForceFieldModel:
    def __init__(self, pair_table,\
                       hidden_layer_size = [], atomic_subnetwork = [], bp_descriptor = None, \
                       Rc = 6.0, Rs_radial_step = 0.5, Rs_angular_step = 1.5, As_angular_step = 30, eta = 1.0, zeta = 1.0,\
                       activation_type = 'silu', first_layer_sin = False, \
                       device = 'cpu'):
        self.localtype_to_globaltype_list_ = pair_table.localtype_to_globaltype_list_
        self.number_of_localtypes_ = pair_table.number_of_localtypes_
        self.number_of_globaltypes_ = pair_table.number_of_globaltypes_
        if bp_descriptor is None:
            self.bp_descriptor_ = BPDescriptor(pair_table, device, Rc = Rc, Rs_radial_step = Rs_radial_step, Rs_angular_step = Rs_angular_step, As_angular_step = As_angular_step, eta = eta, zeta = zeta)
        else:
            self.bp_descriptor_ = bp_descriptor
            self.bp_descriptor_.to(device)
        self.bp_descriptor_.print_details()
        self.create_atomic_subnetwork(hidden_layer_size, atomic_subnetwork, activation_type, first_layer_sin, device)
        self.elambda = None

    def create_atomic_subnetwork(self, hidden_layer_size, atomic_subnetwork, activation_type, first_layer_sin, device):
        self.input_size = self.bp_descriptor_.get_descriptor_dimensions()
        self.atomic_subnetwork_ = []
        print("\t\tinput dimension = %d"%(self.input_size))
        if first_layer_sin: print("\t\tfirst layer use sin activation function")
        for i_localtype in range(self.number_of_localtypes_):
            i_globaltype = self.localtype_to_globaltype_list_[i_localtype]
            if atomic_subnetwork[i_localtype] is None:
                if first_layer_sin == False:
                    self.atomic_subnetwork_.append(AtomicSubNetwork(self.input_size, hidden_layer_size[i_globaltype], i_globaltype, activation_type).to(device))
                else:
                    self.atomic_subnetwork_.append(AtomicSinSubNetwork(self.input_size, hidden_layer_size[i_globaltype], i_globaltype, activation_type).to(device))
            else:
                if atomic_subnetwork[i_localtype].input_size == self.input_size:
                    self.atomic_subnetwork_.append(atomic_subnetwork[i_localtype])
                else:
                    print("Error: an imported Atomic Sub-network has incompatible descriptor size between the BPDescriptor and AtomicSubNetwork.")
        self.param = []
        for a in self.atomic_subnetwork_:
            self.param += list(a.parameters())
    
    def to(self, device):
        for i_localtype in range(self.number_of_localtypes_):
            self.atomic_subnetwork_[i_localtype].to(device)
        self.bp_descriptor_.to(device)
        
    def freeze_model(self):
        for a in self.param:
            a.requires_grad = False
    
    def unfreeze_model(self):
        for a in self.param:
            a.requires_grad = True

    def parameters(self):
        return self.param

    def evaluate_atomic_energy(self, atom_position, lattice_vector, inverse_lattice_vector, pair_table):
        # atom_position: (N*Dim)
        descriptor = self.bp_descriptor_.compute_descriptor(atom_position, lattice_vector, inverse_lattice_vector, pair_table)
        atomic_energy = []
        for j_type in range(self.number_of_localtypes_):
            atomic_energy.append(self.atomic_subnetwork_[j_type].forward(descriptor[j_type]))
        atomic_energy = torch.cat(atomic_energy).unsqueeze(-1)
        return atomic_energy

    def evaluate_energy(self, atom_position, lattice_vector, inverse_lattice_vector, pair_table):
        # atom_position: (N*Dim)
        descriptor = self.bp_descriptor_.compute_descriptor(atom_position, lattice_vector, inverse_lattice_vector, pair_table)
        energy = 0.
        for j_type in range(self.number_of_localtypes_):
            atomic_energy = self.atomic_subnetwork_[j_type].forward(descriptor[j_type])
            energy += atomic_energy.sum()
        return energy
    
    def evaluate_force(self, atom_position, lattice_vector, inverse_lattice_vector, pair_table, create_graph = False):
        atom_position.requires_grad = True
        if atom_position.grad: atom_position.grad.zero_()
        energy = self.evaluate_energy(atom_position,lattice_vector,inverse_lattice_vector, pair_table)
        force = -torch.autograd.grad(energy, atom_position, create_graph = create_graph)[0]
        return force, energy
    
    def evaluate_force_j(self, atom_position, lattice_vector, inverse_lattice_vector, pair_table):
        if self.elambda is None:
            self.elambda = lambda atom_position: self.evaluate_energy(atom_position, lattice_vector, inverse_lattice_vector, pair_table)
        energy = self.elambda(atom_position)
        force = -torch.autograd.functional.jacobian(self.elambda, atom_position)
        return force, energy

    def evaluate_virial(self, atom_position, lattice_vector, inverse_lattice_vector, pair_table):
        if self.unit == 0:
            prefactor = 4184/6.02214076e-7/101325 # kcal/mole/A^3 -> atm
        else:
            prefactor = 1.602176565e+6 # eV/A^3 -> bar
        inverse_volume = 1.0/torch.linalg.det(lattice_vector)
        prefactor *= inverse_volume
        lattice_vector.requires_grad = True
        if lattice_vector.grad: lattice_vector.grad.zero_()
        # inv_lattice_vector = lattice_vector.inverse()
        force, energy = self.evaluate_force(atom_position, lattice_vector, inverse_lattice_vector, pair_table)
        lattice_virial = torch.matmul(torch.autograd.grad(energy, lattice_vector)[0].transpose(-2,-1), lattice_vector)*prefactor
        force_virial = -torch.matmul(force.transpose(-2,-1), atom_position)*prefactor
        virial = -(force_virial+lattice_virial)
        virial = (virial.transpose(0,1) + virial)*0.5
        pressure = virial.diag().mean()
        return pressure, virial, force, energy
    