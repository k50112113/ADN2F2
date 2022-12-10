import torch
from MinimumImage import minimum_image, scale_position
from Clock import Clock

torch.set_default_tensor_type(torch.FloatTensor)

class ClassicalForceField:
    def __init__(self, number_of_types, pair_table,\
                 Rc = 6.0, epsilon = None, sigma = None, pair_epsilon = None, pair_sigma = None, r12_repulsion = True, r6_attraction = True, unit = 'real'):
        self.number_of_types_ = number_of_types
        self.type_index_pair_list_ = pair_table.localtype_pair_list_
        self.number_of_type_pairs_ = pair_table.number_of_localtype_pairs_
        self.Rc_ = Rc
        self.zero_ = torch.tensor(0.0)
        self.unit = 0 if unit == 'real' else 1
        if pair_epsilon:
            self.pair_epsilon_ = pair_epsilon
        else:
            if epsilon: self.epsilon_ = epsilon
            else:       self.epsilon_ = [1.0]*self.number_of_types_
            self.pair_epsilon_ = []
            for p_type_pair in range(self.number_of_type_pairs_):
                j_type = self.type_index_pair_list_[p_type_pair][1]
                i_type = self.type_index_pair_list_[p_type_pair][0]
                self.pair_epsilon_.append((self.epsilon_[j_type]*self.epsilon_[i_type])**0.5)

        if pair_sigma:
            self.pair_sigma_ = pair_sigma
        else:
            if sigma:   self.sigma_ = sigma
            else:       self.sigma_ = [1.0]*self.number_of_types_
            self.pair_sigma_ = []
            for p_type_pair in range(self.number_of_type_pairs_):
                j_type = self.type_index_pair_list_[p_type_pair][1]
                i_type = self.type_index_pair_list_[p_type_pair][0]
                self.pair_sigma_.append((self.sigma_[j_type]+self.sigma_[i_type])*0.5)

        if r12_repulsion and r6_attraction:
            self.lj = lambda pair_distance, epsilon, sigma: self.lj_12_6(pair_distance, epsilon, sigma)
        elif r12_repulsion and not r6_attraction:
            self.lj = lambda pair_distance, epsilon, sigma: self.lj_12(pair_distance, epsilon, sigma)
        elif not r12_repulsion and r6_attraction:
            self.lj = lambda pair_distance, epsilon, sigma: self.lj_6(pair_distance, epsilon, sigma)
        else:
            print("No classical term is selected")
            exit()

    def to(self, device):
        self.zero_ = self.zero_.to(device)

    def evaluate_energy(self, atom_position, lattice_vector, inverse_lattice_vector, pair_table):
        # atom_position: (N*Dim)
        # pair_displacement = self.pair_table_.compute_pair_displacement2(atom_position, lattice_vector, inverse_lattice_vector)
        # pair_distance = self.pair_table_.compute_pair_distance2(pair_displacement)
        # for a in pair_displacement:
        #     print(a.shape)
        pair_displacement = pair_table.compute_pair_displacement(atom_position, lattice_vector, inverse_lattice_vector) 
        pair_distance = pair_table.compute_pair_distance(pair_displacement)
        # for a in pair_displacement:
        #     print(a.shape)
        total_energy = 0.
        for p_type_pair in range(self.number_of_type_pairs_):
            pair_energy = self.lj(pair_distance[p_type_pair], self.pair_epsilon_[p_type_pair], self.pair_sigma_[p_type_pair]).sum(dim=(-2,-1))
            if self.type_index_pair_list_[p_type_pair][1] == self.type_index_pair_list_[p_type_pair][0]:
                pair_energy /= 2.
            total_energy += pair_energy
        return total_energy
    
    def evaluate_force(self, atom_position, lattice_vector, inverse_lattice_vector, pair_table):
        # elambda = lambda atom_position: self.evaluate_energy(atom_position, lattice_vector, inverse_lattice_vector)
        # total_energy = elambda(atom_position)
        # total_force = -torch.autograd.functional.jacobian(elambda, atom_position).sum(dim=0)
        # return total_force, total_energy

        atom_position.requires_grad = True
        if atom_position.grad: atom_position.grad.zero_()

        total_energy = self.evaluate_energy(atom_position,lattice_vector,inverse_lattice_vector, pair_table)
        total_force = -torch.autograd.grad(total_energy, atom_position, create_graph=True)[0]
        # total_force = -torch.autograd.grad(total_energy, ap)[0]
        return total_force, total_energy

    def evaluate_virial(self, atom_position, lattice_vector, inverse_lattice_vector, pair_table):

        if self.unit == 0:
            prefactor = 4184/6.02214076e-7/101325 # kcal/mole/A^3 -> atm
        else:
            prefactor = 1.602176565e+6 # eV/A^3 -> bar
        inverse_volume = 1.0/torch.linalg.det(lattice_vector)
        prefactor*=inverse_volume
        
        lattice_vector.requires_grad = True
        if lattice_vector.grad: lattice_vector.grad.zero_()
        # inv_lattice_vector = lattice_vector.inverse()

        total_force, total_energy = self.evaluate_force(atom_position, lattice_vector, inverse_lattice_vector, pair_table)
        lattice_virial = torch.matmul(torch.autograd.grad(total_energy, lattice_vector)[0].transpose(-2,-1), lattice_vector)*prefactor
        force_virial = -torch.matmul(total_force.transpose(-2,-1), atom_position)*prefactor
        total_virial = -(force_virial+lattice_virial)
        total_virial = (total_virial.transpose(0,1) + total_virial)*0.5
        total_pressure = total_virial.diag().mean()
        return total_pressure, total_virial, total_force, total_energy 

    def lj_12_6(self,pair_distance, epsilon, sigma):
        return torch.where((pair_distance-self.Rc_/2.)**2 < (self.Rc_/2.)**2, 4*epsilon * ( (sigma/pair_distance)**(12) - (sigma/pair_distance)**(6) ), self.zero_)#.nan_to_num(nan=0.0)
    
    def lj_12(self,pair_distance, epsilon, sigma):
        return torch.where((pair_distance-self.Rc_/2.)**2 < (self.Rc_/2.)**2, 4*epsilon * ( (sigma/pair_distance)**(12) ), self.zero_)#.nan_to_num(nan=0.0)
    
    def lj_6(self,pair_distance, epsilon, sigma):
        return torch.where((pair_distance-self.Rc_/2.)**2 < (self.Rc_/2.)**2, 4*epsilon * ( - (sigma/pair_distance)**(6) ), self.zero_)#.nan_to_num(nan=0.0)
    