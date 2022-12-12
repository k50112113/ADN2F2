import numpy as np
import sys
import torch 
from Trajectory import DFTTrajectory
from ClassicalForceField import ClassicalForceField
from PairTable import PairTable
from MinimumImage import minimum_image
from Clock import Clock
from AscentDynamics import AscentDynamics
from NeuralNetworkForceFieldModel import NeuralNetworkForceFieldModelLoader

torch.set_default_tensor_type(torch.FloatTensor)

class MolecularDynamics:
    def __init__(self, device = None, unit = 'atomic'):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.dt     = self.to_tensor(1).to(self.device)
        self.timestep = 0
        if unit == 'real':
            self.kb = self.to_tensor(1.987204259e-3).to(self.device) #(kcal/mole/K)
        elif unit == 'atomic':
            self.kb = self.to_tensor(8.617333262e-5).to(self.device) #(eV/K)
        self.inv_kb = 1.0/self.kb
        self.inv_three_over_two_kb = 1.5 * self.inv_kb
        self.additional_forcefield = {}
        self.pure_repulsion_forcefield = None
        pass
    
    def open_dump_file(self, filename):
        self.dump_file_ptr = open(filename,"w")
    
    def close_dump_file(self):
        self.dump_file_ptr.close()
    
    def dump_frame(self):
        self.dump_file_ptr.write("ITEM: TIMESTEP\n%d\nITEM: NUMBER OF ATOMS\n%d\n"%(self.timestep, self.number_of_atoms_))
        self.dump_file_ptr.write("ITEM: BOX BOUNDS pp pp pp\n0.0 %f\n0.0 %f\n0.0 %f\n"%(self.lattice_vector_[0][0].item(),self.lattice_vector_[1][1].item(),self.lattice_vector_[2][2].item()))
        self.dump_file_ptr.write("ITEM: ATOMS id type x y z\n")
        with torch.no_grad():
            wrap_atom_position = (minimum_image(self.atom_position_, self.lattice_vector_, self.inverse_lattice_vector_) + self.lattice_vector_.diag()*0.5).squeeze(0)
        for i_atom in range(self.number_of_atoms_):
            self.dump_file_ptr.write("%d %d %f %f %f\n"%(i_atom+1,self.atom_type_[i_atom].item()+1,wrap_atom_position[i_atom][0],wrap_atom_position[i_atom][1],wrap_atom_position[i_atom][2]))
    
    def initialize_box(self, atom_position, lattice_vector, atom_type = None, atom_mass = None, atom_velocity = None):
        self.atom_position_                             = self.to_tensor(atom_position).to(self.device)
        self.number_of_atoms_                           = len(self.atom_position_)
        self.inv_number_of_atoms_                       = self.to_tensor(1.0/self.number_of_atoms_).to(self.device)
        self.lattice_vector_                            = self.to_tensor(lattice_vector).to(self.device)
        self.inverse_lattice_vector_                    = self.lattice_vector_.inverse().to(self.device)
        if atom_velocity is None:   self.atom_velocity_ = torch.zeros_like(self.atom_position_).to(self.device)
        else:                       self.atom_velocity_ = self.to_tensor(atom_velocity).to(self.device)
        if atom_mass is None:       self.atom_mass_     = torch.ones(self.number_of_atoms_).to(self.device)
        else:                       self.atom_mass_     = self.to_tensor(atom_mass).to(self.device)
        self.inverse_atom_mass_ = 1.0/self.atom_mass_
        self.inverse_of_sum_of_atom_mass_ = 1.0/self.atom_mass_.sum()
        if atom_type is None:       self.atom_type_     = torch.ones(self.number_of_atoms_).to(self.device)
        else:                       self.atom_type_     = self.to_tensor(atom_type).to(self.device)
        self.atom_type_ = self.atom_type_.to(torch.int64)
        self.type_list_ = [int(i) for i in torch.unique(self.atom_type_, sorted = True)]
        self.number_of_types_ = len(self.type_list_)
        self.set_pair_table()

        if self.atom_position_.shape != self.atom_velocity_.shape:
            print("Error: Tensor of atom position and velocity must have the same shape.")
            exit()
    
    def to_tensor(self, x):
        if torch.is_tensor(x) == False: return torch.tensor(x, dtype = torch.float32)
        else: return x.clone().detach()

    def set_pair_table(self):
        self.type_indices_ = []
        for i_type in range(self.number_of_types_):
            self.type_indices_.append((self.atom_type_ == i_type).nonzero(as_tuple=False).squeeze())
        self.pair_table_ = PairTable(self.type_list_, self.type_indices_, self.number_of_types_)

    def set_forcefield(self, model_name = 'lj', Rc = 2.0, epsilon = [1.0], sigma = [1.0], set_index = 0, return_forcefield = False):
        if model_name == 'lj':
            this_forcefield = ClassicalForceField(self.number_of_types_, self.pair_table_, Rc = Rc, epsilon = epsilon, sigma = sigma)
            this_forcefield.to(self.device)
        elif model_name == 'lj-kob':
            this_forcefield = ClassicalForceField(self.number_of_types_, self.pair_table_, Rc = 2.5*0.8, pair_epsilon = [1.0, 1.5, 0.5], pair_sigma = [1.0, 0.8, 0.88])
            this_forcefield.to(self.device)
        elif model_name[-5:] == ".nnff":
            this_forcefield = NeuralNetworkForceFieldModelLoader(model_name, self.pair_table_, self.device).get_model()
        
        if return_forcefield: return this_forcefield
        
        self.additional_forcefield[set_index] = this_forcefield
        if set_index == 0: self.forcefield = self.additional_forcefield[set_index]

    def add_pure_repulsion_forcefield(self, Rc = 2.0, epsilon = [1.0], sigma = [1.0]):
        self.pure_repulsion_forcefield = ClassicalForceField(self.number_of_types_, self.pair_table_, Rc = Rc, epsilon = epsilon, sigma = sigma, r6_attraction = False)
        self.pure_repulsion_forcefield.to(self.device)

    def potential(self):
        return self.forcefield.evaluate_energy(self.atom_position_, self.lattice_vector_, self.inverse_lattice_vector_, self.pair_table_)

    def force_potential(self):
        force, pe = self.forcefield.evaluate_force(self.atom_position_, self.lattice_vector_, self.inverse_lattice_vector_, self.pair_table_)
        # force, pe = self.forcefield.evaluate_force_j(self.atom_position_, self.lattice_vector_, self.inverse_lattice_vector_, self.pair_table_)
        return force, pe
    
    def atomic_potential(self):
        atomic_pe = self.forcefield.evaluate_atomic_energy(self.atom_position_, self.lattice_vector_, self.inverse_lattice_vector_, self.pair_table_)
        return atomic_pe
    
    def get_temperature(self):
        # mass (float): mass of particles; it is assumed all particles have the same mass
        # velocities (array): velocities of particles, assumed to have shape (N, 3)
        with torch.no_grad():
            ke = self.kinetic()
            temperature = ke * self.inv_three_over_two_kb * self.inv_number_of_atoms_
        return temperature

    def kinetic(self):
        # m (float): mass of particles
        # v (array): velocities of particles, assumed to be a 2D array of shape (N, 3)
        # float: total kinetic energy
        with torch.no_grad():
            ke = (self.atom_mass_*(self.atom_velocity_**2).sum(dim = 1)).sum()*0.5
        return ke

    def advance(self):
        force, pe = self.force_potential()
        with torch.no_grad():
            acceleration = force * self.inverse_atom_mass_.unsqueeze(-1)
            vel_half = self.atom_velocity_ + 0.5*self.dt*acceleration
            self.atom_position_ += self.dt*vel_half
        force, pe = self.force_potential()
        with torch.no_grad():
            acceleration = force * self.inverse_atom_mass_.unsqueeze(-1)
            self.atom_velocity_ = vel_half + 0.5*self.dt*acceleration
        return pe
    
    def remove_drift(self):
        with torch.no_grad():
            velocity_drift = (self.atom_mass_.unsqueeze(-1)*self.atom_velocity_).mean(dim = 0) * self.inverse_of_sum_of_atom_mass_
            self.atom_velocity_ -= velocity_drift

    def thermostat(self, current_temperature, target_temperature):
        with torch.no_grad():
            rescale_ratio = (target_temperature/current_temperature)**0.5
            self.atom_velocity_ *= rescale_ratio   

    def run_md(self, number_of_steps = 1, temperature = 1.0, output_dir = ".", dump_interval = 1):
        
        target_temperature = self.to_tensor(temperature).to(self.device)
        fout = open(output_dir + "/md-log", "w")
        self.open_dump_file(filename = output_dir+"/traj.dump")
        
        traj = self.atom_position_.clone().detach().unsqueeze(0)

        for i_step in range(number_of_steps):
            pe = self.advance()
            current_temperature = self.get_temperature()

            print("%d %e %e"%(i_step, current_temperature, pe))
            fout.write("%d %e %e\n"%(i_step, current_temperature, pe))
            if i_step % dump_interval == 0:
                self.dump_frame()
                traj = torch.vstack((traj,self.atom_position_.clone().detach().unsqueeze(0)))
            if i_step % 10 == 0:
                self.remove_drift()
                if i_step < 3000:
                    self.thermostat(current_temperature, target_temperature)

        fout.close()
        self.close_dump_file()
        return traj
    
    def replicate(self, rx, ry, rz):
        new_atom_position = self.atom_position_.clone().detach()
        new_atom_type = self.atom_type_.clone().detach()
        new_atom_mass = self.atom_mass_.clone().detach()
        for i_dim, r in enumerate([rx, ry, rz]):
            replicate_atom_position = new_atom_position.clone().detach()
            for i_r in range(r-1):
                replicate_atom_position += self.lattice_vector_[i_dim]
                new_atom_position = torch.vstack((new_atom_position, replicate_atom_position.clone().detach()))
            new_atom_type = new_atom_type.tile((r, ))
            new_atom_mass = new_atom_mass.tile((r, ))
            self.lattice_vector_[i_dim] *= r
            self.inverse_lattice_vector_ = self.lattice_vector_.inverse()
        self.atom_position_ = new_atom_position.clone().detach()
        self.atom_type_ = new_atom_type.clone().detach()
        self.atom_mass_ = new_atom_mass.clone().detach()
        self.number_of_atoms_ = len(self.atom_position_)
        self.inv_number_of_atoms_ = self.to_tensor(1.0/self.number_of_atoms_).to(self.device)
        self.set_pair_table()

    def rerun_trajectory(self, trajectory):
        pe_trajectory = []
        for a_atom_position in trajectory:
            pe = self.potential(a_atom_position)
            pe_trajectory.append(pe)
        return pe_trajectory

    def rerun_frame(self, atom_position, lattice_vector = None):
        self.atom_position_          = self.to_tensor(atom_position).to(self.device)
        if lattice_vector:
            self.lattice_vector_         = self.to_tensor(lattice_vector).to(self.device)
            self.inverse_lattice_vector_ = self.lattice_vector_.inverse().to(self.device)
        force, pe = self.force_potential()
        #atomic_pe = self.atomic_potential()
        return atomic_pe

    # def scan_ad(self):
    #     j = 1
    #     # field = lambda x: self.forcefield.evaluate_energy(x, self.lattice_vector_, self.inverse_lattice_vector_, self.pair_table_)
    #     field = lambda x: self.forcefield.evaluate_energy(torch.vstack((x, self.atom_position_[j:])), self.lattice_vector_, self.inverse_lattice_vector_, self.pair_table_)
    #     self.ad = AscentDynamics(functional = field, epsilon = 0.001, tol = 1e-9)

    #     ss = torch.linspace(0, 1, steps=200)[:-1].to(self.device)
    #     fout = open("scan.txt","w")
    #     fout.write("%e %e %e\n"%(self.atom_position_[j-1][0].item(),self.atom_position_[j-1][1].item(),self.atom_position_[j-1][2].item()))
    #     for ix in range(len(ss)):
    #         for iy in range(len(ss)):
    #             self.atom_position_[j-1][0] = ss[ix]*self.lattice_vector_[0][0]
    #             self.atom_position_[j-1][1] = ss[iy]*self.lattice_vector_[1][1]
    #             H = self.ad.compute_hessian(self.atom_position_[:j])
    #             eigenvalues, eigenvectors = self.ad.diagnalize(H)
    #             force = self.ad.compute_force(self.atom_position_[:j])
    #             pe = field(self.atom_position_[:j])
    #             print(self.atom_position_[:j])
    #             fout.write("%e %e %e %e %e %e %e %e %e %e\n"%(self.atom_position_[j-1][0].item(),\
    #                                                             self.atom_position_[j-1][1].item(),\
    #                                                             self.atom_position_[j-1][2].item(),\
    #                                                             pe.item(),\
    #                                                             force[0].item(),\
    #                                                             force[1].item(),\
    #                                                             force[2].item(),\
    #                                                             eigenvalues[0].item(),\
    #                                                             eigenvalues[1].item(),\
    #                                                             eigenvalues[2].item()))
    #     exit()

    #     timer = Clock()
    #     traj = self.ad.Dscent(self.atom_position_[:j])
    #     self.atom_position_ = torch.vstack((traj[-1].clone().detach(), self.atom_position_[j:]))
    #     traj = self.ad.Ascent(self.atom_position_[:j], s = 2, device = self.device)
    #     self.atom_position_ = torch.vstack((traj[-1].clone().detach(), self.atom_position_[j:]))
    #     exit()
    #     timer = Clock()
    #     H = self.ad.compute_hessian(self.atom_position_[:j])
    #     print(timer.get_dt(), H.shape)
    #     eigenvalues, eigenvectors = self.ad.diagnalize(H)
    #     print(timer.get_dt())
    #     loss = self.ad.compute_diagnalization_loss(H, eigenvalues, eigenvectors)
    #     print(loss)

    #     for i, a in enumerate(eigenvalues):
    #         print(i, a.item())
            
        