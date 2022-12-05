import torch
import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
import numpy as np
import os
import sys
import dill
# import pickle
import warnings
from Trajectory import DFTTrajectory, TrajectoryDataset
from NeuralNetworkForceFieldModel import NeuralNetworkForceFieldModel, AtomicSubNetwork
from PairTable import PairTable
from Clock import Clock
warnings.filterwarnings("ignore")

torch.set_default_tensor_type(torch.FloatTensor)
# torch.autograd.set_detect_anomaly(True)
# grads = {}
# def save_grad(name):
#     def hook(grad):
#         grads[name] = grad
#     return hook
# atomic_energy.register_hook(save_grad('atomic_%s'%(j_type)))

class N2F2:
    def __init__(self, input_filename):
        self.input_filename = input_filename
        self.read_input_file()

        ###################################### Load basic information ######################################
        print(self.settings_["mode"])
        self.output_dir = self.settings_["output_dir"]
        self.output_log = self.settings_["output_log"]
        if os.path.isdir(self.output_dir) == False: os.mkdir(self.output_dir)
        self.original_stdout = sys.stdout 
        if self.output_log == "stdout": self.logfile = self.original_stdout
        else:                           self.logfile = open(self.output_dir+"/"+self.output_log, "w")
        sys.stdout = self.logfile
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.random_split_seed              = 42 if not self.settings_.get("random_split_seed") else int(self.settings_["random_split_seed"])
        self.batch_size                     = int(self.settings_["batch_size"])
        self.test_ratio                     = float(self.settings_["test_ratio"])
        self.atomic_subnetwork_map_         = {} # (i_globaltype, j_bootstrap) -> atomic subnetwork
        self.bp_descriptor_                 = None
        ###################################### Load basic information ######################################

        self.timer = Clock()

        if self.settings_["mode"] == "train": self.init_training()
        elif self.settings_["mode"] == "test": self.init_test()

        ###################################### Load DFT trajectories ######################################
        print("Loading trajectories...")
        self.training_data_dir_list_ = self.settings_["training_data_dir_list"].strip().split(',')
        start_frame = self.settings_.get("start_frame")
        number_of_frames = self.settings_.get("number_of_frames")
        for training_index in range(len(self.training_data_dir_list_)): self.training_data_dir_list_[training_index] = self.training_data_dir_list_[training_index].strip().split()
        if start_frame is None:
            start_frame = [[0]*len(i) for i in self.training_data_dir_list_]
        elif " " not in start_frame.strip() and "," not in start_frame.strip():
            try:
                start_frame = int(start_frame)
                start_frame = [[start_frame]*len(i) for i in self.training_data_dir_list_]
            except:
                print("Error: start_frame format not recongnized")
                exit()
        else:
            try:
                start_frame = start_frame.strip().split(',')
                for training_index in range(len(self.training_data_dir_list_)): start_frame[training_index] = [int(i) for i in start_frame[training_index].strip().split()]
                if len(start_frame) != len(self.training_data_dir_list_): raise Exception()
                for training_index in range(len(self.training_data_dir_list_)):
                    if len(start_frame[training_index]) != len(self.training_data_dir_list_[training_index]): raise Exception()
            except:
                print("Error: start_frame should be either one single integer or should have the same data size as training_data_dir_list")     
                exit()
        if number_of_frames is None:
            number_of_frames = [[-1]*len(i) for i in self.training_data_dir_list_]
        elif " " not in number_of_frames.strip() and "," not in number_of_frames.strip():
            try:
                number_of_frames = int(number_of_frames)
                number_of_frames = [[number_of_frames]*len(i) for i in self.training_data_dir_list_]
            except:
                print("Error: number_of_frames format not recongnized")
                exit()
        else:
            try:
                number_of_frames = number_of_frames.strip().split(',')
                for training_index in range(len(self.training_data_dir_list_)): number_of_frames[training_index] = [int(i) for i in number_of_frames[training_index].strip().split()]
                if len(number_of_frames) != len(self.training_data_dir_list_): raise Exception()
                for training_index in range(len(self.training_data_dir_list_)):
                    if len(number_of_frames[training_index]) != len(self.training_data_dir_list_[training_index]): raise Exception()
            except:
                print("Error: number_of_frames should be either one single integer or should have the same data size as training_data_dir_list")     
                exit()
        self.load_training_data(start_frame = start_frame, number_of_frames = number_of_frames)
        print("Complete %.3f s\n"%(self.timer.get_dt()))
        ###################################### Load DFT trajectories ######################################
        
        ###################################### Create Neural-Network ForceField Models ######################################
        self.timer.get_dt()
        print("Creating Neural-Network ForceField Models...")
        self.create_nnff_model()
        self.print_atomic_subnetwork_map()
        print("Complete %.3f s\n"%(self.timer.get_dt()))
        ###################################### Create Neural-Network ForceField Models ######################################
        
        self.apply_loss()

        # loss_test, loss_f_test, loss_e_test = self.test(0, 0)
        # print("%12.3e%12.3e%12.3e"%(loss_test, loss_f_test, loss_e_test))
        # print(self.atomic_subnetwork_map_[0].linear_layer[-1].weight)
        # print(self.nnff_model_list_[0].atomic_subnetwork_[0].linear_layer[-1].weight)
        sys.stdout = self.original_stdout

    def read_input_file(self):
        self.settings_ = {}
        with open(self.input_filename,"r") as fin:
            for aline in fin:
                if "#" not in aline:
                    linelist = aline.strip().split("=")
                    if len(linelist) > 1:
                        self.settings_[linelist[0].strip()] = linelist[1].strip()
    
    def init_test(self):
        self.number_of_epochs_            = None
        self.learning_rate_start          = None
        self.learning_rate_end            = None
        self.energy_loss_prefactor_table_ = torch.tensor([0.0]).to(self.device)
        self.force_loss_prefactor_table_  = torch.tensor([0.0]).to(self.device)
        self.weight_decay                 = None
        print("Starting a NNFF testing on device %s"%(self.device))
        print("")
        
    def init_training(self):
        self.number_of_epochs_   = int(self.settings_["number_of_epochs"])
        self.learning_rate_start = torch.tensor(float(self.settings_["learning_rate_start"]))
        if self.settings_.get("learning_rate_end"): self.learning_rate_end = torch.tensor(float(self.settings_["learning_rate_end"]))
        else:                                       self.learning_rate_end = torch.tensor(float(self.settings_["learning_rate_start"]))
        self.learning_rate_lambda_table_ = (torch.cat((torch.logspace(torch.log10(self.learning_rate_start),torch.log10(self.learning_rate_end),self.number_of_epochs_),torch.tensor([0.])))/self.learning_rate_start).to(self.device)
        
        self.energy_loss_prefactor_start = torch.tensor(float(self.settings_["energy_loss_prefactor_start"]))
        if self.settings_.get("energy_loss_prefactor_end"): self.energy_loss_prefactor_end = torch.tensor(float(self.settings_["energy_loss_prefactor_end"]))
        else:                                               self.energy_loss_prefactor_end = torch.tensor(float(self.settings_["energy_loss_prefactor_start"]))
        self.force_loss_prefactor_start = torch.tensor(float(self.settings_["force_loss_prefactor_start"]))
        if self.settings_.get("force_loss_prefactor_end"):  self.force_loss_prefactor_end = torch.tensor(float(self.settings_["force_loss_prefactor_end"]))
        else:                                               self.force_loss_prefactor_end = torch.tensor(float(self.settings_["force_loss_prefactor_start"]))

        self.energy_loss_prefactor_table_ = torch.logspace(torch.log10(self.energy_loss_prefactor_start), torch.log10(self.energy_loss_prefactor_end), self.number_of_epochs_).to(self.device)
        self.force_loss_prefactor_table_ = torch.logspace(torch.log10(self.force_loss_prefactor_start), torch.log10(self.force_loss_prefactor_end), self.number_of_epochs_).to(self.device)
        self.weight_decay                   = float(self.settings_["weight_decay"])

        print("Starting a NNFF training on device %s"%(self.device))
        print("\n\t Number of epochs = %s"%(self.number_of_epochs_))
        print("\t lr = %s ~ %s"%(self.learning_rate_start.item(),self.learning_rate_end.item()))
        print("\t pref_f = %s ~ %s"%(self.force_loss_prefactor_start.item(),self.force_loss_prefactor_end.item()))
        print("\t pref_e = %s ~ %s"%(self.energy_loss_prefactor_start.item(),self.energy_loss_prefactor_end.item()))
        print("\t random split seed = %s"%(self.random_split_seed))
        print("\t batch size = %s"%(self.batch_size))
        print("\t test ratio = %s"%(self.test_ratio))
        print("\t weight decay = %s"%(self.weight_decay))
        print("")

    def load_training_data(self, start_frame, number_of_frames):
        self.train_loader = []
        self.test_loader = []
        self.dft_trajectory_list = []
        self.globaltype_list = []
        for training_index, training_data_dir in enumerate(self.training_data_dir_list_):
            dft_trajectory = DFTTrajectory(input_dir = training_data_dir, start_frame = start_frame[training_index], number_of_frames = number_of_frames[training_index])
            dataset = TrajectoryDataset(dft_trajectory.atom_position_, dft_trajectory.lattice_vector_, dft_trajectory.inverse_lattice_vector_, dft_trajectory.energy_.unsqueeze(-1), dft_trajectory.atom_force_, self.device)
            self.globaltype_list += dft_trajectory.type_list_
            total_n_test_sample        = int(self.test_ratio*dft_trajectory.number_of_frames_)
            total_n_train_sample       = dft_trajectory.number_of_frames_ - total_n_test_sample
            print("\tNumber of training frames = %s"%(total_n_train_sample))
            print("\tNumber of testing frames  = %s"%(total_n_test_sample))
            self.dft_trajectory_list.append(dft_trajectory)
            if total_n_train_sample > 0:
                train_dataset, test_dataset = torch.utils.data.random_split(dataset, [total_n_train_sample, total_n_test_sample], generator=torch.Generator().manual_seed(self.random_split_seed))
                self.train_loader.append(torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size,shuffle=True))
                self.test_loader.append(torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size,shuffle=False))
            else:
                self.test_loader.append(torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size,shuffle=False))
        self.globaltype_list = [i.item() for i in torch.unique(torch.tensor(self.globaltype_list))]
        self.number_of_globaltypes_ = len(self.globaltype_list)
        if max(self.globaltype_list) != self.number_of_globaltypes_ - 1:
            print("Error: atom type should cover 0 ~ number_of_types - 1, but there are only type ",end="")
            for a_type in self.globaltype_list: print("%d "%(a_type),end="")
            print(".")
            exit()
        self.pair_table_list_ = []
        for i_dft_trajectory in self.dft_trajectory_list:
            self.pair_table_list_.append(PairTable(i_dft_trajectory.type_list_, i_dft_trajectory.type_indices_, i_dft_trajectory.number_of_types_))
    
    def create_nnff_model(self):

        Rc = self.settings_.get("Rc")
        Rs_radial_step = self.settings_.get("Rs_radial_step")
        Rs_angular_step = self.settings_.get("Rs_angular_step")
        As_angular_step = self.settings_.get("As_angular_step")
        eta  = self.settings_.get("eta")
        zeta = self.settings_.get("zeta")
        activation_type = self.settings_.get("activation_type")
        if activation_type is None: activation_type = 'silu'
        first_layer_sin = self.settings_.get("first_layer_sin")
        if first_layer_sin is None: first_layer_sin = False
        elif first_layer_sin == 'True': first_layer_sin = True
        else: first_layer_sin = False

        if self.settings_.get("model_dir"):
            print("\tPath to existing atomic-subnetworks is specified, loading atomic-subnetworks...")
            self.model_dir = self.settings_["model_dir"]
            self.load_atomic_subnetworks(self.number_of_globaltypes_)
            self.load_bpdescriptor()
            self.hidden_layer_size_ = []
            for a in self.atomic_subnetwork_map_.keys():
                self.hidden_layer_size_.append(self.atomic_subnetwork_map_[a].get_hidden_layer_size()[1:-1])
        elif self.settings_.get("hidden_layer_size"):
            print("\tHidden layer size is specified, will create new atomic-subnetworks")
            self.settings_["hidden_layer_size"] = self.settings_["hidden_layer_size"].split(',')
            if len(self.settings_["hidden_layer_size"]) == 1:
                self.settings_["hidden_layer_size"] = self.settings_["hidden_layer_size"]*self.number_of_globaltypes_
            elif len(self.settings_["hidden_layer_size"]) != self.number_of_globaltypes_:
                print("Error: the number of hidden layer sizes should be either 1 or the total number of types.")
                exit()
            self.hidden_layer_size_ = [[int(i_layer) for i_layer in i_hidden_layer_size.strip().split()] for i_hidden_layer_size in self.settings_["hidden_layer_size"]]
            print("\tActivation function: %s"%(activation_type))
        else:
            print("Error: neither existing model directory nor hidden layer size is specified.")
            exit()

        if Rc and Rs_radial_step and Rs_angular_step and As_angular_step and eta and zeta:
            Rc = float(Rc)
            Rs_radial_step = float(Rs_radial_step)
            Rs_angular_step = float(Rs_angular_step)
            As_angular_step = float(As_angular_step)
            eta = float(eta)
            zeta = float(zeta)
        elif self.bp_descriptor_ is None:
            print("Error: neither existing BP Descriptor nor parameters for a BP Descriptor is specified.")
            exit()

        self.nnff_model_list_ = []
        for training_index in range(len(self.training_data_dir_list_)):
            atomic_subnetwork = []
            for j_type in range(len(self.dft_trajectory_list[training_index].type_list_)):
                atomic_subnetwork.append(self.atomic_subnetwork_map_.get(self.dft_trajectory_list[training_index].type_list_[j_type]))
            nnff_model = NeuralNetworkForceFieldModel(self.pair_table_list_[training_index],\
                                                      hidden_layer_size =  self.hidden_layer_size_, \
                                                      atomic_subnetwork = atomic_subnetwork, \
                                                      bp_descriptor = self.bp_descriptor_, \
                                                      Rc = Rc, \
                                                      Rs_radial_step  = Rs_radial_step , \
                                                      Rs_angular_step = Rs_angular_step, \
                                                      As_angular_step = As_angular_step, \
                                                      eta = eta, \
                                                      zeta = zeta, \
                                                      activation_type = activation_type, \
                                                      first_layer_sin = first_layer_sin, \
                                                      device = self.device)
            for j_type in range(len(self.dft_trajectory_list[training_index].type_list_)):
                if atomic_subnetwork[j_type] is None:
                    self.atomic_subnetwork_map_[self.dft_trajectory_list[training_index].type_list_[j_type]] = nnff_model.atomic_subnetwork_[j_type]
            nnff_model.unfreeze_model()
            self.nnff_model_list_.append(nnff_model)
        self.bp_descriptor_ = self.nnff_model_list_[0].bp_descriptor_
    
    def print_atomic_subnetwork_map(self):
        print("\tStructures of the Atomic Sub-networks:")
        for i_globaltype in self.atomic_subnetwork_map_.keys():
            print("\t\ttype%s: "%(i_globaltype),end="")
            hidden_layer_size = self.atomic_subnetwork_map_[i_globaltype].get_hidden_layer_size()
            for a_layer in hidden_layer_size:
                print("%d"%(a_layer),end="\t")
            print("")

    def apply_loss(self):
        self.loss_function = nn.MSELoss(reduction='mean')

    def apply_optimize(self, method='adam'):
        self.optimization = []
        self.learning_rate_schedule = []
        for training_index in range(len(self.training_data_dir_list_)):
            if method == 'adam':
                self.optimization.append(torch.optim.Adam(self.nnff_model_list_[training_index].parameters(), lr=self.learning_rate_start, weight_decay = self.weight_decay))
            elif method == 'sgd':
                self.optimization.append(torch.optim.SGD(self.nnff_model_list_[training_index].parameters(), lr=self.learning_rate_start, momentum = 0.3 ,dampening = 0.01, weight_decay = self.weight_decay))
            self.learning_rate_schedule.append(torch.optim.lr_scheduler.LambdaLR(self.optimization[-1], lr_lambda=lambda epoch: self.learning_rate_lambda_table_[epoch]))

    def compute_loss(self, model_energy, model_force, label_energy, label_force):
        force_loss = self.loss_function(model_force, label_force)
        energy_loss = self.loss_function(model_energy, label_energy)
        return force_loss, energy_loss

    def train_epoch(self, ith_epoch, training_index):
        loss_train = torch.tensor(0.)
        loss_f_train = torch.tensor(0.)
        loss_e_train = torch.tensor(0.)

        inv_n_atom = torch.tensor(1.0/self.dft_trajectory_list[training_index].number_of_atoms_).to(self.device)
        total_n_frames = 0
        # self.optimization[training_index].zero_grad()
        for atom_position, lattice_vector, inverse_lattice_vector, label_energy, label_force in self.train_loader[training_index]:
            # atom_position = atom_position.to(self.device)
            # lattice_vector = lattice_vector.to(self.device)
            # inverse_lattice_vector = inverse_lattice_vector.to(self.device)
            # label_energy = label_energy.to(self.device)
            # label_force = label_force.to(self.device)

            self.optimization[training_index].zero_grad()

            model_force = []
            model_energy = []
            batch_size = len(atom_position)
            total_n_frames += batch_size
            for n_frame in range(batch_size):
                model_force_n, model_energy_n = self.nnff_model_list_[training_index].evaluate_force(atom_position[n_frame], lattice_vector[n_frame], inverse_lattice_vector[n_frame], self.pair_table_list_[training_index], create_graph = True)
                model_force.append(model_force_n)
                model_energy.append(model_energy_n)
            model_force = torch.stack(model_force)
            model_energy = torch.stack(model_energy).unsqueeze(-1)
            loss_f, loss_e = self.compute_loss(model_energy, model_force, label_energy, label_force)
            loss_e *= inv_n_atom
            loss = self.force_loss_prefactor_table_[ith_epoch]*loss_f + self.energy_loss_prefactor_table_[ith_epoch]*loss_e
            loss.backward()

            # print("hook results")
            # print(grads.keys())
            # print(f"{torch.sum(grads['model_force'])}")
            # print(f"{torch.sum(grads['model_energy'])}")
            # print(f"{torch.sum(grads['loss_f'])}")
            # print(f"{torch.sum(grads['loss_e'])}")
            self.optimization[training_index].step()

            loss_train += loss.detach().cpu()*batch_size
            loss_f_train += loss_f.detach().cpu()*batch_size
            loss_e_train += loss_e.detach().cpu()*batch_size
        
        return loss_train/total_n_frames, loss_f_train/total_n_frames, loss_e_train/total_n_frames

    def test(self, ith_epoch, training_index):
        loss_test = torch.tensor(0.)
        loss_f_test = torch.tensor(0.)
        loss_e_test = torch.tensor(0.)
        #return loss_test, loss_f_test, loss_e_test
        inv_n_atom = torch.tensor(1.0/self.dft_trajectory_list[training_index].number_of_atoms_).to(self.device)
        total_n_frames = 0
        for atom_position, lattice_vector, inverse_lattice_vector, label_energy, label_force in self.test_loader[training_index]:
            # atom_position = atom_position.to(self.device)
            # lattice_vector = lattice_vector.to(self.device)
            # inverse_lattice_vector = inverse_lattice_vector.to(self.device)
            # label_energy = label_energy.to(self.device)
            # label_force = label_force.to(self.device)

            model_force = []
            model_energy = []
            batch_size = len(atom_position)
            total_n_frames += batch_size
            for n_frame in range(batch_size):
                # model_energy = self.nnff_model_list_[training_index].evaluate_energy(atom_position[n_frame], lattice_vector[n_frame], inverse_lattice_vector[n_frame])
                model_force_n, model_energy_n = self.nnff_model_list_[training_index].evaluate_force(atom_position[n_frame], lattice_vector[n_frame], inverse_lattice_vector[n_frame], self.pair_table_list_[training_index])
                model_force.append(model_force_n)
                model_energy.append(model_energy_n)
            model_force = torch.stack(model_force)
            model_energy = torch.stack(model_energy).unsqueeze(-1)
            loss_f, loss_e = self.compute_loss(model_energy, model_force, label_energy, label_force)
            loss_e *= inv_n_atom
            loss = self.force_loss_prefactor_table_[ith_epoch]*loss_f + self.energy_loss_prefactor_table_[ith_epoch]*loss_e
            
            loss_test += loss.detach().cpu()*batch_size
            loss_f_test += loss_f.detach().cpu()*batch_size
            loss_e_test += loss_e.detach().cpu()*batch_size

        return loss_test/total_n_frames, loss_f_test/total_n_frames, loss_e_test/total_n_frames

    def train(self):
        sys.stdout = self.logfile
        training_index = 0
        self.apply_optimize()

        print("%5s%12s%12s%12s%12s%12s%12s%12s%12s%12s%12s\n"%("epoch", "l2", "l2_t", "l2f", "l2f_t", "l2e", "l2e_t", "pref_f", "pref_e", "lr", "time"),end="")
        
        loss_test, loss_f_test, loss_e_test = self.test(self.number_of_epochs_ - 1, training_index)
        print("%5d%12s%12.3e%12s%12.3e%12s%12.3e%12s%12s%12s"%(0, \
                                                               "-", loss_test.item(), \
                                                               "-", loss_f_test.item(), \
                                                               "-", loss_e_test.item(), \
                                                               "-", "-", \
                                                               "-"), end="")
        print("")
        for ith_epoch in range(self.number_of_epochs_):
            self.timer.get_dt()
            loss_train, loss_f_train, loss_e_train = self.train_epoch(ith_epoch, training_index)
            torch.cuda.synchronize()
            loss_test, loss_f_test, loss_e_test = self.test(ith_epoch, training_index)
            self.learning_rate_schedule[0].step()

            print("%5d%12.3e%12.3e%12.3e%12.3e%12.3e%12.3e%12.3e%12.3e%12.3e"%(ith_epoch+1, \
                                                                    loss_train.item(), loss_test.item(), \
                                                                    loss_f_train.item(), loss_f_test.item(), \
                                                                    loss_e_train.item(), loss_e_test.item(), \
                                                                    self.force_loss_prefactor_table_[ith_epoch], self.energy_loss_prefactor_table_[ith_epoch], \
                                                                    self.optimization[training_index].param_groups[0]["lr"].item()), end="")
            print("%12.3f"%(self.timer.get_dt()))
        loss_test, loss_f_test, loss_e_test = self.test(self.number_of_epochs_ - 1, training_index)
        print("%5s%12s%12.3e%12s%12.3e%12s%12.3e%12s%12s%12s"%("Final", \
                                                               "-", loss_test.item(), \
                                                               "-", loss_f_test.item(), \
                                                               "-", loss_e_test.item(), \
                                                               "-", "-", \
                                                               "-"), end="")
        print("")
        self.save_atomic_subnetworks()
        self.save_bpdescriptor()
        self.export_model(training_index)
        sys.stdout = self.original_stdout
    
    def save_bpdescriptor(self):
        print("Saving BPDescriptor...")
        fout = open(self.output_dir+"/bp_descriptor.save","wb")
        self.bp_descriptor_.to('cpu')
        dill.dump(self.nnff_model_list_[0].bp_descriptor_, fout)
        self.bp_descriptor_.to(self.device)
        fout.close()

    def load_bpdescriptor(self):
        print("\t\tLoading BPDescriptor...")
        try:
            fin = open(self.model_dir+"/bp_descriptor.save","rb")
            self.bp_descriptor_ = dill.load(fin) #BPDescriptor()
            self.bp_descriptor_.to(self.device)
            fin.close()
        except:
            print("Error: %s not found or not readable."%(self.model_dir+"/bp_descriptor.save"))
            exit()
            
    def save_atomic_subnetworks(self):
        print("Saving Atomic Sub-networks...")
        for i_globaltype in self.atomic_subnetwork_map_.keys():
            network_filename = "network%s.save"%(i_globaltype)
            print("\tsaving %s..."%(network_filename))
            fout = open(self.output_dir+"/"+network_filename,"wb")
            self.atomic_subnetwork_map_[i_globaltype].to('cpu')
            dill.dump(self.atomic_subnetwork_map_[i_globaltype], fout)
            self.atomic_subnetwork_map_[i_globaltype].to(self.device)
            fout.close()
        
    def load_atomic_subnetworks(self,number_of_globaltypes):
        print("\t\tLoading Atomic Sub-networks...")
        self.atomic_subnetwork_map_ = {}
        for i_globaltype in range(number_of_globaltypes):
            network_filename = "network%s.save"%(i_globaltype)
            print("\t\t\tloading %s..."%(network_filename))
            try:
                fin = open(self.model_dir+"/"+network_filename,"rb")
                self.atomic_subnetwork_map_[i_globaltype] = dill.load(fin) #AtomicSubNetwork()
                self.atomic_subnetwork_map_[i_globaltype].to(self.device)
                # self.atomic_subnetwork_map_[i_globaltype] = nn.DataParallel(self.atomic_subnetwork_map_[i_globaltype])
                fin.close()
            except:
                print("Error: %s not found or not readable."%(self.model_dir+"/"+network_filename))
                exit()
    
    def export_model(self, training_index = 0):
        print("Exporting NNFF model...")
        fout = open(self.output_dir+"/model%s.nnff"%(training_index),"wb")
        self.nnff_model_list_[training_index].to('cpu')
        self.nnff_model_list_[training_index].freeze_model()
        dill.dump(self.nnff_model_list_[training_index], fout)
        self.nnff_model_list_[training_index].to(self.device)
        fout.close()
    
    def close(self):
        if self.original_stdout != self.logfile:
            self.logfile.close()
        
    