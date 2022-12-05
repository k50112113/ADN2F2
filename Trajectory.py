import torch 

class TrajectoryDataset(torch.utils.data.Dataset):
  def __init__(self, atom_position, lattice_vector, inverse_lattice_vector, energy, atom_force, device = 'cpu'):
        self.atom_position_ = atom_position.to(device)
        self.lattice_vector_ = lattice_vector.to(device)
        self.inverse_lattice_vector_ = inverse_lattice_vector.to(device)
        self.energy_ = energy.to(device)
        self.atom_force_ = atom_force.to(device)

  def __len__(self):
        return len(self.atom_position_)

  def __getitem__(self, index):
        return self.atom_position_[index], self.lattice_vector_[index], self.inverse_lattice_vector_[index], self.energy_[index], self.atom_force_[index]

class DFTTrajectory:
    def __init__(self, input_dir = ["."], start_frame = [0], number_of_frames = [-1]):
        print("\tLoading trajectories from %s ..."%(input_dir))
        atom_position = []
        for index, i_input_dir in enumerate(input_dir):
            with open(i_input_dir + "/coord.raw") as fin:
                i_line_read = 0
                for i_line, aline in enumerate(fin):
                    if i_line >= start_frame[index]:
                        linelist = aline.strip().split()
                        atom_position.append([float(i) for i in linelist])
                        i_line_read += 1
                        if number_of_frames[index] > -1 and i_line_read == number_of_frames[index]: break
        self.number_of_frames_ = len(atom_position)
        self.number_of_atoms_ = len(atom_position[0])//3
        self.atom_position_ = torch.reshape(torch.tensor(atom_position), (-1,self.number_of_atoms_,3))
        
        atom_force = []
        for index, i_input_dir in enumerate(input_dir):
            with open(i_input_dir + "/force.raw") as fin:
                i_line_read = 0
                for i_line, aline in enumerate(fin):
                    if i_line >= start_frame[index]:
                        linelist = aline.strip().split()
                        atom_force.append([float(i) for i in linelist])
                        if number_of_frames[index] > -1 and i_line_read == number_of_frames[index]: break
        self.atom_force_ = torch.reshape(torch.tensor(atom_force), (-1,self.number_of_atoms_,3))
        
        lattice_vector = []
        for index, i_input_dir in enumerate(input_dir):
            with open(i_input_dir + "/box.raw") as fin:
                i_line_read = 0
                for i_line, aline in enumerate(fin):
                    if i_line >= start_frame[index]:
                        linelist = aline.strip().split()
                        lattice_vector.append([float(i) for i in linelist])
                        if number_of_frames[index] > -1 and i_line_read == number_of_frames[index]: break
        self.lbox_rect_ = torch.index_select(torch.reshape(torch.tensor(lattice_vector), (-1,9)), 1, torch.tensor([0,4,8]))
        self.lattice_vector_ = torch.tensor(lattice_vector).reshape((-1,3,3)).transpose(-2, -1)
        self.inverse_lattice_vector_ = self.lattice_vector_.inverse()
        
        energy = []
        for index, i_input_dir in enumerate(input_dir):
            with open(i_input_dir + "/energy.raw") as fin:
                i_line_read = 0
                for i_line, aline in enumerate(fin):
                    if i_line >= start_frame[index]:
                        energy.append(float(aline.strip()))
                        if number_of_frames[index] > -1 and i_line_read == number_of_frames[index]: break
        self.energy_ = torch.tensor(energy)  

        atom_type = []
        with open(input_dir[0] + "/type.raw") as fin:
            linelist = fin.readline().strip().split()
            atom_type = [int(i) for i in linelist]
        self.atom_type_ = torch.tensor(atom_type, dtype=torch.uint8)
        self.type_list_ = [i.item() for i in torch.unique(self.atom_type_)]
        self.number_of_types_ = len(self.type_list_)
        
        self.type_indices_ = []
        for i_type in range(self.number_of_types_):
            self.type_indices_.append((self.atom_type_ == self.type_list_[i_type]).nonzero(as_tuple=False).squeeze())

        self.atom_mass_ = torch.zeros(len(self.atom_type_))
        self.atom_name_list_ = {}
        with open(input_dir[0] + "/atom.raw") as fin:
            for aline in fin:
                linelist = aline.strip().split()
                this_type = int(linelist[0])
                this_name = linelist[1]
                this_mass = float(linelist[2])
                self.atom_name_list_[this_type] = this_name
                self.atom_mass_[self.type_indices_[self.type_list_.index(this_type)]] = this_mass

        # self.type_index_list_ = torch.arange(self.number_of_types_, dtype=torch.uint8)
        # self.type_index_pair_list_ = torch.combinations(self.type_index_list_, with_replacement=True)
        # self.number_of_type_pairs_ = self.type_index_pair_list_.shape[0]
        # self.type_pair_indices_ = []
        # for i_type_pair in range(self.number_of_type_pairs_):
        #     self.type_pair_indices_.append([self.type_indices_[self.type_index_pair_list_[i_type_pair][0]], self.type_indices_[self.type_index_pair_list_[i_type_pair][1]]])

        # self.timer = Clock()
        print("\t\ttotal number of frames:\t%d"%(self.number_of_frames_))
        print("\t\ttotal number of atoms:\t%d"%(self.number_of_atoms_))
        print("\t\ttotal number of types:\t%d"%(self.number_of_types_))
        print("\t\tlist of atom types:\t",end="")
        for a_type in self.type_list_:
            print("%s\t"%(a_type),end="")
        print("")
        print("\t\tnumber of atoms:\t",end="")
        for i_type in range(self.number_of_types_):
            print("%s\t"%(len(self.type_indices_[i_type])),end="")
        print("")
        
        # print("\tcomputing pair displacements...", end=" ")
        # self.compute_pair_displacement()
        # print("%.3f s"%(self.timer.get_dt()))
        # print("\tcomputing pair distances...", end=" ")
        # self.compute_pair_distance()
        # print("%.3f s"%(self.timer.get_dt()))
        # print("\tcomputing triplet angles...", end=" ")
        # self.compute_triplet_angle()
        # print("%.3f s"%(self.timer.get_dt()))

        # self.number_of_atoms_          # N
        # self.atom_position_            # (F*N*Dim)
        # self.lattice_vector_           # (F*Dim*Dim)
        # self.inverse_lattice_vector_   # (F*Dim*Dim)
        # self.atom_type_                # (N)
        # self.number_of_types_          # m
        # self.type_list_                # (m)  e.g. [0,1,2,3] or [0,1,3]
        # self.type_indices_             # [(N1), (N2), (N3), ... (Nm) ] N1 + N2 + N3 ... + Nm = N

        # self.type_index_list_          # (m)  e.g. [0,1,2,3] or [0,1,2]
        # self.number_of_type_pairs_     # mp
        # self.type_index_pair_list_     # (mp*2) e.g. [[0,0],[0,1],[0,2]]
        # self.type_pair_indices_        # [[(N1), (N1)], [(N1), (N2)], ... [(Nm), (Nm)]], mp pairs
    