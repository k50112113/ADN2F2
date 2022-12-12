import torch
from MinimumImage import minimum_image
import warnings
warnings.filterwarnings("ignore")

torch.set_default_tensor_type(torch.FloatTensor)

class PairTable:
    def __init__(self, localtype_to_globaltype_list, localtype_indices, number_of_globaltypes, dimensions=3):
        self.dimensions = dimensions
        self.localtype_to_globaltype_list_ = localtype_to_globaltype_list
        self.localtype_indices_ = localtype_indices
        self.number_of_localtypes_ = len(self.localtype_indices_)
        localtype_pair_list = torch.combinations(torch.arange(self.number_of_localtypes_, dtype=torch.uint8), with_replacement=True)
        self.localtype_pair_list_ = [tuple([i_type.item() for i_type in p_pair]) for p_pair in localtype_pair_list]
        self.number_of_localtype_pairs_ = len(self.localtype_pair_list_)
        self.localtype_pair_map = {} #(i_localtype, j_localtype) -> (p_localtype_pair, transfg)
        for p_localtype_pair, ij_localtype in enumerate(self.localtype_pair_list_):
            self.localtype_pair_map[ij_localtype] = (p_localtype_pair, False)
            i_localtype, j_localtype = ij_localtype
            if i_localtype != j_localtype:
                self.localtype_pair_map[(j_localtype, i_localtype)] = (p_localtype_pair, True)

        self.number_of_globaltypes_ = number_of_globaltypes
        self.globaltype_to_localtype_list_ = [None]*self.number_of_globaltypes_
        for i_localtype, i_globaltype in enumerate(self.localtype_to_globaltype_list_):
            self.globaltype_to_localtype_list_[i_globaltype] = i_localtype
        
        self.globaltype_list_ = [i.item() for i in torch.arange(self.number_of_globaltypes_, dtype=torch.uint8)]
        globaltype_pair_list = torch.combinations(torch.arange(self.number_of_globaltypes_, dtype=torch.uint8), with_replacement=True)
        self.globaltype_pair_list_ = [tuple([i_type.item() for i_type in p_pair]) for p_pair in globaltype_pair_list]
        # print(self.localtype_to_globaltype_list_)
        # print(self.globaltype_list_)
        # print(self.localtype_pair_list_)
        # print(self.globaltype_pair_list_)

        self.globaltype_pair_map = {} #(i_globaltype, j_globaltype) -> (p_localtype_pair, transfg)
        self.localtype_globaltype_to_localtype_pair_list = [[None]*self.number_of_globaltypes_ for i in range(self.number_of_localtypes_)] #i_localtype, j_globaltype -> p_localtype_pair and transfg of (i_localtype, j_globaltype) pair
        for p_globaltype_pair, ij_globaltype in enumerate(self.globaltype_pair_list_):
            i_globaltype, j_globaltype = ij_globaltype
            i_localtype, j_localtype = self.globaltype_to_localtype_list_[i_globaltype], self.globaltype_to_localtype_list_[j_globaltype]
            if i_localtype != None and j_localtype != None:
                self.globaltype_pair_map[ij_globaltype] = self.localtype_pair_map[(i_localtype, j_localtype)]
                self.localtype_globaltype_to_localtype_pair_list[i_localtype][j_globaltype] = self.localtype_pair_map[(i_localtype, j_localtype)]
                if i_localtype != j_localtype:
                    ji_globaltype = j_globaltype, i_globaltype
                    self.globaltype_pair_map[ji_globaltype] = self.localtype_pair_map[(j_localtype, i_localtype)]
                    self.localtype_globaltype_to_localtype_pair_list[j_localtype][i_globaltype] = self.localtype_pair_map[(j_localtype, i_localtype)]

        # for a in self.localtype_pair_map.keys(): print(a, self.localtype_pair_map[a])
        # for a in self.localtype_globaltype_to_localtype_pair_list: print(a)
        # for a in self.globaltype_pair_map.keys(): print(a, self.globaltype_pair_map[a])

    def compute_pair_displacement2(self, atom_position, lattice_vector, inverse_lattice_vector):
        pair_displacement = []
        for i_type, j_type in self.localtype_pair_list_:
            pair_displacement_ij = atom_position[self.localtype_indices_[j_type], :].unsqueeze(-3) - atom_position[self.localtype_indices_[i_type], :].unsqueeze(-2)
            pair_displacement.append(minimum_image(pair_displacement_ij, lattice_vector, inverse_lattice_vector)) #(F*Ni*Nj*Dim)
        return pair_displacement #[(F*N1*N1*Dim), (F*N1*N2*Dim), ... (F*Nm*Nm*Dim)]

    def compute_pair_distance2(self, pair_displacement):
        pair_distance = []
        for p_type_pair in range(self.number_of_localtype_pairs_):
            pair_distance.append(pair_displacement[p_type_pair].norm(dim = -1))
        return pair_distance #[(F*N*N1), (F*N*N2), ... (F*N*Nm)]
    
    def remove_self_pair(self, pij):
        ni = pij.shape[-3]
        pij = pij.flatten()
        pij = pij[self.dimensions:]
        pij = pij.view(ni-1, ni+1, self.dimensions)
        pij = pij[:,:-1,:]
        pij = pij.reshape(ni, ni-1, self.dimensions)
        return pij
    
    def remove_self_triplet(self, pijk):
        ni = pijk.shape[-3]
        nj = pijk.shape[-2]
        pijk = pijk.flatten(-2,-1)
        pijk = pijk[:,1:]
        pijk = pijk.reshape(ni, nj-1, nj+1)
        pijk = pijk[:,:,:-1]
        pijk = pijk.reshape(ni, nj, nj-1)
        return pijk

    def compute_pair_displacement(self, atom_position, lattice_vector, inverse_lattice_vector):
        pair_displacement = []
        for i_localtype, j_localtype in self.localtype_pair_list_:
            pair_displacement_ij = atom_position[self.localtype_indices_[j_localtype], :].unsqueeze(-3) - atom_position[self.localtype_indices_[i_localtype], :].unsqueeze(-2)
            pair_displacement_ij = minimum_image(pair_displacement_ij, lattice_vector, inverse_lattice_vector) #(F*Ni*Nj*Dim)
            if i_localtype == j_localtype: pair_displacement_ij = self.remove_self_pair(pair_displacement_ij)
            pair_displacement.append(pair_displacement_ij)
        return pair_displacement #[..., (Ni*Nj*Dim),...] Nj = Nj-1 if i == j else Nj = Nj

    def compute_pair_distance(self, pair_displacement):
        pair_distance = []
        for p_localtype_pair in range(len(self.localtype_pair_list_)):
            pair_distance.append(pair_displacement[p_localtype_pair].norm(dim = -1))
        return pair_distance #[..., (F*Ni*Nj),...] Nj = Nj-1 if i == j else Nj = Nj
    
    def compute_triplet_angle(self, pair_displacement, pair_distance):
        triplet_angle = []
        triplet_distance_sum = []
        for i_localtype in range(self.number_of_localtypes_):
            triplet_angle.append([])
            triplet_distance_sum.append([])
            for j_localtype, k_localtype in self.localtype_pair_list_:
                p_localtype_pair_ij, trans_fg_ij = self.localtype_pair_map[(i_localtype, j_localtype)]
                p_localtype_pair_ik, trans_fg_ik = self.localtype_pair_map[(i_localtype, k_localtype)]
                
                pair_displacement_ij = pair_displacement[p_localtype_pair_ij]
                pair_displacement_ik = pair_displacement[p_localtype_pair_ik]
                pair_distance_ij = pair_distance[p_localtype_pair_ij]
                pair_distance_ik = pair_distance[p_localtype_pair_ik]

                if trans_fg_ij:
                    pair_displacement_ij = pair_displacement_ij.transpose(-3,-2)
                    pair_distance_ij = pair_distance_ij.transpose(-2,-1)
                if trans_fg_ik:
                    pair_displacement_ik = pair_displacement_ik.transpose(-3,-2)
                    pair_distance_ik = pair_distance_ik.transpose(-2,-1)

                pair_displacement_ij = pair_displacement_ij.unsqueeze(-2)
                pair_displacement_ik = pair_displacement_ik.unsqueeze(-3)
                pair_distance_ij = pair_distance_ij.unsqueeze(-1)
                pair_distance_ik = pair_distance_ik.unsqueeze(-2)
                triplet_distance_sum_ijk = pair_distance_ij + pair_distance_ik
                triplet_angle_ijk = (pair_displacement_ij * pair_displacement_ik).sum(dim=-1) / (pair_distance_ij * pair_distance_ik)
                if j_localtype == k_localtype:
                    triplet_angle_ijk = self.remove_self_triplet(triplet_angle_ijk)
                    triplet_distance_sum_ijk = self.remove_self_triplet(triplet_distance_sum_ijk)

                triplet_angle[-1].append(triplet_angle_ijk)
                triplet_distance_sum[-1].append(triplet_distance_sum_ijk)
        return triplet_angle, triplet_distance_sum # [..., (Ni*Nj*Nk), ...] N(j/k) = Ni - 1 if i == j/k