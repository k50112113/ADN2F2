import torch
from PairTable import PairTable

torch.set_default_tensor_type(torch.FloatTensor)

torchpi = torch.zeros(1).acos() * 2

class BPDescriptor:
    def __init__(self, pair_table, device = 'cpu', \
                 Rc = 6.0, Rs_radial_step = 0.5, Rs_angular_step = 1.5, As_angular_step = 30, eta = 1.0, zeta = 1.0):
        
        Rs_radial = torch.linspace(0., Rc, steps=int(Rc/Rs_radial_step+1))[1:-1]
        Rs_angular = torch.linspace(0., Rc, steps=int(Rc/Rs_angular_step+1))[1:-1]
        As_angular = torch.linspace(0., torchpi.item(), steps=int(180/As_angular_step+1))[1:-1]
        self.pi_ = torchpi
        self.half_ = torch.tensor(0.5)
        self.zero_ = torch.tensor(0.0)
        self.Rs_radial_step_ = Rs_radial_step
        self.Rs_angular_step_ = Rs_angular_step
        self.As_angular_step_ = As_angular_step
        self.eta_ = torch.tensor(eta)
        self.zeta_ = torch.tensor(zeta)
        self.Rc_ = torch.tensor(Rc)
        self.Rc_half_ = self.Rc_/2.
        self.Rc_recp_ = 1.0/self.Rc_
        self.Rs_radial_  = torch.tensor(Rs_radial) #Sr
        self.Rs_angular_ = torch.tensor(Rs_angular)#Sar
        self.As_angular_ = torch.tensor(As_angular)#Saa
        self.descriptor_dimensions = self.Rs_radial_.shape[0]*pair_table.number_of_globaltypes_ + self.Rs_angular_.shape[0]*self.As_angular_.shape[0]*len(pair_table.globaltype_pair_list_)
        
        self.update_descriptor(device, pair_table)
        self.to(device)

    def update_descriptor(self, device, pair_table):
        self.zero_GR = [torch.zeros(len(i), len(self.Rs_radial_)) for i in pair_table.localtype_indices_] #(Ni*Sr)
        self.zero_GA = [torch.zeros(len(i), len(self.Rs_angular_)*len(self.As_angular_)) for i in pair_table.localtype_indices_] #(Ni*(Saa*Sar))
        for i in range(len(self.zero_GR)): self.zero_GR[i] = self.zero_GR[i].to(device)
        for i in range(len(self.zero_GA)): self.zero_GA[i] = self.zero_GA[i].to(device)

    def to(self, device):
        self.eta_  = self.eta_.to(device)
        self.zeta_ = self.zeta_.to(device)
        self.Rc_   = self.Rc_.to(device)
        self.Rc_half_ = self.Rc_half_.to(device)
        self.Rc_recp_ = self.Rc_recp_.to(device)
        self.Rs_radial_  = self.Rs_radial_.to(device)
        self.Rs_angular_ = self.Rs_angular_.to(device) #Sar
        self.As_angular_ = self.As_angular_.to(device) #Saa
        for i in range(len(self.zero_GR)): self.zero_GR[i] = self.zero_GR[i].to(device)
        for i in range(len(self.zero_GA)): self.zero_GA[i] = self.zero_GA[i].to(device)
        self.pi_ = self.pi_.to(device)
        self.zero_ = self.zero_.to(device)

    def compute_descriptor(self, atom_position, lattice_vector, inverse_lattice_vector, pair_table):
        pair_displacement = pair_table.compute_pair_displacement(atom_position, lattice_vector, inverse_lattice_vector)
        pair_distance = pair_table.compute_pair_distance(pair_displacement)
        triplet_angle, triplet_distance_sum = pair_table.compute_triplet_angle(pair_displacement, pair_distance)
        descriptor = []
        for i_localtype in range(pair_table.number_of_localtypes_):
            GRi = []
            for j_globaltype, p_localtype_pair_trans_fg in enumerate(pair_table.localtype_globaltype_to_localtype_pair_list[i_localtype]):
                if p_localtype_pair_trans_fg != None:
                    p_localtype_pair, trans_fg = p_localtype_pair_trans_fg
                    Rij = pair_distance[p_localtype_pair]
                    if trans_fg: Rij = Rij.transpose(-2,-1)
                    fcij = self.symmetry_function(Rij)
                    GRi.append(self.radial_descriptor(Rij, fcij))
                else:
                    GRi.append(self.zero_GR[i_localtype])
            GAi = []
            for p_globaltype_pair, jk_globaltype in enumerate(pair_table.globaltype_pair_list_):
                j_globaltype, k_globaltype = jk_globaltype
                i_globaltype = pair_table.localtype_to_globaltype_list_[i_localtype]
                if pair_table.globaltype_pair_map.get((i_globaltype, j_globaltype)) != None and pair_table.globaltype_pair_map.get((i_globaltype, k_globaltype)) != None:
                    p_localtype_pair_ij, trans_fg_ij = pair_table.globaltype_pair_map[(i_globaltype, j_globaltype)]
                    p_localtype_pair_ik, trans_fg_ik = pair_table.globaltype_pair_map[(i_globaltype, k_globaltype)]
                    p_localtype_pair_jk, _ = pair_table.globaltype_pair_map[(j_globaltype, k_globaltype)]
                    Rij = pair_distance[p_localtype_pair_ij]
                    Rik = pair_distance[p_localtype_pair_ik]
                    if trans_fg_ij: Rij = Rij.transpose(-2,-1)
                    if trans_fg_ik: Rik = Rik.transpose(-2,-1)
                    fcij = self.symmetry_function(Rij).unsqueeze(-1)
                    fcik = self.symmetry_function(Rik).unsqueeze(-2)
                    fcijfcik = fcij + fcik
                    if j_globaltype == k_globaltype:
                        fcijfcik = pair_table.remove_self_triplet(fcijfcik)
                    Aijk = triplet_angle[i_localtype][p_localtype_pair_jk]
                    Rijk = triplet_distance_sum[i_localtype][p_localtype_pair_jk]
                    GAi.append(self.angular_descriptor(Rijk, Aijk, fcijfcik).flatten(-2,-1))
                else:
                    GAi.append(self.zero_GA[i_localtype])
                
            descriptor.append(torch.cat((torch.cat(GAi, dim=-1), torch.cat(GRi, dim=-1)), dim = -1))
        
        # for a in descriptor:
        #     print(a.shape)
        return descriptor

    def symmetry_function(self, Rij):
        #Rij:     (Ni*Nj)
        #return:  (Ni*Nj)
        return torch.where((Rij-self.Rc_half_)**2 < (self.Rc_half_)**2, 0.5 * (self.pi_ * Rij * self.Rc_recp_).cos() + 0.5, self.zero_)

    def radial_descriptor(self, Rij, fcij):
        #Rij:     (Ni*Nj)
        #fcij:    (Ni*Nj)
        #return:  (Ni*Sr)
        Rij_usq = Rij.unsqueeze(-2) #(Ni*1*Nj)
        Rs_radial_usq = self.Rs_radial_.unsqueeze(-1) #(Sr*1)
        fcij_usq = fcij.unsqueeze(-2) #(Ni*1*Nj)
        return ((-self.eta_ * (Rij_usq - Rs_radial_usq)**2).exp() * fcij_usq).sum(dim = -1) #(Ni*Sr)

    def angular_descriptor(self, Rijk, Aijk, fcijfcik):
        #Rijk:     (Ni*Nj*Nk)
        #Aijk:     (Ni*Nj*Nk)
        #fcijfcik: (Ni*Nj*Nk)
        #return:   (Ni*Saa*Sar)

        fcijik_usq = fcijfcik.unsqueeze(-3).unsqueeze(-3) #(Ni*1*1*Nj*Nk)
        Rijk_usq = Rijk.unsqueeze(-3) #(Ni*1*Nj*Nk)
        Rs_angular_usq = self.Rs_angular_.unsqueeze(-1).unsqueeze(-1) #(Sar*1*1)
        radial_component = (-self.eta_ * (0.5 * Rijk_usq - Rs_angular_usq)**2).exp() #(Ni*Sar*Nj*Nk)
        
        Aijk_usq = Aijk.unsqueeze(-3) #(Ni*1*Nj*Nk)
        As_angular_usq = self.As_angular_.unsqueeze(-1).unsqueeze(-1) #(Saa*1*1)
        angular_component = (1.0 + (Aijk_usq - As_angular_usq).cos())**self.zeta_  #(Ni*Saa*Nj*Nk)

        radial_component = radial_component.unsqueeze(-4)   #(Ni*1*Sar*Nj*Nk)
        angular_component = angular_component.unsqueeze(-3) #(Ni*Saa*1*Nj*Nk)
        return 2.0**(1.0 - self.zeta_) * (radial_component*angular_component*fcijik_usq).sum(dim=(-2, -1)) #(Ni*Saa*Sar)
    
    def get_descriptor_dimensions(self):
        return self.descriptor_dimensions 
    
    def print_details(self):
        print("\tBP Descriptor:")
        print("\t\tRc\t\t= %s"%(self.Rc_.item()))
        print("\t\tRs_radial_step\t= %s"%(self.Rs_radial_step_))
        print("\t\tRs_angular_step\t= %s"%(self.Rs_angular_step_))
        print("\t\tAs_angular_step\t= %s"%(self.As_angular_step_))
        print("\t\teta\t= %s"%(self.eta_.item()))
        print("\t\tzeta\t= %s"%(self.zeta_.item()))
        print("")
        print("\t\tRs_r\t%d\t%s"%(self.Rs_radial_.shape[0], self.Rs_radial_.cpu().detach().numpy()))
        print("\t\tRs_a\t%d\t%s"%(self.Rs_angular_.shape[0], self.Rs_angular_.cpu().detach().numpy()))
        print("\t\tTh_a\t%d\t%s"%(self.As_angular_.shape[0], (self.As_angular_*180/self.pi_).cpu().detach().numpy()))   