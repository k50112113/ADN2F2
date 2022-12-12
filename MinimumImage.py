import torch

torch.set_default_tensor_type(torch.FloatTensor)

def minimum_image(rij, lattice_vector, inverse_lattice_vector):
    #rij:                     (F*N*N*Dim) or (N*N*Dim) or (N*Dim)
    #lattice_vector:          (F*Dim*Dim) or (Dim*Dim) or (Dim*Dim)
    #inverse_lattice_vector: (F*Dim*Dim) or (Dim*Dim) or (Dim*Dim)
    #return:                  (F*Dim*Dim) or (N*N*Dim) or (1*N*Dim)
    return rij - torch.matmul(lattice_vector.unsqueeze(-3), torch.matmul(inverse_lattice_vector.unsqueeze(-3), rij.transpose(-2, -1)).round()).transpose(-2, -1)

def minimum_image_rect(rij, lbox):
    #rij:  (F*N*Dim) or (N*Dim)
    #lbox: (F*  Dim) or (  Dim)
    return rij - lbox.unsqueeze(-2)*(rij/lbox.unsqueeze(-2)).round()

def scale_position(rij, inverse_lattice_vector):
    #rij:                     (N*Dim)
    #inverse_lattice_vector: (Dim*Dim)
    #return:                  (N*Dim)
    return torch.matmul(inverse_lattice_vector, rij.transpose(-2, -1)).transpose(-2, -1)

def wrap(r, lbox):
    #wrap 0~L box
    return r%lbox