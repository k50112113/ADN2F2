import torch
from Clock import Clock
from MinimumImage import wrap

torch.set_default_tensor_type(torch.DoubleTensor)

def read_ad_log(filename):
    data = []
    with open(dir+"/%s/method%d/ad_log.txt"%(s,imethod),"r") as fin:
        for aline in fin:
            if "tolerance" in aline:
                time = float(fin.readline().strip().split()[1])
                break
            elif "time" in aline:
                time = float(aline.strip().split()[1])
                break
            linelist = aline.strip().split()
            linelist = list(filter(lambda val: val !=  '|', linelist))
            data.append([float(i) for i in linelist])
    data = np.array(data)
    log = {}
    log["index"]             = data[:, 0]
    log["energy"]            = data[:, 1]
    log["alp"]               = data[:, 2]
    log["scale_prefactor"]   = data[:, 3]
    log["dmax"]              = data[:, 4] 
    log["force_len"]         = data[:, 5]
    log["biased_force_len"]  = data[:, 6]
    log["force_s_len"]       = data[:, 7]
    log["force_s_tilde_len"] = data[:, 8]
    log["fmax"]              = data[:, 9]
    log["max_disp"]          = data[:, 10]
    log["eigs"]              = data[:, 11:] # (Nitr*Neig)
    n_eigs                   = []
    for i in log["eigs"]:
        n_eigs.append(np.where(i<0)[0].shape[0])
    log["n_eigs"]            = n_eigs
    log["eigs"]              = log["eigs"].transpose() # (Neig*Nitr)
    log["time"]              = time

    return log

class VariableSubset:
    def __init__(self, x, var_flag):
        self.x_shape = x.shape
        self.var_flag = var_flag
        self.var_indeces = (self.var_flag == 1).nonzero().flatten()
        self.const_indeces = (self.var_flag == 0).nonzero().flatten()

    def separate(self, x):
        var = x.flatten()[self.var_indeces]
        const = x.flatten()[self.const_indeces]
        return var, const
    
    def combine(self, x, var, const):
        x = x.flatten()
        x[self.var_indeces] = var
        x[self.const_indeces] = const
        return x.reshape(self.x_shape)

def separate_last_j_elements_as_variables(x, j):
    original_shape = x.shape
    var = x.flatten()[-j:]
    const = x.flatten()[:-j]
    return var, const, original_shape

def combine_constants_and_variables(var, const, original_shape):
    return torch.cat((const, var)).reshape(original_shape)

def separate_first_j_elements_as_variables(x, j):
    original_shape = x.shape
    var = x.flatten()[:j]
    const = x.flatten()[j:]
    return var, const, original_shape

def combine_variables_and_constants(var, const, original_shape):
    return torch.cat((var, const)).reshape(original_shape)

class AscentDynamics:
    def __init__(self, functional, alpha = 1e-3, etol = 1e-9, ftol = 1e-9, atol = 1e-9, \
                       dmax = 0.02, dtol = 5e-3, alpha_decay_rate = 0.998):
        '''
            x: torch.tensor (dim)
            E: torch.float

            E = self.functional(x)
        '''
        self.functional = functional
        self.alpha = alpha # learning rate
        self.etol = etol   # energy tolerance
        self.ftol = ftol   # foce tolerance
        self.atol = atol   # learning rate tolerance
        self.dmax = dmax   # max allowed displacment
        self.dtol = dtol   # dmax tolerance
        self.alpha_decay_rate = alpha_decay_rate # decay rate for learning rate (used by adpative force contribution method)
        ''' Deprecated, used by old schemes.
        # self.traj_ma_window = 50
        # self.traj_ma_check_interval = 10
        # self.traj_ma_tol = 0.1
        # self.traj_ma_shrink_rate = 0.5
        '''

    def compute_force(self, x):
        '''
        Compute force

        arguments:
            x:  torch.tensor (dim)
        
        return:
            F:  torch.tensor (dim)
        '''
        F = -torch.autograd.functional.jacobian(self.functional, x).squeeze(0)
        return F.detach()

    def compute_force_using_grad(self, x):
        '''
        An alternative way to compute force
        '''
        x.requires_grad = True
        F = -torch.autograd.grad(self.functional(x), x)[0]
        return F.detach()
    
    def compute_diagnalization_loss(self, M, eigenvalues, eigenvectors):
        '''
        Check M * eigenvectors - eigenvalues * eigenvectors = 0
        '''
        return ((torch.mm(M, eigenvectors)-torch.mm(eigenvectors, eigenvalues.diag()))**2).mean()

    def diagnalize(self, M):
        '''
        Diagnalize M and sort the eigenvectors and eigenvalues by the eigenvalues.
        
        arguments:
            M:  torch.tensor (dim * dim)
        
        return:
            eigenvalues:  torch.tensor (N_eig)
            eigenvectors: torch.tensor (dim * N_eig)
        '''
        eigenvalues, eigenvectors = torch.linalg.eig(M) 
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        eigenvalues = eigenvalues.detach()
        eigenvectors = eigenvectors.detach()
        eigenvalues_rank = torch.argsort(eigenvalues, descending = False).detach()
        return eigenvalues[eigenvalues_rank], eigenvectors[:, eigenvalues_rank]

    def compute_hessian(self, x):
        '''
        Compute Hessian.
        
        arguments:
            x:  torch.tensor (dim)
        
        return:
            H:  torch.tensor (dim * dim)
        '''
        n_elements = x.numel()
        H = torch.reshape(torch.autograd.functional.hessian(self.functional, x),(n_elements,n_elements))
        H = 0.5*(H + H.transpose(0,1)) # H has to be symmetric
        return H
    
    # def compute_new_eigen_from_previous_eigen(self, x, previous_eigenvector, previous_eigenvalue):
    #     '''
    #     Compute a new eigenvalue and new eigenvector for a small increment of x.
        
    #     arguments:
    #         x:                      torch.tensor (dim)
    #         previous_eigenvector:   torch.tensor (dim)
    #         previous_eigenvalue:    torch.float
        
    #     return:
    #         new_eigenvector:        torch.tensor (dim)
    #         new_eigenvalue:         torch.float
    #     '''
    #     new_eigenvector = previous_eigenvector.clone().detach()
    #     parm = torch.nn.Parameter(new_eigenvector, requires_grad=True)
    #     optimization = torch.optim.Adam([parm], lr=1e-3)
    #     for i in range(1000):
    #         with torch.no_grad():
    #             parm /= parm.norm()
    #         optimization.zero_grad()
    #         E, Hv = self.compute_hvp(x, parm)
    #         loss = 2*(torch.inner(Hv, parm)/Hv.norm() - 1.0)**2 + (torch.inner(previous_eigenvector, parm)/previous_eigenvalue/parm.norm() - 1.0)**2 + (parm.norm() - previous_eigenvalue)**2
    #         print(loss)
    #         loss.backward()
    #         optimization.step()
        
    #     new_eigenvector = new_eigenvector.detach()
    #     new_eigenvalue = new_eigenvector.norm()
    #     new_eigenvector = new_eigenvector/new_eigenvalue
    #     E, Hv = self.compute_hvp(x, new_eigenvector)

    #     print(previous_eigenvalue, new_eigenvalue)
    #     print(torch.inner(previous_eigenvector, new_eigenvector))
    #     print(torch.inner(Hv, new_eigenvector)/new_eigenvalue , Hv.norm())

    def compute_hvp(self, x, v):
        '''
        Compute Hessian-vector product.
        if v is one of the eigenvectors of H, then Hv = eig * v where eig is the corresponding eigenvalue.

        arguments:
            x:  torch.tensor (dim)
            v:  torch.tensor (dim)
        
        return:
            E:  torch.float
            Hv:tensor (dim)
        '''
        E, Hv = torch.autograd.functional.hvp(self.functional, x, v)
        return E, Hv

    def compute_force_components(self, eigenvectors, force, s):
        '''
        Compute the ascending (force_s) and descending (force_s_tilde) forces.

        arguments:
            eigenvectors:   torch.tensor (dim * N_eig)
            force:          torch.tensor (dim)
            s:              int
        
        return:
            force_s:        torch.tensor (dim)
            force_s_tilde:  torch.tensor (dim)
        '''
        row_s_eigenvectors = eigenvectors.transpose(0,1)[:s]
        force_s = torch.mm(torch.mm(row_s_eigenvectors, force.unsqueeze(-1)).transpose(0,1), row_s_eigenvectors).squeeze(0)
        force_s_tilde = force - force_s
        return force_s, force_s_tilde
    
    def compute_one_force_component(self, eigenvector, force):
        '''
        Compute the ascending (force_s) and descending (force_s_tilde) forces,
        but along only one given eigenvector.

        arguments:
            eigenvector:   torch.tensor (dim)
            force:          torch.tensor (dim)
        
        return:
            force_s:        torch.tensor (dim)
            force_s_tilde:  torch.tensor (dim)
        '''
        force_s = torch.inner(eigenvector, force) * eigenvector
        force_s_tilde = force - force_s
        return force_s, force_s_tilde

    def evaluate_trail_step(self, x, dx):
        '''
        Do a trail step at x + dx.

        arguments:
            x:              torch.tensor (dim)
            dx:             torch.tensor (dim)
        
        return:
            trail_x:        torch.tensor (dim)
            trail_energy:   torch.float
        '''
        trail_x = x + dx
        trail_energy = self.functional(trail_x).detach()
        return trail_x, trail_energy

    def get_scale_prefactor(self, force, alp):
        '''
        Return a scaling factor so that the max allowed displacement for each dimension
        is bounded by self.dmax

        arguments:
            force:              torch.tensor (dim)
            alp:                float
        
        return:
            scale_prefactor:    torch.float or 1.0
            max_disp:           torch.float
        '''
        max_move = force.abs().max()*alp
        if self.dmax is None or max_move < self.dmax: scale_prefactor = 1.0
        else: scale_prefactor = self.dmax/max_move
        max_disp = max_move*scale_prefactor
        return scale_prefactor, max_disp
    
    def adaptive_force_contribution(self, force_s, force_s_tilde):
        force_s_len       = torch.linalg.norm(force_s)
        force_s_tilde_len = torch.linalg.norm(force_s_tilde)
        if force_s_len < force_s_tilde_len:
            force_s_tilde *= force_s_len/force_s_tilde_len
            force_s_tilde_len = force_s_len
        return force_s_len, force_s_tilde_len

    def print_line_ad(self, flog, itr, current_energy, alp, scale_prefactor, dmx, \
                            current_force_len, current_biased_force_len, force_s_len, force_s_tilde_len, abs_force_max, max_disp, current_eigenvalues):
        print("%d | %.10e | %e %e %e | %e %e %e %e | %e %e | "%(\
                    itr, current_energy, \
                    alp, scale_prefactor, dmx, \
                    current_force_len, current_biased_force_len, force_s_len, force_s_tilde_len, \
                    abs_force_max, max_disp \
                    ),end="")
        print("")
        flog.write("%d | %.10e | %e %e %e | %e %e %e %e | %e %e | "%(\
                    itr, current_energy, \
                    alp, scale_prefactor, dmx,\
                    current_force_len, current_biased_force_len, force_s_len, force_s_tilde_len, \
                    abs_force_max, max_disp \
                    ))
        for a in current_eigenvalues:
            flog.write("%.10e "%(a))
        flog.write("\n")

    def Ascent_01(self, x, s, method = 0, max_itr = 20000, output_dir = "./"):
        print("Ascent")
        flog = open(output_dir+"/ad_log.txt", "w")

        alp = self.alpha
        dmx = self.dmax
        first_time_reach_s = False

        self.timer = Clock()

        traj = x.clone().detach().unsqueeze(0)
        current_energy = self.functional(x).detach()
        current_force = self.compute_force(x)
        current_force_len = torch.linalg.norm(current_force)

        current_eigenvalues, current_eigenvectors = self.diagnalize(self.compute_hessian(x))
        force_s, force_s_tilde = self.compute_force_components(current_eigenvectors, current_force, s)
        force_s_len       = torch.linalg.norm(force_s)
        force_s_tilde_len = torch.linalg.norm(force_s_tilde)
        current_biased_force = current_force - 2*force_s
        current_biased_force_len = torch.linalg.norm(current_biased_force)

        scale_prefactor, max_disp = self.get_scale_prefactor(current_biased_force, alp)
        align_sign = torch.inner(current_force, current_biased_force).sign()
        abs_force_max = current_force.abs().max()
        previous_abs_force_max = abs_force_max

        self.print_line_ad(flog, 0, current_energy, alp, scale_prefactor, dmx, current_force_len, current_biased_force_len, force_s_len, force_s_tilde_len, abs_force_max, max_disp, current_eigenvalues)
        
        for i in range(max_itr):

            trail_x, trail_energy = self.evaluate_trail_step(x, scale_prefactor*current_biased_force*alp)
            if (method == 0 and i > 100      and trail_energy < current_energy) or\
               (method == 1 and align_sign < 0 and trail_energy < current_energy):
                while trail_energy < current_energy:
                    alp *= 0.5
                    trail_x, trail_energy = self.evaluate_trail_step(x, scale_prefactor*current_biased_force*alp)
                    print("decrease alpha", alp)
                    if alp < self.atol: break
                if alp < self.atol:
                    print("Alpha tolerance")
                    flog.write("Alpha tolerance\n")
                    break
                
            dE = ((trail_energy-current_energy)/current_energy.abs())
            previous_eigenvalues = current_eigenvalues.clone()
            x = trail_x.clone()
            traj = torch.vstack((traj,x.unsqueeze(0)))
            current_energy = trail_energy
            current_eigenvalues, current_eigenvectors = self.diagnalize(self.compute_hessian(x))
            current_force = self.compute_force(x)
            current_force_len = torch.linalg.norm(current_force)

            force_s, force_s_tilde = self.compute_force_components(current_eigenvectors, current_force, s)
            force_s_len       = torch.linalg.norm(force_s)
            force_s_tilde_len = torch.linalg.norm(force_s_tilde)
            current_biased_force = force_s_tilde - force_s #(force_s + force_s_tilde) - 2*force_s
            current_biased_force_len = torch.linalg.norm(current_biased_force)

            scale_prefactor, max_disp = self.get_scale_prefactor(current_biased_force, alp)
            align_sign = torch.inner(current_force, current_biased_force).sign()
            abs_force_max = current_force.abs().max()

            self.print_line_ad(flog, i+1, current_energy, alp, scale_prefactor, dmx, current_force_len, current_biased_force_len, force_s_len, force_s_tilde_len, abs_force_max, max_disp, current_eigenvalues)
            
            abs_dE = dE.abs()
            if abs_force_max < self.ftol:
                print("force tolerance")
                flog.write("force tolerance: %e\n"%(abs_force_max))
                break
            if alp < self.atol:
                print("alpha tolerance")
                flog.write("alpha tolerance\n")
                break
            if max_disp < self.dtol:
                print("max displacement tolerance")
                flog.write("max displacement tolerance\n")
                break

        current_eigenvalues, current_eigenvectors = self.diagnalize(self.compute_hessian(x))
        dt = self.timer.get_dt()
        print("time: %s s"%(dt))
        flog.write("time: %s s\n"%(dt))
        flog.close()        
        deigs = (current_eigenvalues-previous_eigenvalues)/previous_eigenvalues   

        with open(output_dir+"/ad_eigs.txt", "w") as fout:
            for a in range(len(current_eigenvalues)):
                fout.write("%d %e %e\n"%(a+1, current_eigenvalues[a], deigs[a]))
        
        return traj, current_eigenvalues, current_eigenvectors, deigs

    def Ascent_8(self, x, s, max_itr = 3000, output_dir = "./", use_hvp = True, check_etol = False):
        print("Ascent")
        flog = open(output_dir+"/ad_log.txt", "w")

        alp = self.alpha
        dmx = self.dmax
        first_time_reach_s = False

        self.timer = Clock()
        
        traj = x.clone().detach().unsqueeze(0)
        current_energy = self.functional(x).detach()
        current_force = self.compute_force(x)
        current_force_len = torch.linalg.norm(current_force)

        current_eigenvalues, current_eigenvectors = self.diagnalize(self.compute_hessian(x))
        force_s, force_s_tilde = self.compute_force_components(current_eigenvectors, current_force, s)
        force_s_len, force_s_tilde_len = self.adaptive_force_contribution(force_s, force_s_tilde)
        current_biased_force = current_force - 2*force_s
        current_biased_force_len = torch.linalg.norm(current_biased_force)

        scale_prefactor, max_disp = self.get_scale_prefactor(current_biased_force, alp)
        force_s_len       = torch.linalg.norm(force_s)
        force_s_tilde_len = torch.linalg.norm(force_s_tilde)
        abs_force_max = current_force.abs().max()
        previous_abs_force_max = abs_force_max

        self.print_line_ad(flog, 0, current_energy, alp, scale_prefactor, dmx, current_force_len, current_biased_force_len, force_s_len, force_s_tilde_len, abs_force_max, max_disp, current_eigenvalues)
        
        for i in range(max_itr):

            trail_x, trail_energy = self.evaluate_trail_step(x, scale_prefactor*current_biased_force*alp)

            dE = ((trail_energy-current_energy)/current_energy.abs())
            previous_eigenvalues = current_eigenvalues.clone()
            x = trail_x.clone()
            traj = torch.vstack((traj,x.unsqueeze(0)))
            current_energy = trail_energy
            if use_hvp:
                _, Hv0 = self.compute_hvp(x, current_eigenvectors[:,0])
                fg0 = (torch.inner(Hv0,current_eigenvectors[:,0])/current_eigenvalues[0] - 1.0).abs()
                _, Hvs = self.compute_hvp(x, current_eigenvectors[:,s])
                fgs = (torch.inner(Hvs,current_eigenvectors[:,s])/current_eigenvalues[s] - 1.0).abs()
                if fg0 > 0.1 or fgs > 0.1: current_eigenvalues, current_eigenvectors = self.diagnalize(self.compute_hessian(x))
            else:
                current_eigenvalues, current_eigenvectors = self.diagnalize(self.compute_hessian(x))
            current_force = self.compute_force(x)
            current_force_len = torch.linalg.norm(current_force)

            force_s, force_s_tilde = self.compute_force_components(current_eigenvectors, current_force, s)
            force_s_len, force_s_tilde_len = self.adaptive_force_contribution(force_s, force_s_tilde)
            current_biased_force = force_s_tilde - force_s #(force_s + force_s_tilde) - 2*force_s
            current_biased_force_len = torch.linalg.norm(current_biased_force)

            scale_prefactor, max_disp = self.get_scale_prefactor(current_biased_force, alp)
            abs_force_max = current_force.abs().max()

            self.print_line_ad(flog, i+1, current_energy, alp, scale_prefactor, dmx, current_force_len, current_biased_force_len, force_s_len, force_s_tilde_len, abs_force_max, max_disp, current_eigenvalues)

            if first_time_reach_s: alp *= self.alpha_decay_rate
            elif (previous_eigenvalues<0).nonzero().shape[0] == s: first_time_reach_s = True

            abs_dE = dE.abs()
            # if check_etol and i > 100:
            #     if abs_dE < self.etol:
            #         print("energy tolerance")
            #         flog.write("energy tolerance: %e\n"%(abs_dE))
            #         break
            if abs_force_max < self.ftol:
                print("force tolerance")
                flog.write("force tolerance: %e\n"%(abs_force_max))
                break
            if alp < self.atol:
                print("alpha tolerance")
                flog.write("alpha tolerance\n")
                break
            if max_disp < self.dtol:
                print("max displacement tolerance")
                flog.write("max displacement tolerance\n")
                break

        current_eigenvalues, current_eigenvectors = self.diagnalize(self.compute_hessian(x))
        dt = self.timer.get_dt()
        print("time: %s s"%(dt))
        flog.write("time: %s s\n"%(dt))
        flog.close()        
        deigs = (current_eigenvalues-previous_eigenvalues)/previous_eigenvalues   

        with open(output_dir+"/ad_eigs.txt", "w") as fout:
            for a in range(len(current_eigenvalues)):
                fout.write("%d %e %e\n"%(a+1, current_eigenvalues[a], deigs[a]))
        
        return traj, current_eigenvalues, current_eigenvectors, deigs
        
    def Dscent(self, x, max_itr = 1000, output_dir = "./"):
        print("Steepest Gradient Dscent")
        flog = open(output_dir+"/sd_log.txt", "w")
        alp = self.alpha
        traj = x.clone().detach().unsqueeze(0)
        current_energy = self.functional(x).detach()
        current_force = self.compute_force(x)
        scale_prefactor, max_disp = self.get_scale_prefactor(current_force, alp)
        print("%d | %.10e | %.3e %.3e"%(0, current_energy, alp, scale_prefactor))
        flog.write("%d | %.10e | %.3e %.3e\n"%(0, current_energy, alp, scale_prefactor))
        for i in range(max_itr):
            trail_x, trail_energy = self.evaluate_trail_step(x, scale_prefactor*current_force*alp)
            while trail_energy > current_energy:
                alp *= 0.5
                trail_x, trail_energy = self.evaluate_trail_step(x, scale_prefactor*current_force*alp)

            dE = ((current_energy-trail_energy)/current_energy)
            x = trail_x.clone()
            traj = torch.vstack((traj,x.unsqueeze(0)))
            current_energy = trail_energy
            current_force = self.compute_force(x)
            scale_prefactor, max_disp = self.get_scale_prefactor(current_force, alp)

            abs_dE = dE.abs()
            abs_Fmax = current_force.abs().max()
            print("%d | %.10e | %.3e %.3e %.3e %.3e"%(i+1, current_energy, alp, scale_prefactor, abs_dE, abs_Fmax))
            flog.write("%d | %.10e | %.3e %.3e %.3e %.3e\n"%(i+1, current_energy, alp, scale_prefactor, abs_dE, abs_Fmax))
            if abs_dE < self.etol: break
            if abs_Fmax < self.ftol: break
            alp *= 1.1
        
        flog.close()

        return traj
    
    def hvp(self, x, s):
        self.timer = Clock()
        
        H = self.compute_hessian(x)
        eig, evec = self.diagnalize(H)
        print(self.timer.get_dt())

        E, Hv = self.compute_hvp(x, evec[:,0])
        print(torch.inner(Hv,evec[:,0])/eig[0], Hv.norm(), eig[0])

        alp = self.alpha
        dmx = self.dmax

        current_force = self.compute_force(x)
        force_s, force_s_tilde = self.compute_force_components(evec, current_force, s)
        # force_s_len       = torch.linalg.norm(force_s)
        # force_s_tilde_len = torch.linalg.norm(force_s_tilde)
        # if force_s_len < force_s_tilde_len:
        #     force_s_tilde *= force_s_len/force_s_tilde_len
        #     force_s_tilde_len = force_s_len
        current_biased_force = current_force - 2*force_s
        scale_prefactor, max_disp = self.get_scale_prefactor(current_biased_force, alp)
        x1, trail_energy = self.evaluate_trail_step(x, scale_prefactor*current_biased_force*alp)

        E_trail, Hv_trail = self.compute_hvp(x1, evec[:,0])
        print(torch.inner(Hv_trail,evec[:,0])/eig[0], Hv_trail.norm(), eig[0])

        H1 = self.compute_hessian(x1)
        eig1, evec1 = self.diagnalize(H1)
        E1, Hv1 = self.compute_hvp(x1, evec1[:,0])
        print(torch.inner(Hv1,evec1[:,0])/eig1[0], Hv1.norm(), eig1[0])
        
        self.compute_new_eigen_from_previous_eigen(x1, evec[:,0], eig[0])

    
    # def Ascent(self, x, s, max_itr = 1000, method = 0, output_dir = "./"):
    #     print("Ascent")
    #     flog = open(output_dir+"/ad_log.txt", "w")
            
    #     alp = self.alpha
    #     dmx = self.dmax

    #     first_time_reach_s = False
    #     revised_s = s
    #     #revise_ratio = 1.5
    #     #max_revised_s = int(s*revise_ratio)
    #     # if method == 3:
    #     #     revised_s = max_revised_s
    #     #     m3_fg = torch.tensor(0).to(x.device)
    #     traj = x.clone().detach().unsqueeze(0)
    #     if method >= 6: current_ma = traj[0]
    #     current_energy = self.functional(x).detach()
    #     current_eigenvalues, current_eigenvectors = self.diagnalize(self.compute_hessian(x))
    #     current_force = self.compute_force(x)
    #     current_force_len = torch.linalg.norm(current_force)

    #     force_s, force_s_tilde = self.compute_force_components(current_eigenvectors, current_force, s)
    #     current_biased_force = current_force - 2*force_s
    #     current_biased_force_len = torch.linalg.norm(current_biased_force)

    #     scale_prefactor, max_disp = self.get_scale_prefactor(current_biased_force, alp)
    #     align_sign = torch.inner(current_force, current_biased_force).sign()
    #     force_s_len       = torch.linalg.norm(force_s)
    #     force_s_tilde_len = torch.linalg.norm(force_s_tilde)
    #     # abs_force_max = torch.linalg.norm(current_force.reshape(-1, 3), dim=1).max()
    #     abs_force_max = current_force.abs().max()
    #     previous_abs_force_max = abs_force_max
    #     # abs_biased_force_max = torch.linalg.norm(current_biased_force.reshape(-1, 3), dim=1).max()
        

    #     self.print_line_ad(flog, 0, current_energy, alp, scale_prefactor, dmx, current_force_len, current_biased_force_len, force_s_len, force_s_tilde_len, abs_force_max, max_disp, current_eigenvalues)
        
    #     for i in range(max_itr):
            
    #         if method < 2:
    #             trail_x, trail_energy = self.evaluate_trail_step(x, scale_prefactor*current_biased_force*alp)
    #             if (method == 0 and i > 100      and trail_energy < current_energy) or\
    #                (method == 1 and align_sign < 0 and trail_energy < current_energy):
    #                 while trail_energy < current_energy:
    #                     alp *= 0.5
    #                     trail_x, trail_energy = self.evaluate_trail_step(x, scale_prefactor*current_biased_force*alp)
    #                     print("decrease alpha", alp)
    #                     if alp < self.atol: break
    #                 if alp < self.atol:
    #                     print("Alpha tolerance")
    #                     flog.write("Alpha tolerance\n")
    #                     break
    #         # elif 2 <= method <= 4:
    #         #     trail_x, trail_energy = self.evaluate_trail_step(x, scale_prefactor*current_biased_force*alp)
    #         #     if align_sign < 0 and trail_energy < current_energy:
    #         #         while trail_energy < current_energy:
    #         #             alp *= 0.5
    #         #             trail_x, trail_energy = self.evaluate_trail_step(x, scale_prefactor*current_biased_force*alp)
    #         #             print("decrease alpha", alp)
    #         #             if alp < self.atol: break
    #         #         if alp < self.atol:
    #         #             print("Alpha tolerance")
    #         #             flog.write("Alpha tolerance\n")
    #         #             break
    #         elif method >= 5:
    #             trail_x, trail_energy = self.evaluate_trail_step(x, scale_prefactor*current_biased_force*alp)

    #         dE = ((trail_energy-current_energy)/current_energy.abs())
    #         previous_eigenvalues = current_eigenvalues.clone()
    #         x = trail_x.clone()
    #         traj = torch.vstack((traj,x.unsqueeze(0)))
    #         current_energy = trail_energy
    #         current_eigenvalues, current_eigenvectors = self.diagnalize(self.compute_hessian(x))
    #         n_negative_eigs = (current_eigenvalues<0).nonzero().numel()
    #         current_force = self.compute_force(x)
    #         current_force_len = torch.linalg.norm(current_force)

    #         if method < 3:
    #             force_s, force_s_tilde = self.compute_force_components(current_eigenvectors, current_force, s)
    #             force_s_len       = torch.linalg.norm(force_s)
    #             force_s_tilde_len = torch.linalg.norm(force_s_tilde)
    #             # if method == 2 and i > 100:
    #             #     while force_s_len < force_s_tilde_len and revised_s < s + 100:
    #             #         revised_s += 1
    #             #         force_s, force_s_tilde = self.compute_force_components(current_eigenvectors, current_force, revised_s)
    #             #         force_s_len       = torch.linalg.norm(force_s)
    #             #         force_s_tilde_len = torch.linalg.norm(force_s_tilde)
    #         # elif method == 3:
    #         #     if n_negative_eigs >= s:
    #         #         m3_fg += 1
    #         #         revised_s = int(s+(max_revised_s-s)*((-m3_fg/(max_revised_s-s)).exp()))
    #         #     force_s, force_s_tilde = self.compute_force_components(current_eigenvectors, current_force, revised_s)
    #         #     force_s_len       = torch.linalg.norm(force_s)
    #         #     force_s_tilde_len = torch.linalg.norm(force_s_tilde)
    #         elif method >= 4:
    #             force_s, force_s_tilde = self.compute_force_components(current_eigenvectors, current_force, s)
    #             force_s_len       = torch.linalg.norm(force_s)
    #             force_s_tilde_len = torch.linalg.norm(force_s_tilde)
    #             if force_s_len < force_s_tilde_len:
    #                 if method >= 7:
    #                     force_s_tilde *= force_s_len/force_s_tilde_len
    #                     force_s_tilde_len = force_s_len
    #                 # elif (method == 4 and i > 100) or method >= 5:
    #                 #     force_s *= force_s_tilde_len/force_s_len
    #                 #     force_s_len       = torch.linalg.norm(force_s)

    #         current_biased_force = force_s_tilde - force_s #(force_s + force_s_tilde) - 2*force_s
    #         current_biased_force_len = torch.linalg.norm(current_biased_force)

    #         scale_prefactor, max_disp = self.get_scale_prefactor(current_biased_force, alp)
    #         align_sign = torch.inner(current_force, current_biased_force).sign()
    #         # abs_force_max = torch.linalg.norm(current_force.reshape(-1, 3), dim=1).max()
    #         abs_force_max = current_force.abs().max()
    #         # abs_biased_force_max = torch.linalg.norm(current_biased_force.reshape(-1, 3), dim=1).max()

    #         self.print_line_ad(flog, i+1, current_energy, alp, scale_prefactor, dmx, current_force_len, current_biased_force_len, force_s_len, force_s_tilde_len, abs_force_max, max_disp, current_eigenvalues)

    #         # if method == 5 : alp *= scale_prefactor
    #         # if method == 6:
    #         #     if len(traj) > 100 and len(traj) % 10 == 0:
    #         #         new_ma = torch.mean(traj[-50:], dim = 0)
    #         #         ma_dmax_ratio = (current_ma-new_ma).abs().max()/dmx
    #         #         if ma_dmax_ratio < 1e-2:
    #         #             dmx *= 0.1
    #         #         current_ma = new_ma
    #         # if method == 7:
    #         #     if len(traj) > self.traj_ma_window and len(traj) % self.traj_ma_check_interval == 0:
    #         #         new_ma = torch.mean(traj[-self.traj_ma_window:], dim = 0)
    #         #         ma_dmax_ratio = (current_ma-new_ma).abs().max()/dmx
    #         #         if ma_dmax_ratio < self.traj_ma_tol:
    #         #             dmx *= self.traj_ma_shrink_rate
    #         #         current_ma = new_ma
    #         if method == 8:
    #             if first_time_reach_s: alp *= self.alpha_decay_rate
    #             elif (previous_eigenvalues<0).nonzero().shape[0] == s: first_time_reach_s = True
    #         if method == 9:
    #             if first_time_reach_s and previous_abs_force_max < abs_force_max:
    #                     alp *= self.alpha_decay_rate
    #             elif (previous_eigenvalues<0).nonzero().shape[0] == s: first_time_reach_s = True
    #             previous_abs_force_max = abs_force_max

    #         abs_dE = dE.abs()
    #         # if dE.abs() < self.etol:
    #         #     print("Energy tolerance")
    #         #     break
    #         if abs_force_max < self.ftol:
    #             print("force tolerance")
    #             flog.write("force tolerance: %e\n"%(abs_force_max))
    #             break
    #         # if abs_biased_force_max < self.ftol:
    #         #     print("Biased force tolerance")
    #         #     flog.write("Biased Force tolerance: %e\n"%(abs_biased_force_max))
    #         #     break
    #         if alp < self.atol:
    #             print("alpha tolerance")
    #             flog.write("alpha tolerance\n")
    #             break
    #         if max_disp < self.dtol:
    #             print("max displacement tolerance")
    #             flog.write("max displacement tolerance\n")
    #             break
    #         # if dmx < self.dtol:
    #         #     print("dmax tolerance")
    #         #     flog.write("dmax tolerance\n")
    #         #     break
                
    #     deigs = (current_eigenvalues-previous_eigenvalues)/previous_eigenvalues   

    #     with open(output_dir+"/ad_eigs.txt", "w") as fout:
    #         for a in range(len(current_eigenvalues)):
    #             fout.write("%d %e %e\n"%(a+1, current_eigenvalues[a], deigs[a]))

    #     return traj, current_eigenvalues, current_eigenvectors, deigs

    ''' Deprecated
    # def set_eps(self, x):
    #     self.epsilon = 1e-8
    #     self.eps = torch.diag(torch.ones_like(x.flatten())*self.epsilon).reshape(-1,*x.shape)
    # def compute_hessian_n(self, x):   
    #     # tt = Clock()    
    #     n_elements = x.numel()
    #     x_f = x + self.eps
    #     # x_b = x - self.eps
    #     H = []
    #     f = self.compute_force(x)
    #     for i in range(x_f.shape[0]):
    #         f_f = self.compute_force(x_f[i])
    #         H.append((f-f_f)/self.epsilon)
    #         # f_b = self.compute_force(x_b[i])
    #         # H.append(0.5 * (f_b-f_f)/self.epsilon)
    #     H = torch.reshape(torch.stack(H),(n_elements,n_elements))
    #     H = 0.5*(H + H.transpose(0,1))
    #     return H
    '''



    