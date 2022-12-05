import os
import numpy as np

class VASPDataConvertor:
    def __init__(self, input_dir_list = ["."], iterative = True, type_list = None, virial = True, sort_folder = False):
        self.virial_fg = virial
        self.type_list = type_list
        self.find_files(input_dir_list, iterative, sort_folder)
        self.import_data()
        #type_list coord_ froce_ lattice_vector_ energy_ virial_

    def export(self, output_dir = ".", shuffle = True):
        self.export_data(output_dir, shuffle)

    def find_files(self, input_dir_list, iterative, sort_folder):
        if iterative:
            self.file_list = []
            for input_dir in input_dir_list:
                f = os.popen("find %s -type f -name OUTCAR"%(input_dir)).read()
                f = f.split('\n')
                self.file_list += f[:-1]
        else:
            self.file_list = [input_dir+"/OUTCAR" for input_dir in input_dir_list]
        if sort_folder:
            self.file_list = sorted(self.file_list)

    def import_data(self):
        self.get_atom_number_and_type()
        self.coord_ = []
        self.force_ = []
        self.energy_ = []
        self.virial_ = []
        self.lattice_vector_ = []
        for filename in self.file_list:
            print("reading %s..."%(filename))
            temp, press, coord, force, energy, virial, lattice_vector = self.read_vasp_outcar(filename)
            self.coord_ += coord
            self.force_ += force
            self.energy_ += energy
            self.virial_ += virial
            self.lattice_vector_ += lattice_vector
            print("%d frames."%(len(coord)))
        
    def get_atom_number_and_type(self):
        with open(self.file_list[0][:-6]+"XDATCAR","r") as fin:
            for i in range(5): fin.readline()
            linelist = fin.readline().strip().split()
            self.atom_name_ = [i for i in linelist]
            self.number_of_types_ = len(self.atom_name_)
            linelist = fin.readline().strip().split()
            self.number_of_atoms_of_each_type_ = [int(i) for i in linelist]
            number_of_atoms = 0
            for number_of_atoms_of_one_type in self.number_of_atoms_of_each_type_: number_of_atoms += number_of_atoms_of_one_type
            if self.type_list is None:
                self.type_list = []
                for i, number_of_atoms_of_one_type in enumerate(self.number_of_atoms_of_each_type_):
                    self.type_list += [i]*number_of_atoms_of_one_type
            self.number_of_atoms_ = number_of_atoms
            
    def read_vasp_outcar(self, filename):
        temp = []
        press = []
        coord = []
        force = []
        virial = []
        energy = []
        lattice_vector = []
        with open(filename,"r") as outcar_in:
            for aline in outcar_in:
                if "(temperature" in aline:
                    tmp = aline[aline.index("(temperature")+len("(temperature"):]
                    tmp = tmp.split()[0]
                    temp.append(float(tmp))
                elif "total pressure  =" in aline:
                    press.append(float(aline.strip().split()[3]))
                elif "POSITION" in aline and "TOTAL-FORCE (eV/Angst)" in aline:
                    coord.append([])
                    force.append([])
                    outcar_in.readline() #-------...
                    newline = outcar_in.readline()
                    while "---" not in newline:
                        linelist = newline.strip().split()
                        coord[-1].append(linelist[0:3])
                        force[-1].append(linelist[3:6])
                        newline = outcar_in.readline()
                elif "in kB" in aline and self.virial_fg:
                    #2  3  4  5  6  7
                    #XX YY ZZ XY YZ ZX
                    #2  5  7  5  3  6  7  6  4
                    #xx xy xz yx yy yz zx zy zz
                    linelist = aline.strip().split()
                    virial.append([str(float(linelist[i])*1000) for i in [2,5,7,5,3,6,7,6,4]])
                elif "energy  without entropy=" in aline: 
                    linelist = aline.strip().split()
                    energy.append(linelist[3])
                elif "direct lattice vectors" in aline:
                    lattice_vector.append([])
                    newline = outcar_in.readline()
                    linelist = newline.strip().split()
                    lattice_vector[-1].append(linelist[0:3])
                    newline = outcar_in.readline()
                    linelist = newline.strip().split()
                    lattice_vector[-1].append(linelist[0:3])
                    newline = outcar_in.readline()
                    linelist = newline.strip().split()
                    lattice_vector[-1].append(linelist[0:3])
                # elif"total energy   ETOTAL =" in aline:
                #     linelist = aline.strip().split()
                #     energy.append(linelist[4])
        del lattice_vector[0]
        return temp, press, coord, force, energy, virial, lattice_vector
   
    def export_data(self, output_dir, shuffle):
        number_of_frames = len(self.coord_)
        scrambled_index = np.arange(number_of_frames)
        if shuffle: np.random.shuffle(scrambled_index)

        print("Exporting type.raw...")    
        fout = open(output_dir+"/type.raw","w")
        for i in range(len(self.type_list)):
            fout.write("%s "%(self.type_list[i]))
        fout.close()        
        print("Exporting coord.raw...")
        fout = open(output_dir+"/coord.raw","w")
        for index in scrambled_index:
            for i in range(len(self.coord_[index])):
                fout.write("%s %s %s "%(self.coord_[index][i][0],self.coord_[index][i][1],self.coord_[index][i][2]))
            fout.write("\n")
        fout.close()
        print("Exporting force.raw...")    
        fout = open(output_dir+"/force.raw","w")
        for index in scrambled_index:
            for i in range(len(self.force_[index])):
                fout.write("%s %s %s "%(self.force_[index][i][0],self.force_[index][i][1],self.force_[index][i][2]))
            fout.write("\n")
        fout.close()
        print("Exporting box.raw...")    
        fout = open(output_dir+"/box.raw","w")
        for index in scrambled_index:
            for i in range(len(self.lattice_vector_[index])):
                for j in range(len(self.lattice_vector_[index][i])):
                    fout.write("%s "%(self.lattice_vector_[index][i][j]))
            fout.write("\n")
        fout.close()
        print("Exporting energy.raw...")    
        fout = open(output_dir+"/energy.raw","w")
        for index in scrambled_index:
            fout.write("%s\n"%(self.energy_[index]))
        fout.close()
        if self.virial_fg == True:
            print("Exporting virial.raw...")    
            fout = open(output_dir+"/virial.raw","w")
            for index in scrambled_index:
                for i in range(len(self.virial_[index])):
                    fout.write("%s "%(self.virial_[index][i]))
                fout.write("\n")
            fout.close()

class VASPRunner:
    def __init__(self, output_dir, \
                       atom_position = None, lattice_vector = None, element = None, atom_name = None, atom_num = None, coord_type = 'c', \
                       POSCAR_path = None, INCAR_path = None, POTCAR_path = "POTCAR", \
                       system_name = "system", vasp_executable = "vasp",\
                       run_type = "md", vdw_type = None, scan = False, \
                       ENMAX = None, POTIM = 1, NSW = 100,\
                       TEBEG = 1000, TEEND = 1000, EDIFFG = 0.001, \
                       NCORE = 1):
        self.root_dir = os.getcwd()
        self.vasp_executable = vasp_executable
        self.POTCAR_path = POTCAR_path
        self.output_dir = output_dir
        self.create_folder()
        if POSCAR_path:
            self.POSCAR_path = POSCAR_path
            os.system("cp %s %s"%(self.POSCAR_path.replace('CONTCAR', 'POSCAR'), self.output_dir))
        else:
            coord_type = coord_type[0].lower()
            self.create_poscar(atom_position, lattice_vector, element, atom_name, atom_num, system_name, coord_type)
        self.create_kpoint()
        os.system("cp %s %s"%(self.POTCAR_path, self.output_dir))
        if INCAR_path:
            self.INCAR_path = INCAR_path
            os.system("cp %s %s"%(self.INCAR_path, self.output_dir))
        else:
            self.create_incar(run_type, vdw_type, scan, ENMAX, POTIM, NSW, TEBEG, TEEND, EDIFFG, NCORE)

    def create_kpoint(self):
        kpointfile = open("%s/KPOINTS"%(self.output_dir), "w")
        kpointfile.write('''K-Points\n 0\nMonkhorst\n 1  1  1\n 0  0  0\n''')
        kpointfile.close()

    def create_poscar(self, atom_position, lattice_vector, element, atom_name, atom_num, system_name, coord_type):
        if element is None:
            if atom_name is None and atom_num is None:
                print("Error: both element and atom_name/atom_num are not specified")
                exit()
        else:
            atom_name = [element[0]]
            atom_num = [1]
            for a_element in element[1:]:
                if atom_name[-1] != a_element:
                    atom_name.append(a_element)
                    atom_num.append(1)
                else:
                    atom_num[-1] += 1
        poscarfile = open("%s/POSCAR"%(self.output_dir), "w")
        poscarfile.write("%s\n 1.00000\n"%(system_name))
        for i in range(len(lattice_vector)):
            for j in range(len(lattice_vector[i])):
                poscarfile.write("%s "%(lattice_vector[i][j]))
            poscarfile.write("\n")
        for i in range(len(atom_name)):
            poscarfile.write("%s "%(atom_name[i]))
        poscarfile.write("\n")
        for i in range(len(atom_num)):
            poscarfile.write("%s "%(atom_num[i]))
        poscarfile.write("\n")
        if coord_type == 'c':   coord_type_str = 'Cartesian'
        elif coord_type == 'd': coord_type_str = 'Direct'
        else:
            print("Error: unknown coord_type")
            exit()
        poscarfile.write("%s\n"%(coord_type_str))
        for i in range(len(atom_position)):
            for j in range(len(atom_position[i])):
                poscarfile.write("%s "%(atom_position[i][j]))
            poscarfile.write("\n")
        poscarfile.close()

    def create_incar(self, run_type, vdw_type, scan, \
                           ENMAX, POTIM, NSW,\
                           TEBEG, TEEND, EDIFFG,\
                           NCORE):
        parm_map = {}
        if run_type == "nvt":      
            parm_map['IBRION'] = 0
            parm_map['SMASS']  = 1
            parm_map['TEBEG']  = TEBEG
            parm_map['TEEND']  = TEEND
        elif run_type == "min":
            parm_map['IBRION'] = 2
            parm_map['EDIFFG'] = EDIFFG

        parm_map['POTIM'] = POTIM
        parm_map['NSW']   = NSW

        parm_map['ENMAX'] = ENMAX
        if parm_map['ENMAX'] is None:
            parm_map['ENMAX'] = 0
            with open(self.POTCAR_path, "r") as fin:
                for aline in fin:
                    if "ENMAX" in aline:
                        e = float(aline.strip().split()[2][:-1])
                        parm_map['ENMAX'] = max(parm_map['ENMAX'], e)
            parm_map['ENMAX'] *= 1.5
        
        parm_map['PREC']    = "Normal"
        parm_map['ISMEAR']  = 0
        parm_map['SIGMA']   = 0.1
        parm_map['ISYM']    = 0
        parm_map['MDALGO']  = 2
        parm_map['ISIF']    = 2
        parm_map['LREAL']   = "Auto"
        parm_map['LCHARG']  = ".FALSE."
        parm_map['IVDW']    = 0
        parm_map['LASPH']   = ".False."
        if vdw_type is not None:    parm_map['LASPH'] = ".TRUE."                    
        if vdw_type == "d3":        parm_map['IVDW']  = 11
        elif vdw_type == "d3-damp": parm_map['IVDW']  = 12
        if scan:
            parm_map['METAGGA'] = 'SCAN'
            parm_map['LASPH']   = '.TRUE.'
            parm_map['ADDGRID'] = '.TRUE.'

        incarfile = open("%s/INCAR"%(self.output_dir), "w")
        for key in parm_map.keys(): incarfile.write("%s = %s\n"%(key, parm_map[key]))
        incarfile.close()
    
    def create_folder(self):
        os.system("mkdir -p %s"%(self.output_dir))

    def run(self, c = 0, n = 1):
        os.chdir(self.output_dir)
        os.system("export CUDA_VISIBLE_DEVICES=%d && %s"%(c, self.vasp_executable))
        os.chdir(self.root_dir)