#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import nglview as nv
import numpy as np
import math, itertools
import numpy as np
import math, itertools

from Bio.SeqUtils import seq1, seq3
from Bio.Seq import Seq

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
import MDAnalysis as mda
import pickle


# In[9]:


# this script takes input 

#parameters

#inmporting the first three atoms coordinators from original pdb files.

using_opt = False


#with open('../Encoding/toy_data_10_20.pickle', 'rb') as f:
#    filtered_pdb_lst = pickle.load(f)

#file_address = filtered_pdb_lst[69] # 43 can be changed to any number
#u_pre = mda.Universe(file_address, topology_format='PDB')
#bb_pre = u_pre.select_atoms("name N CA C")

#lst_first_3_atoms =[bb_pre.positions[0+6],bb_pre.positions[1+6],bb_pre.positions[2+6]]
lst_first_3_atoms = [np.array([-0.52274139,  1.36320998,  0.        ]),np.array([0.0, 0.0, 0.0]),np.array([1.52, 0.  , 0.  ])]

#test
#with open('generated_data1122.pickle', 'rb') as f:
#    g_data = pickle.load(f)
#    g_data = g_data.detach().numpy()


# In[8]:

def optimizer(angle,angle_type):
    ref_phi = -1.117
    ref_psi = -0.715
    ref_omega = 3.141 if (angle>0) else -3.141
    ref_angle = angle
    ref_na = math.radians(121)
    ref_ac = math.radians(109)
    ref_cn = math.radians(115)
    if angle_type=="phi":
        ref_angle=ref_phi
    if angle_type=="psi":
        ref_angle=ref_psi
    if angle_type=="omega":
        ref_angle=ref_omega
    if angle_type=="na":
        ref_angle=ref_na
    if angle_type=="ac":
        ref_angle=ref_ac
    if angle_type=="cn":
        ref_angle=ref_cn    
    if abs(angle-ref_angle)>0.2 *abs(ref_angle):
        return angle + 0.2 *(ref_angle - angle)
    return angle
    
        
        
        
            



def rev_one_hot(seq_numps): # decoder from numpy array to primary sequence.
    REV_ONE_HOT = 'ACDEFGHIKLMNPQRSTVWY'
    sequence = []

    for aa_nump in seq_numps:
        if aa_nump.any()==0:
            return ''.join(sequence)
        else:
            sequence.append(REV_ONE_HOT[np.argmax(aa_nump)])
    return ''.join(sequence)


def get_1dseq_main_side(num_arr):
    #return 3 list for one example not batch. so num_arr is 30 * 50.Mar 5th  THe arr shape is changed. udpated 62 to 42.
    seq_raw, main_raw, side_raw =[],[],[]
    for aa in num_arr:
        aa_seq_one = aa[0:20]
        aa_main_one = aa[20:32]
        #aa_side_one = aa[32:50]
        aa_side_one = aa[32:42]
        seq_raw.append(aa_seq_one)
        main_raw.append(aa_main_one)
        side_raw.append(aa_side_one)
    return seq_raw, main_raw, side_raw 
    
    

class NeRF(object):

    def __init__(self,first3atoms,a_c,c_n,n_a):
    # TODO - PROLINE has different lengths which we should take into account
    # TODO - A_TO_C angle differs by +/- 5 degrees
    #bond_lengths = { "N_TO_A" : 1.4615, "PRO_N_TO_A" : 1.353, "A_TO_C" : 1.53, "C_TO_N" : 1.325 }
        #u2 = mda.Universe(pdb_address, topology_format='PDB')
        #bb2 = u2.select_atoms("name N CA C")
        #a_c,c_n,n_a = get_bond_angles(bb2.positions)
        self.bond_lengths = { "N_TO_A" : 1.4615,  "A_TO_C" : 1.53, "C_TO_N" : 1.325 }
        self.bond_angles = { "A_TO_C" : a_c, "C_TO_N" : c_n, "N_TO_A" : n_a }
        self.bond_order = ["C_TO_N", "N_TO_A", "A_TO_C"]
        self.first3atoms = first3atoms

    def _next_data(self, key):
   # ''' Loop over our bond_angles and bond_lengths '''
        ff = itertools.cycle(self.bond_order)
        for item in ff:
            if item == key:
                next_key = next(ff)
                break
        return (self.bond_angles[next_key], self.bond_lengths[next_key], next_key)

    def _place_atom(self, atom_a, atom_b, atom_c, bond_angle, torsion_angle, bond_length) :
  #  ''' Given the three previous atoms, the required angles and the bond
  #  lengths, place the next atom. Angles are in radians, lengths in angstroms.''' 
    # TODO - convert to sn-NeRF
        ab = np.subtract(atom_b, atom_a)
        bc = np.subtract(atom_c, atom_b)
        bcn = bc / np.linalg.norm(bc)
        R = bond_length

    # numpy is row major
        d = np.array([-R * math.cos(bond_angle),
            R * math.cos(torsion_angle) * math.sin(bond_angle),
            R * math.sin(torsion_angle) * math.sin(bond_angle)])

        n = np.cross(ab,bcn)
        n = n / np.linalg.norm(n)
        nbc = np.cross(n,bcn)

        m = np.array([ 
          [bcn[0],nbc[0],n[0]],
          [bcn[1],nbc[1],n[1]],
          [bcn[2],nbc[2],n[2]]])

        d = m.dot(d)
        d = d + atom_c
        return d

    def compute_positions(self, torsions):
    #''' Call this function with a set of torsions (including omega) in degrees.'''
       # atoms = [[0, -1.355, 0], [0, 0, 0], [1.4466, 0.4981, 0]]
        #atoms = [[62.879,  24.715,  19.025],[62.601,  24.888,  17.612],[61.239,  25.552,  17.408]]
        atoms = self.first3atoms
        #torsions = list(map(math.radians, torsions))
        key = "C_TO_N"
        angle = self.bond_angles[key]
        length = self.bond_lengths[key]

        for i,torsion in enumerate(torsions):
                atoms.append(self._place_atom(atoms[-3], atoms[-2], atoms[-1], angle[int(i/3)], torsion, length))
                (angle, length, key) = self._next_data(key)

        return atoms

def get_mainchain_positions(lst_first_3_atoms,torsionangle_lst, a_c, c_n, n_a):
    #torsionangle_lst, a_c, c_n, n_a = get_mainchain_angles(mc_angle_lst)
    #nerf = NeRF([[50.402,  16.719,  30.101],[49.761,  15.415,  30.197],[48.375,15.418,29.601]],a_c, c_n, n_a)
    
    nerf = NeRF(lst_first_3_atoms,a_c, c_n, n_a)
    atoms_positions = []
    atoms_positions = nerf.compute_positions(torsionangle_lst) #omega is calculated
    del nerf
    return atoms_positions


def get_mainchain_angles(mc_np):
    mc_angle_lst = []
    a_c = []
    c_n = []
    n_a = []
    #for i in range(1,len(mc_np)):# # one padding in the left, the rest all padding in the right.
    for i in range(0,len(mc_np)-1):
        if mc_np[i].any()==0:
            return mc_angle_lst, a_c, c_n, n_a
        else:
            angle1 = np.arctan2(mc_np[i][0],mc_np[i][1])
            angle2 = np.arctan2(mc_np[i][2],mc_np[i][3])
            angle3 = np.arctan2(mc_np[i][4],mc_np[i][5])
            angle_ac = np.arctan2(mc_np[i][6],mc_np[i][7])
            angle_cn = np.arctan2(mc_np[i][8],mc_np[i][9])
            angle_na = np.arctan2(mc_np[i][10],mc_np[i][11])
            
            if using_opt:
                angle1 = optimizer(angle1,"psi")
                angle2 = optimizer(angle2,"omega")
                angle3 = optimizer(angle3,"phi")
                angle_ac = optimizer(angle_ac,"ac")
                angle_cn = optimizer(angle_cn,"cn")
                angle_na = optimizer(angle_na,"na")
                
            mc_angle_lst.append(angle1)
            mc_angle_lst.append(angle2)
            mc_angle_lst.append(angle3)
            a_c.append(angle_ac)
            c_n.append(angle_cn)
            n_a.append(angle_na)
    return mc_angle_lst, a_c, c_n, n_a



def get_sidechain_angles(arr):
    s,m,sc_np = get_1dseq_main_side(arr)
    sc_angle_lst = []
    for i in range(len(sc_np)):
        angle_lst_one_res = []
        angle1 = np.arctan2(sc_np[i][0],sc_np[i][1])
        angle2 = np.arctan2(sc_np[i][2],sc_np[i][3])
        angle3 = np.arctan2(sc_np[i][4],sc_np[i][5])
        angle4 = np.arctan2(sc_np[i][6],sc_np[i][7])
        angle5 = np.arctan2(sc_np[i][8],sc_np[i][9])
        angle1=math.degrees(angle1)
        angle2=math.degrees(angle2)
        angle3=math.degrees(angle3)
        angle4=math.degrees(angle4)
        angle5=math.degrees(angle5)
        angle_lst_one_res.append(angle1)
        angle_lst_one_res.append(angle2)
        angle_lst_one_res.append(angle3)
        angle_lst_one_res.append(angle4)
        angle_lst_one_res.append(angle5)
        sc_angle_lst.append(angle_lst_one_res)
    return sc_angle_lst



def get_1d_mc_sd(arr):
    sd= get_sidechain_angles(arr)
    s,m,d = get_1dseq_main_side(arr)
    one_letter_seq = rev_one_hot(s)
    the_lst_first_3_atoms = [np.array([-0.52274139,  1.36320998,  0.        ]),np.array([0.0, 0.0, 0.0]),np.array([1.52, 0.  , 0.  ])]
    mc  = np.array(get_mainchain_positions(the_lst_first_3_atoms,np.array(get_mainchain_angles(m)[0]),
        np.array(get_mainchain_angles(m)[1]),np.array(get_mainchain_angles(m)[2]),np.array(get_mainchain_angles(m)[3])))
    return one_letter_seq, mc,sd
def v_mainchain(arr,lst_first_3_atoms = lst_first_3_atoms,
                saved_mc_filename = "./mc_test_len14.pdb", show_structure = True):
    
    #lst_first_3_atoms =[bb_pre.positions[0+6],bb_pre.positions[1+6],bb_pre.positions[2+6]]

    lst_first_3_atoms = [np.array([-0.52274139,  1.36320998,  0.        ]),np.array([0.0, 0.0, 0.0]),np.array([1.52, 0.  , 0.  ])]

    s,m,d = get_1dseq_main_side(arr)
    
    one_letter_seq = rev_one_hot(s)
   # print(one_letter_seq)
    three_lett_seq = seq3(str(one_letter_seq)).upper()
   # print(three_lett_seq)
    
    new_main_chain_positions  = np.array(get_mainchain_positions(lst_first_3_atoms, 
                                                                 np.array(get_mainchain_angles(m)[0])
                        ,np.array(get_mainchain_angles(m)[1])
                        ,np.array(get_mainchain_angles(m)[2])
                        ,np.array(get_mainchain_angles(m)[3])))
    
    
    
#    print(len(new_main_chain_positions))
    
    
    u2 = mda.Universe("mainchain_len14.pdb")
   # print(u2.atoms.positions.shape)
   # print(new_main_chain_positions.shape)
    u2.atoms.positions = new_main_chain_positions
    for i in range(14):
        #u2.atoms[3*i:3*i+3].residues.resnames = np.array([three_lett_seq[3*i:3*i+3]], dtype=object)
        u2.atoms[3*i:3*i+3].residues.resnames = np.array([three_lett_seq[3*i:3*i+3]], dtype=object)
    u2.atoms.write("./mc_test_len14.pdb")
    if not show_structure:
        return
    parser = PDBParser()
    structure = parser.get_structure("x",saved_mc_filename)
    view5 = nv.show_biopython(structure)
    #print("test update 3\n")
    #you can decide show or not show here
   # print(get_mainchain_angles(m)[0])
    return view5


# In[10]:
import warnings
warnings.filterwarnings('ignore')

#v_mainchian(g_data[3])

import ipywidgets

def show_mutiply_structures(lst_numpy):
    lst_view = []
    for one in lst_numpy:
        show_img = one.detach().numpy()
        view = v_mainchain(show_img)
        #view = v_mainchain(one)
        lst_view.append(view)
    vbox = ipywidgets.VBox(lst_view)
    return vbox

# In[ ]:




