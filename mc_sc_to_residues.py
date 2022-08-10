from Bio.SeqUtils import seq1, seq3
import math
import math, warnings
from typing import List, Optional, Union

from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
from Bio.PDB.vectors import Vector, rotaxis, calc_dihedral, calc_angle
import numpy as np
import math
import MDAnalysis as mda
#import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
#import rmsd
#import nglview as nv
import numpy as np
from Bio.PDB import PDBIO
from PeptideBuilder import *



# In[4]:


def grep_chi_lst (test_d_lst):
    chi_lst=[]
    for element in test_d_lst:
        merged_angles=[]
        for ang in element["chi"]:
            if ang!=0:
                merged_angles.append(ang)
        chi_lst.append(merged_angles)
    return chi_lst



def Deliver(residue_name, phi, psi):
    #phi = math.degrees(phi)
   # psi = math.degrees(psi)
        
        #ignore user call in cases of GLY/ALA. Simply send list of zeros:
    if residue_name == "GLY" or residue_name == "ALA":
        print("code is run, GLY or ALA doesnt have rotamers")
        return [0, 0, 0, 0]
        
        #calculate closest angles to that of the input:
    phi_flag = float("inf")
    closest_phi = 0
    for i in range(-180, 181, 10):
        if abs(phi-i) < phi_flag:
            phi_flag = abs(phi-i)
            closest_phi = i
        
    psi_flag = float("inf")
    closest_psi = 0
    for i in range(-180, 181, 10):
        if abs(psi-i) < psi_flag:
            psi_flag = abs(psi-i)
            closest_psi = i
        
        #create string for dictionary search:
    search_string=residue_name+'% 4d' % closest_phi+'% 4d' % closest_psi
        
        #search database, output results as a list of sub-dictionaries:
    return search_string

def get_phi_psi_lst (file_address): #helper function
    p = PDBParser()
    structure = p.get_structure("X", file_address)
    ppb = PPBuilder()
    phi=[]
    psi=[]
    for tup in ppb.build_peptides(structure)[0].get_phi_psi_list():
        phi.append(tup[0])
        psi.append(tup[1])
    d_phi = list(map(math.degrees,phi[1:]))
    d_psi = list(map(math.degrees,psi[:-1]))
    aa_str = ""
    for res in ppb.build_peptides(structure)[0]:
        aa_str+=seq1(res.get_resname())
    return aa_str,d_phi, d_psi

def get_omegas(file_address):
    u2 = mda.Universe(file_address, topology_format='PDB')
    protein = u2.select_atoms("name N CA C")
    
    omegas = [res.omega_selection() for res in protein.residues]
    lst_omega = []
    for atomgroups in omegas[:-1]:
        angle = atomgroups.dihedral.value()
        lst_omega.append(angle)
    return lst_omega


def grep_chi_lst(test_d_lst):
    chi_lst=[]
    for element in test_d_lst:
        merged_angles=[]
        for ang in element["chi"]:
            if ang!=0:
                merged_angles.append(ang)
        chi_lst.append(merged_angles)
    return chi_lst


def get_one_atom_lev1(pdb_file_address):
    lst_coord = []
    p = PDBParser()
    structure = p.get_structure("X", pdb_file_address)
    ppb = PPBuilder()
    for res in ppb.build_peptides(structure)[0]:
        res3_name = res.get_resname().upper()
        lev1_atom_name = one_atom_dict_lev1[res3_name][0]
        lev1_coord = res[lev1_atom_name].get_coord()
        lst_coord.append(lev1_coord)
    return np.array(lst_coord)



def get_atoms_lev1_2(pdb_file_address):
    lst_coord = []
    p = PDBParser()
    structure = p.get_structure("X", pdb_file_address)
    ppb = PPBuilder()
    for res in ppb.build_peptides(structure)[0]:
        res3_name = res.get_resname().upper()
        lev1_atom_name_lst = dict_lev1_2[res3_name]
        atom_coord_per_res = []
        for lev1_atom_name in lev1_atom_name_lst:
            #print(lev1_atom_name)
            lev1_coord = res[lev1_atom_name].get_coord()
            atom_coord_per_res.append(lev1_coord)
        lst_coord.append(atom_coord_per_res)
    return np.array(lst_coord)


def get_mainchain_ag(file_address):
    #take a pdb file return the main chain atom group as N, CA,C 
    u2 = mda.Universe(file_address, topology_format='PDB')
    mc_protein = u2.select_atoms("name N CA C")
    return mc_protein


# In[5]:


# Purposes: A function which could help to use rotamer. So the function will take the res name, N,C,CA coords, and the rotamer.
#Then to return the residue class.
# plan. use the initialing function with given input rotamer

def rotamer_res(res_str,mc_position_reshape,segID,chi_lst=[]) -> Residue:
    """Creates a new structure containing a single amino acid. The type and
    geometry of the amino acid are determined by the argument, which has to be
    either a geometry object or a single-letter amino acid code.
    The amino acid will be placed into chain A of model 0."""
    
    res3_name = seq3(res_str).upper()
    
    
    N_coord = mc_position_reshape[0]
    CA_coord = mc_position_reshape[1]
    C_coord = mc_position_reshape[2]
    
    no_rotamer_lst = ["GLY","ALA","PRO"]
  
    if isinstance(res_str, str):
        #print(res_str)
        geo = geometry(res_str)
        if res3_name not in no_rotamer_lst:
            geo.inputRotamers(chi_lst)
            #print("rotamer list is used")
    else:
        raise ValueError("Invalid residue argument:", residue)

    #segID = 1
    AA = geo.residue_name
    CA_N_length = geo.CA_N_length
    CA_C_length = geo.CA_C_length
    N_CA_C_angle = geo.N_CA_C_angle
    

    N = Atom("N", N_coord, 0.0, 1.0, " ", " N", 0, "N")
    CA = Atom("CA", CA_coord, 0.0, 1.0, " ", " CA", 0, "C")
    C = Atom("C", C_coord, 0.0, 1.0, " ", " C", 0, "C")

    ##Create Carbonyl atom (to be moved later)
    C_O_length = geo.C_O_length
    CA_C_O_angle = geo.CA_C_O_angle
    N_CA_C_O_diangle = geo.N_CA_C_O_diangle

    carbonyl = calculateCoordinates(
        N, CA, C, C_O_length, CA_C_O_angle, N_CA_C_O_diangle
    )
    O = Atom("O", carbonyl, 0.0, 1.0, " ", " O", 0, "O")

    res = make_res_of_type(segID, N, CA, C, O, geo)

    return res
    


# In[16]:

def cal_distance_one_atom(rotamer_lev1_atom_cood,ref_coord):
    squared_dist = np.sum((rotamer_lev1_atom_cood-ref_coord)**2, axis=0)
    check_distance = np.sqrt(squared_dist)
    return check_distance

def cal_distance_atom_lst(lst_rotamer_lev1_atom_cood,lst_ref_coord):
    distance_lst = []
    for i in range(len(lst_rotamer_lev1_atom_cood)):
        dist = cal_distance_one_atom(lst_rotamer_lev1_atom_cood[i],lst_ref_coord[i])
        distance_lst.append(dist)
    average_dist = sum(distance_lst) / len(distance_lst) 
    return average_dist




# def lst_residue(file_address):
#     #mc_position,lev1_one_atom_coord_np, res_str_lst
#     aa_str,d_phi, d_psi = get_phi_psi_lst(file_address)
#     lev1_one_atom_coord_np = get_atoms_lev1_2(file_address)
#     mc = get_mainchain_ag(file_address)
#     mc_position=mc.positions
#     lst_res = []
#     mc_position_reshape = mc_position.reshape(-1,3,3)
    
#     i=0
#     res_str = aa_str[i]
#     res = rotamer_res(res_str,mc_position_reshape[i],1,[])
#     lst_res.append(res)
#     distance_lst=[]
    
#     for i in range(1,len(mc_position_reshape)):
#         res_str = aa_str[i] 
#         res3_name = seq3(res_str).upper()    
#         no_rotamer_lst = ["GLY","ALA","PRO"]
#         if res3_name in no_rotamer_lst:
#             chi_lsts=[]
#             res = rotamer_res(res_str,mc_position_reshape[i],i+1,chi_lsts)
#         else:
#             search_str = Deliver(res3_name,d_phi[i-1],d_psi[i-1])
#             chi_lsts = grep_chi_lst(d_r[search_str])
#             opt_pos = 0 # optimal position for ith in chi_lst
#             distance = float("inf")
#             atom_name_lst = dict_lev1_2[res3_name]  
#             ref_coord_lst = lev1_one_atom_coord_np[i]

#             for j in range(len(chi_lsts)):
#                 check_res = rotamer_res(res_str,mc_position_reshape[i],i+1,chi_lsts[j])
#                 rotamer_lev1_atom_cood_lst = []
#                 for atom_name in atom_name_lst:
#                     rotamer_lev1_atom_cood = check_res[atom_name].get_coord()
#                     rotamer_lev1_atom_cood_lst.append(rotamer_lev1_atom_cood)
#                 check_distance = cal_distance_atom_lst(rotamer_lev1_atom_cood_lst,ref_coord_lst)
#                 if check_distance<distance:
#                     distance = check_distance
#                     opt_pos=j
#             #print(res3_name + "    ")
#             #print(distance)
#             distance_lst.append(distance)
        
#             res = rotamer_res(res_str,mc_position_reshape[i],i+1,chi_lsts[opt_pos])
#         lst_res.append(res)
#     return lst_res,distance_lst
    
def make_structure_from_lst_res (lst_res):
    cha = Chain("A")
    for res in lst_res:            
        cha.add(res)
    mod = Model(0)
    mod.add(cha)
    struc = Structure("X")
    struc.add(mod)
    return struc
    


# In[17]:


# this is for i=50
#distance_lst=[]
def g_rotamer(read_filename,write_filename = "g_rotamer.pdb"):
    distance_lst=[]
    l_r, distance_lst = lst_residue(read_filename)
    new_struc = make_structure_from_lst_res(l_r)
    io = PDBIO()
    io.set_structure(new_struc)
    #print(sum(distance_lst) / len(distance_lst))
    io.save(write_filename)
    return sum(distance_lst) / len(distance_lst)
    #print(distance_lst)
    

def lst_residue(aa_str,mc_coords,sd_coords):
    mc_position=mc_coords
    lst_res = []
    mc_position_reshape = mc_position.reshape(-1,3,3)
    
    for i in range(0,len(mc_position_reshape)):
        res_str = aa_str[i] 
        res3_name = seq3(res_str).upper()
        chi_lsts=sd_coords
        res = rotamer_res(res_str,mc_position_reshape[i],i+1,chi_lsts[i])
        lst_res.append(res)
        #print(lst_res)
    return lst_res
    

# In[ ]:




    
# from a list of rotamers(or saying residues), pick up the one with the miniumum score.    
def select_best_rotamer(pre_lst_res, top_k_rotamers):
    mini_score = float("inf")
    best_rotamer = top_k_rotamers[0]
    j=0
    for i in range(len(top_k_rotamers)):
        score= score_function(pre_lst_res,top_k_rotamers[i])
        if score < mini_score:
            mini_score=score
            best_rotamer=top_k_rotamers[i]
            j=i
    return best_rotamer,j     

import copy
def score_function(pre_lst_res,rotamer):
    pdb_test_name="test_score.pdb"
    new_lst = copy.deepcopy(pre_lst_res)
    new_lst.append(rotamer)
    new_struc = make_structure_from_lst_res(new_lst)
    io = PDBIO()
    io.set_structure(new_struc)
    io.save(pdb_test_name)
    score = rosetta_score(pdb_test_name)
    return score


from pyrosetta.teaching import *

import pyrosetta
pyrosetta.init()


def rosetta_score(pdb_file_name):
    ras = pyrosetta.pose_from_pdb(pdb_file_name)
    sfxn = get_score_function(True)
    total_score = sfxn(ras)
    return total_score

def Extract(lst): 
    return [item[0] for item in lst] 

def Sort(sub_li): 
  
    # reverse = None (Sorts in Ascending order) 
    # key is set to sort using second element of  
    # sublist lambda has been used 
    sub_li.sort(key = lambda x: x[1]) 
    return sub_li 



def get_distance(lst):
    return [item[1] for item in lst] 
    
    
# for using rosetta. using default rotmaers to generate full structures.



def c_pdb(seq,mc,sd,write_filename="test_file.pdb"):
    l_r = lst_residue(seq,mc,sd)
    new_struc = make_structure_from_lst_res(l_r)
    io = PDBIO()
    io.set_structure(new_struc)
    io.save(write_filename)
  #  print(rosetta_score(write_filename))






#the below code is updated on Jun1 2021 to quickly grep the levl1 atom coordinates without generating pdb files


dic_lev1 = {'ARG': ['NH1', 'CZ'],
 'THR': ['CB', 'CG2'],
 'VAL': ['CA', 'CB'],
 'ASP': ['CG', 'OD1'],
 'TYR': ['CZ', 'OH'],
 'PHE': ['CB', 'CG', 'CD2'],
 'GLU': ['CD', 'OE1'],
 'ALA': ['CA', 'CB'],
 'ILE': ['CG1', 'CD1'],
 'LEU': ['CG', 'CD2'],
 'LYS': ['CE', 'NZ'],
 'CYS': ['CB', 'SG'],
 'SER': ['CB', 'OG'],
 'ASN': ['CG', 'OD1'],
 'MET': ['SD', 'CE'],
 'GLN': ['CD', 'NE2'],
 'PRO': ['CB', 'CG'],
 'HIS': ['CB', 'CG', 'ND1'],
 'GLY': ['CA'],
 'TRP': ['CB', 'CG', 'CD2']}



 # given list of residues to produce the level1 atom coordinates
def get_lst_lev1_atoms_coords(seq,mc,sd):
    lst_res = lst_residue(seq,mc,sd)
    length = len(lst_res)
    lst_lev1_per_peptide = []
    for i in range(length):
        lst_lev1_per_res = []
        aa = seq[i]
        res3_name = seq3(aa).upper()
        lev1_atom_lst =  dic_lev1[res3_name]
        for atom_name in lev1_atom_lst:
            lst_lev1_per_res.append(lst_res[i][atom_name].get_coord().tolist())
        lst_lev1_per_peptide.append(lst_lev1_per_res)
    return lst_lev1_per_peptide

