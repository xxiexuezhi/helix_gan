#!/usr/bin/env python
# coding: utf-8

# In[2]:


from Bio.PDB.Atom import Atom
from Bio.PDB import PDBIO
import numpy as np
from visualization_mainchain_len14_lev1_2 import *
import MDAnalysis as mda

#visualization_mainchain_len14 is a direct copy from ../

# make sidechain
dict_lev1_2 = {'ARG': ['NH1', 'CZ', 'NE'],
 'THR': ['CB', 'CG2', 'CA'],
 'ASP': ['CG', 'OD1', 'CB'],
 'TYR': ['CZ', 'OH', 'CB', 'CG', 'CD2'],
 'PHE': ['CB', 'CG', 'CD2'],
 'GLU': ['CD', 'OE1', 'CG'],
 'ILE': ['CG1', 'CD1', 'CB'],
 'LEU': ['CG', 'CD2', 'CB'],
 'LYS': ['CE', 'NZ', 'CD'],
 'CYS': ['CB', 'SG', 'CA'],
 'SER': ['CB', 'OG', 'CA'],
 'ASN': ['CG', 'OD1', 'CB'],
 'MET': ['SD', 'CE', 'CG'],
 'GLN': ['CD', 'NE2', 'CG'],
 'PRO': ['CB', 'CG', 'CA'],
 'HIS': ['CB', 'CG', 'ND1'],
 'TRP': ['CB', 'CG', 'CD2'],
 'VAL': ['CA', 'CB'],
 'GLY': ['CA'],
 'ALA': ['CA', 'CB']}

side_level_atoms = ['CG1',
 'CD1',
 'CG',
 'NH1',
 'CD2',
 'OD1',
 'ND1',
 'CZ',
 'OH',
 'CD',
 'NE2',
 'SD',
 'SG',
 'CG2',
 'OG',
 'CB',
 'CE',
# 'CA',
 'NZ',
 'OE1']


def deocde_sidechain_one_atom(atom_arr):
    bond_length = (atom_arr[0]+atom_arr[1])/2
    if bond_length <0:
        bond_length=0
    sin1 = atom_arr[2]
    cos1 = atom_arr[3]
    angle1 = np.arctan2(sin1,cos1)
    sin2 = atom_arr[4]
    cos2 = atom_arr[5]
    angle2 = np.arctan2(sin2,cos2)   
    return [bond_length * 10,angle1,angle2]


def decode_sidechain_residue (np_3atom_arr):
    #return lst in [distance, angle1,angle2]
    lst_one_res = []
    atom1 = np_3atom_arr[0:6]
    atom2 = np_3atom_arr[6:12]
    atom3 = np_3atom_arr[12:18]
    atom4 = np_3atom_arr[18:24]
    atom5 = np_3atom_arr[24:30]
    lst_one_res.append(deocde_sidechain_one_atom(atom1))
    lst_one_res.append(deocde_sidechain_one_atom(atom2))
    lst_one_res.append(deocde_sidechain_one_atom(atom3))
    lst_one_res.append(deocde_sidechain_one_atom(atom4))
    lst_one_res.append(deocde_sidechain_one_atom(atom5))
    return lst_one_res


def decode_sidechain_peptide(lst_np):
    pep_lst = []
    for one_res in lst_np:
        pep_lst.append(decode_sidechain_residue(one_res))
        
    return pep_lst
        


def cal_sidechain_atom(atom_a, atom_b, atom_c, bond_angle, torsion_angle, bond_length) :
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


def create_sidechain (file_address,sd_pep_lst):
    p = PDBParser()
    structure = p.get_structure("X", file_address)
    ppb = PPBuilder()
    lst_names = []
    lst_positions = []
    for i in range(len(ppb.build_peptides(structure)[0])):
        res = ppb.build_peptides(structure)[0][i]
        atom_lst = dict_lev1_2[res.get_resname().upper()]
        #atom_name_per_res = []
        for j in range(len(atom_lst)):
            atom_name = atom_lst[j]
            N = list(res["N"].get_coord())
            CA = list(res["CA"].get_coord())
            C = list(res["C"].get_coord())
            #Y_actual = list(res[atom_name].get_coord())
            bond_length  = sd_pep_lst[i][j][0]
            torsion_angle = sd_pep_lst[i][j][1]
            bond_angle = sd_pep_lst[i][j][2]
            lst=[N,CA,C,bond_angle, torsion_angle,bond_length]
            #print(lst)
            position = cal_sidechain_atom(N,CA,C,bond_angle, torsion_angle,bond_length)
            
            lst_names.append(res.get_resname() + "_"+atom_name)
            lst_positions.append(position)
    return lst_names,lst_positions
                    

def c_structure(file_address,lst_positions,saved_file_name):
    p = PDBParser()
    structure = p.get_structure("X", file_address)
    ppb = PPBuilder()
    p = PDBParser()
    count=0
    for i in range(len(ppb.build_peptides(structure)[0])):
        res = ppb.build_peptides(structure)[0][i]
        atom_lst = dict_lev1_2[res.get_resname().upper()]
        for j in range(len(atom_lst)):
            atom_name = atom_lst[j]
            new_atom = Atom(atom_name, lst_positions[count], 0.0, 1.0, " "," "+ atom_name+" ",atom_name )
            count+=1
            if atom_name!="CA":
                res.add(new_atom)
    io = PDBIO()
    io.set_structure(structure)
    #saved_filename = "test_all_structure.pdb"
    io.save(saved_file_name)
    parser = PDBParser()
    new_structure = parser.get_structure("x",saved_file_name)
    view = nv.show_biopython(new_structure)
    return view    
 
def v_full_structure(np_arr,saved_filename = "test_all_structure.pdb",show_full_structure=False):
    #np_arr is the numpy array length * nc. in my case, 14 * 50
    v_mainchain(np_arr,lst_first_3_atoms = lst_first_3_atoms,
                saved_mc_filename = "./mc_test_len14.pdb", show_structure = False)
    file_address = "./mc_test_len14.pdb"
    seq_raw, main_raw, side_raw = get_1dseq_main_side(np_arr)
    sd_pep_lst = decode_sidechain_peptide(side_raw)
    lst_names,lst_positions = create_sidechain (file_address,sd_pep_lst)
    full_view = c_structure(file_address,lst_positions,saved_filename)
    return full_view



import warnings
warnings.filterwarnings('ignore')

#v_mainchian(g_data[3])

def get_sidechain_index(file_address):
    u2 = mda.Universe(file_address, topology_format='PDB')
    index = 1 
    lst_str_sidechain_index = []
    for atom in u2.atoms:
        if (atom.name in side_level_atoms):
            lst_str_sidechain_index.append(str(index))
        index+=1    
    return lst_str_sidechain_index

import ipywidgets



def show_mutiply_structures(lst_numpy):
    lst_view = []
    for one in lst_numpy:
        show_img = one.detach().numpy()
        view = v_full_structure(show_img)
        sd_index = get_sidechain_index("test_all_structure.pdb")
        view.add_ball_and_stick(selection=sd_index)
#         view.representations = [
#             {"type": "cartoon", "params": {
#                 "sele": "protein", "color": "residueindex"
#                 }},
#             {"type": "ball+stick", "params": {
#                 "sele": "1 2 3 4 5 6 7 8"
#                 }}
#             ]

        lst_view.append(view)
    vbox = ipywidgets.VBox(lst_view)
    return vbox




def show_mutiply_input_structures(lst_numpy):
    lst_view = []
    for one in lst_numpy:
        show_img = one
        view = v_full_structure(show_img)
        sd_index = get_sidechain_index("test_all_structure.pdb")
        view.add_ball_and_stick(selection=sd_index)
        lst_view.append(view)
    vbox = ipywidgets.VBox(lst_view)
    return vbox


def v_rotamer_full_structure(np_arr,file_address,saved_filename, show_full_structure=False):
    #np_arr is the numpy array length * nc. in my case, 14 * 50
    seq_raw, main_raw, side_raw = get_1dseq_main_side(np_arr)
    sd_pep_lst = decode_sidechain_peptide(side_raw)
    lst_names,lst_positions = create_sidechain (file_address,sd_pep_lst)
    full_view = c_structure(file_address,lst_positions,saved_filename)
    return full_view


# In[3]:


def decoder_lev1_2_mc(g_array,saved_filename=""):
    if saved_filename=="":    
        saved_filename = "generated_lev1_2_mc.pdb"    
    show_img = g_array.detach().numpy()
    v_full_structure(show_img,saved_filename,show_full_structure=False)


# In[ ]:




