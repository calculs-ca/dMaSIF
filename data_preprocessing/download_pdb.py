import Bio
from Bio.PDB import *
import sys
import importlib
import os
import numpy as np
from subprocess import Popen, PIPE
from pathlib import Path
from convert_pdb2npy import load_structure_np
import argparse

parser = argparse.ArgumentParser(description="Arguments")
parser.add_argument(
    "--pdb", type=str,default='', help="PDB code along with chains to extract, example 1ABC_A_B", required=False
)
parser.add_argument(
    "--pdb_list", type=str,default='', help="Path to a text file that includes a list of PDB codes along with chains, example 1ABC_A_B", required=False
)

# tmp_dir = Path('./tmp')
# pdb_dir = Path('./pdbs')
# npy_dir = Path('./npys')


def get_single(pdb_id: str,chains: list):
    protonated_file = pdb_dir/f"{pdb_id}.pdb"
    if not protonated_file.exists():
        # Download pdb 
        pdbl = PDBList()
        pdb_filename = pdbl.retrieve_pdb_file(pdb_id, pdir=tmp_dir,file_format='pdb')

        ##### Protonate with reduce, if hydrogens included.
        # - Always protonate as this is useful for charges. If necessary ignore hydrogens later.
        protonate(pdb_filename, protonated_file)

    pdb_filename = protonated_file

    # Extract chains of interest.
    for chain in chains:
        out_filename = pdb_dir/f"{pdb_id}_{chain}.pdb"
        extractPDB(pdb_filename, str(out_filename), chain)
        protein = load_structure_np(out_filename,center=False)
        np.save(npy_dir / f"{pdb_id}_{chain}_atomxyz", protein["xyz"])
        np.save(npy_dir / f"{pdb_id}_{chain}_atomtypes", protein["types"])





# if __name__ == '__main__':
#     args = parser.parse_args()
#     if args.pdb != '':
#         pdb_id = args.pdb.split('_')
#         chains = pdb_id[1:]
#         pdb_id = pdb_id[0]
#         get_single(pdb_id,chains)
#
#     elif args.pdb_list != '':
#         with open(args.pdb_list) as f:
#             pdb_list = f.read().splitlines()
#         for pdb_id in pdb_list:
#            pdb_id = pdb_id.split('_')
#            chains = pdb_id[1:]
#            pdb_id = pdb_id[0]
#            get_single(pdb_id,chains)
#     else:
#         raise ValueError('Must specify PDB or PDB list')