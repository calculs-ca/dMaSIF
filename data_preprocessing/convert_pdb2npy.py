import argparse

import numpy as np
from pathlib import Path
from tqdm import tqdm
from Bio.PDB import *

from data_preprocessing.download_pdb import protonate, extractPDB

ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}


def load_structure_np(fname, center):
    """Loads a .ply mesh to return a point cloud and connectivity."""
    # Load the data
    parser = PDBParser()
    structure = parser.get_structure("structure", fname)
    atoms = structure.get_atoms()

    coords = []
    types = []
    for atom in atoms:
        coords.append(atom.get_coord())
        types.append(ele2num[atom.element])

    coords = np.stack(coords)
    types_array = np.zeros((len(types), len(ele2num)))
    for i, t in enumerate(types):
        types_array[i, t] = 1.0

    # Normalize the coordinates, as specified by the user:
    if center:
        coords = coords - np.mean(coords, axis=0, keepdims=True)

    return {"xyz": coords, "types": types_array}


def convert_pdbs(pdb_dir, npy_dir):
    print("Converting PDBs")
    for p in tqdm(pdb_dir.glob("*.pdb")):
        protein = load_structure_np(p, center=False)
        np.save(npy_dir / (p.stem + "_atomxyz.npy"), protein["xyz"])
        np.save(npy_dir / (p.stem + "_atomtypes.npy"), protein["types"])


def convert_pdb_file_to_npy(pdb_filepath: Path, out_path: Path, include_complex=True, include_indiv_chains=True):
    file_stem = pdb_filepath.stem
    protonated_file = out_path / (file_stem + '_proton.pdb')

    ##### Protonate with reduce, if hydrogens included.
    # - Always protonate as this is useful for charges. If necessary ignore hydrogens later.
    protonate(pdb_filepath, protonated_file)

    def make_npy_from_pdb(chain):
        if chain is None:
            chain_id = 'all'
        else:
            chain_id = chain

        out_filename = out_path / f"{file_stem}_{chain_id}.pdb"
        extractPDB(protonated_file, str(out_filename), chain)
        protein = load_structure_np(out_filename, center=False)
        np.save(out_path / f"{file_stem}_{chain_id}_atomxyz", protein["xyz"])
        np.save(out_path / f"{file_stem}_{chain_id}_atomtypes", protein["types"])

    if include_complex:
        make_npy_from_pdb(None)

    if include_indiv_chains:
        # Get list of chains
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure(protonated_file, protonated_file)
        model = Selection.unfold_entities(struct, "M")[0]
        all_chains = [chain.get_id() for chain in model]

        if include_complex and len(all_chains) == 1:
            print(f'{pdb_filepath} contains a single chain: not outputting per-chain pdb since the complex has been '
                  f'output.')
        else:
            for chain in all_chains:
                make_npy_from_pdb(chain)


def main():
    ap = argparse.ArgumentParser("Convert a dir of pdb files into a dir of npy files, for use with dMaSIF.\n"
                                 "By default, include whole PDBs (complexes) AND split PDBs (chains) in the output.")
    ap.add_argument('pdb_dir', type=Path)
    ap.add_argument('--out_path', type=Path, required=True)
    ap.add_argument('--no_complex', action='store_true', help="Don't include the full pdb as a single protein")
    ap.add_argument('--no_indiv_chains', action='store_true', help="Don't split the pdb into its chains")
    args = ap.parse_args()

    include_complex = not args.no_complex
    include_indiv_chains = not args.no_indiv_chains

    pdb_paths = args.pdb_dir.glob('*.pdb')
    for p in tqdm(pdb_paths):
        convert_pdb_file_to_npy(p, args.out_path, include_complex=include_complex,
                                include_indiv_chains=include_indiv_chains)


if __name__ == '__main__':
    main()
