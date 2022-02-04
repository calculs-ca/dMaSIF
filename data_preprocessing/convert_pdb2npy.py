import numpy as np
from pathlib import Path
from tqdm import tqdm
from Bio.PDB import *

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
    g = list(pdb_dir.glob("*.pdb"))
    for p in tqdm(g):
        try:
            protein = load_structure_np(p, center=False)
            np.save(npy_dir / (p.stem + "_atomxyz.npy"), protein["xyz"])
            np.save(npy_dir / (p.stem + "_atomtypes.npy"), protein["types"])
        except Exception as e:
            print(e, p)


if __name__ == "__main__":
    protein_dir = Path('surface_data/raw/01-benchmark_surfaces_npy')
    assert protein_dir.exists()
    convert_pdbs(Path('surface_data/raw/01-benchmark_pdbs'), protein_dir)

