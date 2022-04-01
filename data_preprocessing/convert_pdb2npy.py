import argparse
from pathlib import Path

import numpy as np
from pathlib import Path
from tqdm import tqdm
from Bio.PDB import *
from Bio.SeqUtils import IUPACData


ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}


PROTEIN_LETTERS = [x.upper() for x in IUPACData.protein_letters_3to1.keys()]

# Exclude disordered atoms.
class NotDisordered(Select):
    def accept_atom(self, atom):
        return not atom.is_disordered() or atom.get_altloc() == "A"  or atom.get_altloc() == "1"


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


def find_modified_amino_acids(path):
    """
    Contributed by github user jomimc - find modified amino acids in the PDB (e.g. MSE)
    """
    res_set = set()
    for line in open(path, 'r'):
        if line[:6] == 'SEQRES':
            for res in line.split()[4:]:
                res_set.add(res)
    for res in list(res_set):
        if res in PROTEIN_LETTERS:
            res_set.remove(res)
    return res_set


def extractPDB(
        infilename, outfilename, chain_ids=None
):
    # extract the chain_ids from infilename and save in outfilename.
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(infilename, infilename)
    model = Selection.unfold_entities(struct, "M")[0]
    chains = Selection.unfold_entities(struct, "C")
    # Select residues to extract and build new structure
    structBuild = StructureBuilder.StructureBuilder()
    structBuild.init_structure("output")
    structBuild.init_seg(" ")
    structBuild.init_model(0)
    outputStruct = structBuild.get_structure()

    # Load a list of non-standard amino acid names -- these are
    # typically listed under HETATM, so they would be typically
    # ignored by the orginal algorithm
    modified_amino_acids = find_modified_amino_acids(infilename)

    for chain in model:
        if (
                chain_ids == None
                or chain.get_id() in chain_ids
        ):
            structBuild.init_chain(chain.get_id())
            for residue in chain:
                het = residue.get_id()
                if het[0] == " ":
                    outputStruct[0][chain.get_id()].add(residue)
                elif het[0][-3:] in modified_amino_acids:
                    outputStruct[0][chain.get_id()].add(residue)

    # Output the selected residues
    pdbio = PDBIO()
    pdbio.set_structure(outputStruct)
    pdbio.save(outfilename, select=NotDisordered())


def protonate(in_pdb_file, out_pdb_file):
    # protonate (i.e., add hydrogens) a pdb using reduce and save to an output file.
    # in_pdb_file: file to protonate.
    # out_pdb_file: output file where to save the protonated pdb file.

    # Remove protons first, in case the structure is already protonated
    args = ["reduce", "-Trim", in_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode('utf-8').rstrip())
    outfile.close()
    # Now add them again.
    args = ["reduce", "-HIS", out_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode('utf-8'))
    outfile.close()


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
