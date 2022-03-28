# Standard imports:
import itertools

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose
from pathlib import Path

# Custom data loader and model:
from data import ProteinPairsSurfaces, PairData, CenterPairAtoms, load_protein_pair, load_indiv_protein_pair
from data import RandomRotationPairAtoms, NormalizeChemFeatures, iface_valid_filter
from model import dMaSIF
from data_iteration import iterate
from helper import *
from Arguments import parser

parser.add_argument('--npy_dir')
args = parser.parse_args()
model_path = "models/" + args.experiment_name
save_predictions_path = Path("preds/" + args.experiment_name)

# Ensure reproducability:
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)


# Load the train and test datasets:
transformations = (
    Compose([NormalizeChemFeatures(), CenterPairAtoms(), RandomRotationPairAtoms()])
    if args.random_rotation
    else Compose([NormalizeChemFeatures()])
)

if args.npy_dir is not None:
    npy_paths = Path(args.npy_dir).glob('*_atomxyz.npy')
    npy_pairs = itertools.product(npy_paths, npy_paths)
    test_dataset = []
    test_pdb_ids = []

    def get_stem(p: Path):
        # e.g. 'dir/ABCD_A_atomxyz.npy' -> 'ABCD_A'
        return '_'.join(p.stem.split('_')[:-1])

    for npy1, npy2 in npy_pairs:
        npy1_id = get_stem(npy1)
        npy2_id = get_stem(npy2)
        test_dataset.append(load_indiv_protein_pair(npy1_id, npy1_id, npy1.parent))
        test_pdb_ids.append(f'{npy1_id}__{npy2_id}')
elif args.single_pdb != "":
    single_data_dir = Path("./data_preprocessing/npys/")
    test_dataset = [load_protein_pair(args.single_pdb, single_data_dir,single_pdb=True)]
    test_pdb_ids = [args.single_pdb]
elif args.pdb_list != "":
    with open(args.pdb_list) as f:
        pdb_list = f.read().splitlines()
    single_data_dir = Path("./data_preprocessing/npys/")
    test_dataset = [load_protein_pair(pdb, single_data_dir,single_pdb=True) for pdb in pdb_list]
    test_pdb_ids = [pdb for pdb in pdb_list]
else:
    test_dataset = ProteinPairsSurfaces(
        "surface_data", train=False, ppi=args.search, transform=transformations
    )
    test_pdb_ids = (
        np.load("surface_data/processed/testing_pairs_data_ids.npy")
        if args.site
        else np.load("surface_data/processed/testing_pairs_data_ids_ppi.npy")
    )

    test_dataset = [
        (data, pdb_id)
        for data, pdb_id in zip(test_dataset, test_pdb_ids)
        if iface_valid_filter(data)
    ]
    test_dataset, test_pdb_ids = list(zip(*test_dataset))


# PyTorch geometric expects an explicit list of "batched variables":
batch_vars = ["xyz_p1", "xyz_p2", "atom_coords_p1", "atom_coords_p2"]
test_loader = DataLoader(
    test_dataset, batch_size=args.batch_size, follow_batch=batch_vars
)

net = dMaSIF(args)
# net.load_state_dict(torch.load(model_path, map_location=args.device))
net.load_state_dict(
    torch.load(model_path, map_location=args.device)["model_state_dict"]
)
net = net.to(args.device)

# Perform one pass through the data:
info = iterate(
    net,
    test_loader,
    None,
    args,
    test=True,
    save_path=save_predictions_path,
    pdb_ids=test_pdb_ids,
)

#np.save(f"timings/{args.experiment_name}_convtime.npy", info["conv_time"])
#np.save(f"timings/{args.experiment_name}_memoryusage.npy", info["memory_usage"])
