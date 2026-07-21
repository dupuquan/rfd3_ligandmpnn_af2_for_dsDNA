import os
os.environ['DEBUG'] = 'False'  # 或 '0'
import biotite.structure.io as strucio

from mpnn.inference_engines.mpnn import MPNNInferenceEngine
from biotite.structure import get_residue_starts
from biotite.sequence import ProteinSequence
from lightning.fabric import seed_everything
from scipy.spatial import cKDTree
import numpy as np
from Bio.PDB import Selection, PDBParser,MMCIFParser
#读取指定目录cif文件
import argparse
parser=argparse.ArgumentParser(description='please put your input_directory and your output_directory here')
parser.add_argument('-i','--input',help="input your directory")
parser.add_argument('-o','--output',help="output will be saved in this path ")
parser.add_argument('-n','--number',help="how many residue chain you want (*10) ",default=2)
#parser.add_argument('-f','--fixed_residues', nargs='+', help="list of fixed residues",default=None)
args=parser.parse_args()
input_file = args.input#这里是外部接口,指定输入文件
OUTPUT_MPNN = args.output#这是外部接口，指定输出文件
n=int(args.number)
#fixed_residue= args.fixed_residues if hasattr(args, 'fixed_residues') else None
def get_cif_files_recursive(directory):
    """
    递归遍历指定目录及其所有子目录，返回所有 .cif 文件的完整路径。
    """
    cif_files = []
    try:
        for root, dirs, files in os.walk(directory):
            for name in files:
                if name.lower().endswith('.cif'):
                    cif_files.append(os.path.join(root, name))
    except FileNotFoundError:
        print(f"错误：目录 '{directory}' 不存在")
    return cif_files

# 假设全局变量 OUTPUT_MPNN 已定义，例如：
# OUTPUT_MPNN = "/path/to/output"

def save_dict_to_txt(seq_dict, output_dir,cif_file_count,input_path):
    """
    将字典保存为文本文件，每行格式为 'key: value'。
    如果输出目录不存在则自动创建。
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")

    # 构建输出文件路径（可自定义文件名）
    basename = os.path.basename(input_path)   # "ppi_tutorial_dsDNA_complex_0_model_0.cif"

    # 2. 去掉扩展名（只保留主文件名）
    stem = os.path.splitext(basename)[0]     # "ppi_tutorial_dsDNA_complex_0_model_0"
    file_path = os.path.join(output_dir, f"{stem}.txt")

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for key, value in seq_dict.items():
                # 若值为复杂对象（如列表），可转换为字符串；这里假设为简单类型
                f.write(f"{key}: {value}\n")
        print(f"字典已保存至: {file_path}")
    except Exception as e:
        print(f"写入文件失败: {e}")




# 自定义转换函数：支持标准氨基酸 + 常见核酸
def convert_residue_3to1(res_name):
    # 先尝试氨基酸转换
    try:
        return ProteinSequence.convert_letter_3to1(res_name)
    except KeyError:
        # 核酸残基映射（根据 PDB/CIF 常见命名）
        nucleic_mapping = {
            "DA": "A", "DC": "C", "DG": "G", "DT": "T",   # DNA
            "A": "A", "C": "C", "G": "G", "T": "T", "U": "U",  # RNA 或裸碱基
            "D": "X", "I": "X", "N": "X"                 # 其他非标准
        }
        return nucleic_mapping.get(res_name.upper(), "X")  # 未知返回 X





def find_DNA_protein_hotspots(
        pdb_file,
        atom_distance_cutoff=4.0
):
    """
    Automatically detect DNA-protein interface residues.

    Parameters
    ----------
    pdb_file : str
        input pdb file

    atom_distance_cutoff : float
        atom-atom distance cutoff (Å)

    Returns
    -------
    dict
        DNA residues
        Protein residues
        residue-residue contacts
    """

    DNA_RES = {
        "DA", "DT", "DG", "DC",
        "A", "T", "G", "C",
        "ADE", "THY", "GUA", "CYT"
    }

    PROTEIN_RES = {
        "ALA","ARG","ASN","ASP",
        "CYS","GLN","GLU","GLY",
        "HIS","ILE","LEU","LYS",
        "MET","PHE","PRO","SER",
        "THR","TRP","TYR","VAL"
    }


    parser = MMCIFParser(QUIET=True)

    structure = parser.get_structure(
    "complex",
    pdb_file
)

    model = next(structure.get_models())


    dna_atoms = []
    protein_atoms = []

    dna_chains = []
    protein_chains = []


    # ==========================
    # Detect chain type
    # ==========================

    for chain in model:

        residues = list(chain.get_residues())

        dna_score = 0
        protein_score = 0


        for res in residues:

            name = res.get_resname().strip()

            if name in DNA_RES:
                dna_score += 1

            elif name in PROTEIN_RES:
                protein_score += 1


        if dna_score > protein_score:
            dna_chains.append(chain.id)

            dna_atoms.extend(
                Selection.unfold_entities(
                    chain,
                    "A"
                )
            )

        else:
            protein_chains.append(chain.id)

            protein_atoms.extend(
                Selection.unfold_entities(
                    chain,
                    "A"
                )
            )


    if len(dna_atoms)==0 or len(protein_atoms)==0:
        raise ValueError(
            "Cannot detect DNA or protein chain"
        )


    # ==========================
    # KDTree search
    # ==========================

    dna_coords = np.array(
        [a.coord for a in dna_atoms]
    )

    protein_coords = np.array(
        [a.coord for a in protein_atoms]
    )


    dna_tree = cKDTree(dna_coords)
    protein_tree = cKDTree(protein_coords)


    contacts = dna_tree.query_ball_tree(
        protein_tree,
        atom_distance_cutoff
    )


    dna_residues = set()
    protein_residues = set()

    residue_contacts = []


    # ==========================
    # Collect residue contacts
    # ==========================

    for i, protein_indices in enumerate(contacts):

        if not protein_indices:
            continue


        dna_res = dna_atoms[i].get_parent()

        dna_info = (
            dna_res.get_parent().id,
            dna_res.id[1],
            dna_res.get_resname()
        )


        dna_residues.add(dna_info)


        for j in protein_indices:

            pro_res = protein_atoms[j].get_parent()


            pro_info = (
                pro_res.get_parent().id,
                pro_res.id[1],
                pro_res.get_resname()
            )


            protein_residues.add(pro_info)


            residue_contacts.append(
                (
                    dna_info,
                    pro_info
                )
            )


    # ==========================
    # Convert protein residues to hotspot format
    # ==========================

    protein_hotspots = []

    for chain, resid, resname in sorted(protein_residues):

        protein_hotspots.append(
            f"{chain}{resid}"
        )


    protein_hotspots = []

    for chain, resid, resname in sorted(protein_residues):
        protein_hotspots.append(
            f"{chain}{resid}"
        )


    return protein_hotspots
# Set seed for reproducibility
seed_everything(0)
files = get_cif_files_recursive(input_file)
cif_file_count=1
for cif_file in files:

    fixed_residue=find_DNA_protein_hotspots(cif_file)
    # 加载 CIF 文件
    #cif_file = "/home/dupuquan/RFDF3/rfd3_ppi_tutorial/output_cif/ppi_tutorial_dsDNA_complex_0_model_0.cif"
    atom_array = strucio.load_structure(cif_file)

    # 此时 atom_array 就是 biotite.structure.AtomArray 类型
    #print(atom_array)  # <class 'biotite.structure.atoms.AtomArray'>







    # Configure MPNN inference engine
    # See mpnn.utils.inference.MPNN_GLOBAL_INFERENCE_DEFAULTS for all options
    engine_config = {
        "model_type": "ligand_mpnn",  # or "protein_mpnn" for vanilla ProteinMPNN
        "is_legacy_weights": True,    # Required for now for ligand_mpnn and protein_mpnn
        "out_directory": False,        # Return results in memory
        "write_structures": False,
        "write_fasta": True,
    }

    # Configure per-input inference options
    # See mpnn.utils.inference.MPNN_PER_INPUT_INFERENCE_DEFAULTS for all options
    input_configs = [
        {
            "batch_size": 10,           # 每批10条
            "number_of_batches": n,     # 2批 → 总共20条
            "remove_waters": True,
            "temperature": 0.1,         # 可在此调整采样温度
            "seed": 42,
            # Design scope - if all None, design all residues
            "fixed_residues": fixed_residue,
            "designed_residues": None,
            "fixed_chains": None,
            "designed_chains": None
        }
    ]

    # Run sequence design on the RFD3-generated backbone
    model = MPNNInferenceEngine(**engine_config)
    mpnn_outputs = model.run(input_dicts=input_configs, atom_arrays=[atom_array])





    # Extract and display the designed sequences
    print(f"Generated {len(mpnn_outputs)} designed sequences:\n")

    seq_dict={}


    for i, item in enumerate(mpnn_outputs):
        res_starts = get_residue_starts(item.atom_array)
        seq_1letter = ''.join(
            convert_residue_3to1(res_name)
            for res_name in item.atom_array.res_name[res_starts]
        )
        #print(f"Sequence {i+1}: {seq_1letter}")
        seq_dict[f"Sequence_{i+1}"] = seq_1letter
    #print(seq_dict)
   
    save_dict_to_txt(seq_dict, OUTPUT_MPNN,cif_file_count,cif_file)
    cif_file_count+=1