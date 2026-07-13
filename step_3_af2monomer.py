#调用Bindcraft的AF2权重，及其colab模型，所有包加载

import os
import warnings
# 用 argparse 处理命令行参数，替代 Colab 的文件上传
import argparse  
import tempfile
from colabdesign import mk_afdesign_model, clear_mem
import numpy as np
#from colabdesign import mk_afdesign_model, clear_mem
#import numpy as np
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from natsort import natsorted
import numpy as np
from pdbfixer import PDBFixer
from openmm import LangevinIntegrator, Platform
from openmm.app import (
    ForceField,
    Simulation,
    NoCutoff,
    HBonds,
    PDBFile,
)
from openmm.unit import kelvin, picosecond, picoseconds

import argparse

from scipy.spatial import cKDTree
import numpy as np
import os
from Bio.PDB import PDBParser, MMCIFParser, Superimposer

parser=argparse.ArgumentParser(description='please put your MPNN generate directory , your RFD3 generate cif directory and your output_directory here')
parser.add_argument('-ir','--inputresidue',help="input your MPNN generate directory")
parser.add_argument('-ic','--inputcif',help="input your RFD3 generate directory")
parser.add_argument('-o','--output',help="output will be saved in this path ")
parser.add_argument('-n','--nc',help="the length of your nucleotides, default is 28",default=28)
parser.add_argument('-af','--af2model',help="path to your AF2 model directory",default='/home/dupuquan/RFDF3/validation_RFDF3/ColabDesign-main/params')
args=parser.parse_args()
inputresidue_directory = args.inputresidue#这里是外部接口,指定输入文件MPNN生成的残基文件目录
inputcif_directory=args.inputcif#这里放输入的原本的cif文件
output_directory = args.output#这是外部接口，指定输出文件
na_length = int(args.nc)*2  # 获取核苷酸长度
AF2_model_path = args.af2model#这里是外部接口，指定AF2权重文件夹


def get_txt_files_recursive(directory):
    txt_files = []
    try:
        for root, dirs, files in os.walk(directory):
            for name in files:
                if name.lower().endswith('.txt'):
                    txt_files.append(os.path.join(root, name))
    except FileNotFoundError:
        print(f"错误：目录 '{directory}' 不存在")
        return []
    return natsorted(txt_files)


def get_pdb_files_recursive(directory):
    txt_files = []
    try:
        for root, dirs, files in os.walk(directory):
            for name in files:
                if name.lower().endswith('.pdb'):
                    txt_files.append(os.path.join(root, name))
    except FileNotFoundError:
        print(f"错误：目录 '{directory}' 不存在")
        return []
    return natsorted(txt_files)

def read_dict_from_txt(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")
    result = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '=' not in line:
                continue
            key, value = line.split('=', 1)
            result[key.strip()] = value.strip()
    return result

def read_dict_from_txt(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")
    
    result = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue  # 跳过空行或无效行
            key, value = line.split(':', 1)   # 以冒号分割一次
            result[key.strip()] = value.strip()
    return result

def monomer_single_predict(key,na_length,seq,output_dir,AF2_model_path):
    clear_mem()
    seq_new=seq[na_length:]
    print(seq_new)
    binder_prediction_model = mk_afdesign_model(protocol="hallucination", use_templates=False, initial_guess=False,use_initial_atom_pos=False, num_recycles=1,data_dir=AF2_model_path, use_multimer=False)
    binder_prediction_model.prep_inputs(length=len(seq_new))
    binder_prediction_model.set_seq(seq_new)
    binder_prediction_model.predict()
    log = binder_prediction_model.aux["log"]
   # 确保目录存在
    output_dir_path=os.path.dirname(output_dir)+ os.sep
    #print(output_dir_path)
    basename = os.path.basename(output_dir) 
    stem = os.path.splitext(basename)[0]  
    os.makedirs(os.path.dirname(output_dir_path), exist_ok=True)
    output_pdb = os.path.join(
    output_dir_path,
    f"{stem}_{key}.pdb"
    )
    print(output_pdb)
    # 保存PDB
    if log['plddt']>0.85 and log['ptm']>0.65:
        binder_prediction_model.save_current_pdb(output_pdb)

        return log



def openmm_relax(
    input_pdb,
    output_pdb,
    use_gpu=True,
    max_iterations=1000,
):
    """
    使用 OpenMM 对蛋白进行 Energy Minimization。

    Parameters
    ----------
    input_pdb : str
        输入PDB路径

    output_pdb : str
        输出PDB路径

    use_gpu : bool
        是否使用CUDA

    max_iterations : int
        最大最小化步数
    """

    # 修复结构
    fixer = PDBFixer(filename=input_pdb)

    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()

    # 一般AF2结构没有水
    fixer.removeHeterogens(True)

    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    # Amber14力场
    forcefield = ForceField(
        "amber14-all.xml",
        "amber14/tip3p.xml"
    )

    system = forcefield.createSystem(
        fixer.topology,
        nonbondedMethod=NoCutoff,
        constraints=HBonds,
    )

    integrator = LangevinIntegrator(
        300 * kelvin,
        1.0 / picosecond,
        0.002 * picoseconds,
    )

    platform = Platform.getPlatformByName(
        "CUDA" if use_gpu else "Reference"
    )

    simulation = Simulation(
        fixer.topology,
        system,
        integrator,
        platform,
    )

    simulation.context.setPositions(
        fixer.positions
    )

    # Energy before
    state = simulation.context.getState(getEnergy=True)
    print("Before:",
          state.getPotentialEnergy())

    # 最小化
    simulation.minimizeEnergy(
        maxIterations=max_iterations
    )

    # Energy after
    state = simulation.context.getState(getEnergy=True)
    print("After:",
          state.getPotentialEnergy())

    # 输出
   
    with open(output_pdb, "w") as f:

        PDBFile.writeFile(
            simulation.topology,
            simulation.context.getState(
                getPositions=True
            ).getPositions(),
            f,
        )

def monomer_predict(dict_sequence,na_length,input_file,AF2_model_path):#应该返回该蛋白质的基本信息，plddt，i_PTM等
    seq_dict=dict_sequence
    #design_name= int(np.random.randint(0, high=999999, size=1, dtype=int)[0])
    #print(input_file)
    for key,value in seq_dict.items():
        
        log=monomer_single_predict(key,na_length,value,input_file,AF2_model_path)

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


from scipy.spatial import cKDTree
import numpy as np
import os
from Bio.PDB import PDBParser, MMCIFParser, Superimposer
import tree


class StructureValidator:

    def __init__(self, reference_file, prediction_file):

        self.reference_file = reference_file
        self.prediction_file = prediction_file

        self.reference = self._load_structure(reference_file)#RFD3 generate
        self.prediction = self._load_structure(prediction_file)#AF2 generate

    # ==========================
    # IO
    # ==========================

    def _load_structure(self, filepath):

        ext = os.path.splitext(filepath)[1].lower()

        if ext == ".pdb":
            parser = PDBParser(QUIET=True)
        elif ext in [".cif", ".mmcif"]:
            parser = MMCIFParser(QUIET=True)
        else:
            raise ValueError(f"Unsupported format: {ext}")

        return parser.get_structure("model", filepath)

    # ==========================
    # Chain utilities
    # ==========================

    def get_chain(self, structure, chain_id):

        model = next(structure.get_models())
        return model[chain_id]

    def get_ca_atoms(self, chain):

        return [
            residue["CA"]
            for residue in chain
            if "CA" in residue
        ]



    def get_heavy_atoms(self, chain):

        atoms = []

        for residue in chain:

            for atom in residue:

                if atom.element == "H":
                    continue

            atoms.append(atom)

        return atoms
    # ==========================
    # RMSD
    # ==========================

    def calculate_rmsd(
        self,
        reference_chain="C",
        prediction_chain="A"
    ):

        ref_chain = self.get_chain(self.reference, reference_chain)
        pred_chain = self.get_chain(self.prediction, prediction_chain)

        ref_ca = self.get_ca_atoms(ref_chain)
        pred_ca = self.get_ca_atoms(pred_chain)

        n = min(len(ref_ca), len(pred_ca))

        sup = Superimposer()
        sup.set_atoms(ref_ca[:n], pred_ca[:n])

        return sup.rms

    # ==========================
    # Alignment
    # ==========================

    def align_prediction(
        self,
        reference_chain="C",
        prediction_chain="A"
    ):

        import copy

        pred_aligned = copy.deepcopy(self.prediction)

        ref_chain = self.get_chain(self.reference, reference_chain)
        pred_chain = self.get_chain(pred_aligned, prediction_chain)

        ref_ca = self.get_ca_atoms(ref_chain)
        pred_ca = self.get_ca_atoms(pred_chain)

        n = min(len(ref_ca), len(pred_ca))

        sup = Superimposer()
        sup.set_atoms(ref_ca[:n], pred_ca[:n])
        sup.apply(pred_aligned.get_atoms())

        return pred_aligned, sup.rms

    # ==========================
    # 🔥 UPDATED CLASH FUNCTION（重点）
    # ==========================
    

        # ==========================
# Hbond / Salt bridge helper
# ==========================

    def _is_hbond_pair(self, e1, e2):
        """
    Possible hydrogen bond donor/acceptor pair.
    Heavy atoms only.
        """

        HBOND = {
        ("N", "O"),
        ("O", "N"),
        ("N", "N"),
        ("O", "O"),
    }

        return (e1, e2) in HBOND


    def _is_salt_bridge(self, atom1, atom2):
        """
    Very simple salt bridge detection.

    Protein:
        Lys NZ
        Arg NH1 NH2 NE

    DNA:
        phosphate O atoms
        """

        protein_positive = {
        "NZ",
        "NH1",
        "NH2",
        "NE",
    }

        dna_negative = {
        "OP1",
        "OP2",
        "O1P",
        "O2P",
    }

        return (
            atom1.get_name() in protein_positive
            and
            atom2.get_name() in dna_negative
        )
        # ==========================    # Interface clash
    # Binder vs DNA
    # BindCraft style
    # ==========================
    '''
    def calculate_interface_clashes(
        self,
        aligned_prediction,
        target_chains=("A", "B"),
        prediction_chain="A",
        cutoff=2.4
        ):

        # -----------------------
        # Binder atoms
        # -----------------------

        pred_chain = self.get_chain(
            aligned_prediction,
            prediction_chain
        )

        pred_coords = []
        
        for residue in pred_chain:

            for atom in residue:

                if atom.element == "H":
                    continue

                pred_coords.append(atom.coord)


        pred_coords = np.asarray(pred_coords)


        # -----------------------
        # Target DNA atoms
        # -----------------------

        target_coords = []

        for chain_id in target_chains:

            chain = self.get_chain(
                self.reference,
                chain_id
            )

            for residue in chain:

                for atom in residue:

                    if atom.element == "H":
                        continue

                    target_coords.append(atom.coord)


        target_coords = np.asarray(target_coords)


        # -----------------------
        # KDTree
        # -----------------------

        tree = cKDTree(target_coords)


        neighbors = tree.query_ball_point(
            pred_coords,
            r=cutoff
        )


        clash_pairs = 0


        for hits in neighbors:

            clash_pairs += len(hits)


        return {

            "interface_clashes": clash_pairs

        }
    
        '''
    def calculate_interface_clashes(
    self,
    aligned_prediction,
    target_chains=("A", "B"),
    prediction_chain="A",
    search_radius=4.0,
):

    # -----------------------------
    # Bondi vdW radii
    # -----------------------------
        VDW = {
        "H": 1.20,
        "C": 1.70,
        "N": 1.55,
        "O": 1.52,
        "S": 1.80,
        "P": 1.80,
    }

    # -----------------------------
    # binder atoms
    # -----------------------------
        pred_chain = self.get_chain(
            aligned_prediction,
            prediction_chain
    )

        binder_atoms = self.get_heavy_atoms(pred_chain)

        binder_coords = np.array(
            [a.coord for a in binder_atoms]
    )

    # -----------------------------
    # DNA atoms
    # -----------------------------
        target_atoms = []

        for chain_id in target_chains:

            chain = self.get_chain(
                self.reference,
                chain_id
        )

            target_atoms.extend(
                self.get_heavy_atoms(chain)
        )

        target_coords = np.array(
            [a.coord for a in target_atoms]
    )

    # -----------------------------
    # KDTree
    # -----------------------------
        tree = cKDTree(target_coords)

        neighbors = tree.query_ball_point(
            binder_coords,
            r=search_radius
    )

    # -----------------------------
    # statistics
    # -----------------------------
        severe_pairs = 0
        overlap_sum = 0.0
        max_overlap = 0.0

    # -----------------------------
    # loop
    # -----------------------------
        for i, hits in enumerate(neighbors):

            atom_i = binder_atoms[i]

            ri = VDW.get(
                atom_i.element.upper(),
                1.7
        )

            for j in hits:

                atom_j = target_atoms[j]

                rj = VDW.get(
                    atom_j.element.upper(),
                    1.7
            )

                d = np.linalg.norm(
                    atom_i.coord -
                    atom_j.coord
            )

                overlap = (ri + rj) - d

            # -----------------------------------
            # default MolProbity tolerance
            # -----------------------------------
                tolerance = 0.4

            # -----------------------------------
            # possible hydrogen bond
            # -----------------------------------
                if (
                    atom_i.element.upper() in ("N", "O")
                    and
                    atom_j.element.upper() in ("N", "O")
            ):
                    tolerance = 0.8

            # -----------------------------------
            # possible salt bridge
            # -----------------------------------
                if (
                    atom_i.get_name() in ("NZ", "NH1", "NH2", "NE")
                    and
                    atom_j.get_name() in ("OP1", "OP2", "O1P", "O2P")
            ):
                    tolerance = 1.0

                if overlap > tolerance:

                    severe_pairs += 1

                    overlap_sum += overlap

                    max_overlap = max(
                        max_overlap,
                        overlap
                )

        binder_atom_num = len(binder_atoms)

        interface_clashscore = (
            severe_pairs /
            binder_atom_num *
            1000
    )

        return {

        "interface_clash_pairs": severe_pairs,

        "interface_clashscore": interface_clashscore,

        "overlap_sum": overlap_sum,

        "max_overlap": max_overlap,

        "binder_atoms": binder_atom_num,

    }
    # ==========================
    # Internal binder clashes
    # ==========================

    def calculate_internal_clashes(
        self,
        aligned_prediction,
        prediction_chain="A",
        cutoff=2.4
        ):


        chain = self.get_chain(
            aligned_prediction,
            prediction_chain
        )


        coords = []
        residue_ids = []


        for residue in chain:

            res_id = residue.id[1]


            for atom in residue:

                if atom.element == "H":
                    continue


                coords.append(atom.coord)

                residue_ids.append(res_id)



        coords = np.asarray(coords)



        tree = cKDTree(coords)


        pairs = tree.query_pairs(
            r=cutoff
        )


        clash_pairs = 0


        for i,j in pairs:


            res_i = residue_ids[i]
            res_j = residue_ids[j]


            # same residue
            if res_i == res_j:
                continue


            # peptide bond neighbors
            if abs(res_i-res_j)==1:
                continue


            clash_pairs += 1



        return {

            "internal_clashes": clash_pairs

        }
    
    # ==========================
    # FULL VALIDATION
    # ==========================

    def validate(
        self,
        reference_chain="C",
        prediction_chain="A",
        dna_chains=("A", "B"),
        rmsd_cutoff=2.5,
        clashscore_cutoff=10.0,
        internal_clash_cutoff=0
):

    # -----------------------
    # Alignment
    # -----------------------
        aligned_pred, rmsd = self.align_prediction(
        reference_chain,
        prediction_chain
    )

    # -----------------------
    # Interface clashes
    # -----------------------
        interface = self.calculate_interface_clashes(
        aligned_pred,
        target_chains=dna_chains,
        prediction_chain=prediction_chain
    )

    # -----------------------
    # Binder internal clashes
    # -----------------------
        internal = self.calculate_internal_clashes(
        aligned_pred,
        prediction_chain=prediction_chain
    )

    # -----------------------
    # Results
    # -----------------------
        result = {

        "rmsd": rmsd,

        # -------- Interface --------
        "interface_clash_pairs":
            interface["interface_clash_pairs"],

        "interface_clashscore":
            interface["interface_clashscore"],

        "interface_overlap_sum":
            interface["overlap_sum"],

        "interface_max_overlap":
            interface["max_overlap"],

        # -------- Internal --------
        "internal_clashes":
            internal["internal_clashes"],

        # -------- Pass / Fail --------
        "pass_rmsd":
            rmsd < rmsd_cutoff,

        "pass_interface":
            interface["interface_clashscore"] < clashscore_cutoff,

        "pass_internal":
            internal["internal_clashes"] <= internal_clash_cutoff,
    }

        result["pass"] = (
            result["pass_rmsd"]
            and
            result["pass_interface"]
            and
            result["pass_internal"]
    )

        return result







files=get_txt_files_recursive(inputresidue_directory)
na_length=28#外部输入核苷酸长度
for file in files:
    
    dict_sequence = read_dict_from_txt(file)  
    print(file) 
    monomer_predict(dict_sequence,na_length,file,AF2_model_path)#这里是外部接口，指定AF2权重文件夹


residue_files=get_pdb_files_recursive(inputresidue_directory)
rfd3_files=get_cif_files_recursive(inputcif_directory)
cif_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in rfd3_files}
#print(cif_dict)
for input_pdb in residue_files:
        #relax
    base = os.path.basename(input_pdb)
    no_ext = os.path.splitext(base)[0]
    pdb_stem = no_ext.rsplit('_Sequence_', 1)[0] #识别的文件同名的关键
    #print(pdb_stem)
    cif_path = cif_dict.get(pdb_stem)
    #print(cif_path)
    if cif_path is None:
        continue
    else:
        afterrelax_pdb = os.path.splitext(input_pdb)[0] + "_afterrelax.pdb"   
        openmm_relax(input_pdb,afterrelax_pdb)
        #print(afterrelax_pdb)
        validator = StructureValidator(
            reference_file=cif_path,
            prediction_file=afterrelax_pdb
    )
        print(validator.reference_file)
        print(validator.prediction_file)


        result = validator.validate(
            reference_chain="C",
            prediction_chain="A",
            dna_chains=("A", "B"),
            rmsd_cutoff=2.5
)
        print(result)
        if result["pass"]:
            print(f"Validation passed for {afterrelax_pdb}")
            # 确保目标文件夹存在（否则报错）
            os.makedirs(os.path.dirname(output_directory), exist_ok=True)


            filename = os.path.basename(afterrelax_pdb) 
            dest_path = os.path.join(output_directory, filename)
            os.rename(afterrelax_pdb, dest_path)
      
        
        
        