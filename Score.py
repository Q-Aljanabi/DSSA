from pathlib import Path
import os
from typing import  List
import math
import pickle, gzip
from pydantic import BaseModel
import torch
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from DSSA_model import DSSA
from  data_preprocess import smiles_to_graph




SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
MODEL_DIR = SCRIPT_DIR

print(f"Script directory: {SCRIPT_DIR}")
print(f"Base directory: {BASE_DIR}")
print(f"Model directory: {MODEL_DIR}")
print(f"Current working directory: {os.getcwd()}")


model_name = "best_model.pth"
possible_paths = [
    os.path.join(SCRIPT_DIR, model_name),
    os.path.join(BASE_DIR, model_name),
    os.path.join(os.getcwd(), model_name),
    model_name
]

print("Checking model file in these locations:")
for path in possible_paths:
    exists = os.path.exists(path)
    print(f"  - {path}: {'Found' if exists else 'Not found'}")


print("\nListing files in script directory:")
try:
    for file in os.listdir(SCRIPT_DIR):
        if file.endswith('.pth'):
            print(f"  - {file}")
except Exception as e:
    print(f"Error listing files: {e}")


def numBridgeheadsAndSpiro(mol):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro

def numMacroAndMulticycle(mol, nAtoms):
    ri = mol.GetRingInfo()
    nMacrocycles = 0
    multi_ring_atoms = {i:0 for i in range(nAtoms)}
    for ring_atoms in ri.AtomRings():
        if len(ring_atoms) > 6:
            nMacrocycles += 1
        for atom in ring_atoms:
            multi_ring_atoms[atom] += 1
    nMultiRingAtoms = sum([v-1 for k, v in multi_ring_atoms.items() if v > 1])
    return nMacrocycles, nMultiRingAtoms

class pre_score():
    def __init__(self, reaction_from='uspto', buildingblock_from='emolecules', 
                 frag_penalty=-6.0, complexity_buffer=1.0):
        # Fixed: Changed the order to match your actual file name
        pickle_filename = 'datasets_%s_%s.pkl.gz' % (buildingblock_from, reaction_from)
        
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(current_dir, pickle_filename),
            os.path.join(current_dir, 'pickle', pickle_filename),
            os.path.join(current_dir, 'datasets', 'pickle', pickle_filename)
        ]
        
        pickle_path = None
        for path in possible_paths:
            if os.path.exists(path):
                pickle_path = path
                break
                
        if pickle_path is None:
            raise FileNotFoundError(f"Could not find pickle file {pickle_filename} in any of these locations: {possible_paths}")
            
        self._fscores = pickle.load(gzip.open(pickle_path))
        self.frag_penalty = frag_penalty
        self.max_score = 0
        self.min_score = frag_penalty - complexity_buffer
        
    def calculateScore(self, smi):
        sascore = 0        
        m = Chem.MolFromSmiles(smi)
        contribution = {}
        
        # fragment score
        bi = {}
        fp = rdMolDescriptors.GetMorganFingerprint(m, 2, useChirality=True, bitInfo=bi)

        fps = fp.GetNonzeroElements()
        score1 = 0.
        nf, nn = 0, 0
        for bitId, vs in bi.items():
            if vs[0][1] != 2:
                continue
            fscore = self._fscores.get(bitId, self.frag_penalty)
            if fscore < 0:
                nf += 1
                score1 += fscore
                for v in vs:
                    contribution[v[0]] = fscore
            if fscore == self.frag_penalty:
                nn += len(vs)
        if nf != 0:
            score1 /= nf
        sascore += score1

        # features score
        nAtoms = m.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
        nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m)
        nMacrocycles, nMulticycleAtoms = numMacroAndMulticycle(m, nAtoms)
            
        sizePenalty = nAtoms**1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = math.log10(2) if nMacrocycles > 0 else 0
        multicyclePenalty = math.log10(nMulticycleAtoms + 1)

        score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty - multicyclePenalty
        sascore += score2
           
        # correction for fingerprint density
        score3 = 0.
        if nAtoms > len(fps):
            score3 = math.log(float(nAtoms) / len(fps)) * .999
        sascore += score3

        # transform to scale between 0 and 1
        sascore = 1 - (sascore - self.min_score) / (self.max_score - self.min_score)            
        sascore = max(0., min(1., sascore))
        
        return sascore, contribution
    

class DSSAPredictor:
    def __init__(self, model_path: str = "best_model.pth"):
         import os
         
         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
         print(f"Using device: {self.device}")
         
         # Try multiple possible locations for the model file
         possible_paths = [
             model_path,  # Try the provided path first
             os.path.join(os.path.dirname(__file__), model_path),  # Same directory as this file
             os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path),  # Absolute path to same directory
             os.path.join(Path(__file__).parent, model_path),  # Using pathlib
             os.path.join(SCRIPT_DIR, model_path),  # Using defined SCRIPT_DIR
             os.path.join(BASE_DIR, "models", model_path),  # In models subdirectory of base dir
             os.path.join(MODEL_DIR, model_path)  # Using predefined MODEL_DIR
         ]
         
         # Print debug info
         print(f"Looking for model in these locations:")
         for path in possible_paths:
             print(f"  - {path} (exists: {os.path.exists(path)})")
         
         # Try to load from each possible location
         model_loaded = False
         for path in possible_paths:
             if os.path.exists(path):
                 try:
                     print(f"Loading model from: {path}")
                     self.model = self._load_gnn_model(path)
                     model_loaded = True
                     break
                 except Exception as e:
                     print(f"Error loading from {path}: {str(e)}")
         
         if not model_loaded:
             raise FileNotFoundError(f"Model file {model_path} not found in any of the searched locations")
         
         self.prepeard_scorer = pre_score()
         self.gnn_weight = 0.01
         self.br_weight = 0.81
         self.binary_threshold = 5.0
 
    def _load_gnn_model(self, model_path: str):
        checkpoint = torch.load(model_path, map_location=self.device)
        model = DSSA(
            in_dim=12,
            hidden_dim=128,
            num_layers=4,
            num_heads=4,
            dropout=0,
            num_classes=1,
            num_node_types=13,
            num_edge_types=5,
            processing_steps=3
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def predict(self, smiles: str, return_details: bool = False):
        prepeard_score, contribution = self.prepeard_scorer.calculateScore(smiles)
        prepeard_score = 1 + 9 * prepeard_score
        
        graph = smiles_to_graph(smiles).to(self.device)
        with torch.no_grad():
            raw_gnn_score = self.model(graph).cpu().numpy()[0]
        
        min_val = -12.0
        max_val = -9.5
        gnn_score = (raw_gnn_score - min_val) / (max_val - min_val)
        gnn_score = 1 + 9 * np.clip(gnn_score, 0, 1)
        
        combined_score = (self.gnn_weight * gnn_score + self.br_weight * prepeard_score)
        binary_class = 1 if combined_score >= self.binary_threshold else 0
        es_score = np.clip(combined_score / 10, 0, 1)
        hs_score = 1 - es_score

        if return_details:
            return {
                'combined_score': float(combined_score),
                'binary_class': binary_class,
                'es_score': float(es_score),
                'hs_score': float(hs_score),
                'gnn_score': float(gnn_score),
                'gnn_raw': float(raw_gnn_score),
                'contribution': contribution
            }
        return float(combined_score)

    def batch_predict(self, smiles_list: List[str], batch_size: int = 32):
        results = {}
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i + batch_size]
            for smi in batch:
                result = self.predict(smi, return_details=True)
                if result is not None:
                    results[smi] = result
        return results

class MoleculeInput(BaseModel):
    smiles: str



def main():
    predictor = DSSAPredictor("best_model.pth")
    test_smiles = [
               'c12c(OC)ccc3CCN([C@@H](Cc4c(cc(c(c4)OC)OC)O2)c13)C',
               '[C@@]12(CC)[C@H](n3c(c(CCO)c4ccccc43)CC2)NCCC1',
               '[C@H]1([C@H]([C@H](C=C(C)/C=C/C(C(C)C(C)=O)OC)[C@@H](OC)O1)C)C(C)=CC=CC(CCCC)C',
               ' C1CCCCCCN(C=O)CC[C@@H]2[C@@H]3C[C@]4(CCC1)C(C2=O)=C[C@H](O)CC=CCC=CCCCN(C4)C3=O',
               ' C([C@H]1N2[C@H](CCC2)[C@H](O)[C@@H]1O)O',
              ' C(C[C@H]1[C@H]([C@H](C)OC1=O)O)CCCCCCCCCCC1=C[C@@H](OC1=O)C',
              '  O=C(C)N[C@H]1[C@@H](O)C=C(CO)[C@@H](O)[C@@H]1O',
              '  O=C1C=COC21CC(OC2)=O',
              '  O[C@H]1[C@H]([C@@H]([C@@H](CO)O[C@H]1OC(CC(=O)O)CC(CCCCC)OC(CC(CC(O)CCC)O)=O)O)O',
               ' CC([C@@H](CC[C@](C(=C)Cl)(CBr)Cl)Br)(C)Cl',
               ' C1C(CCC)NC(CC1)C',
               ' C(Cl)C#CC=C=CCO',
               ' C(=CC(CC1(C)OC(=O)CC1)=O)(C)C',
                'CCCCCCCCCCC[C@]1(OC(CCC1)=O)CO',
               ' O[C@H](C[C@H]1NCCCC1)C',
               ' C1CC[C@@H](CCC)O[C@@H](CCCCCC)CC1',
                'C1CC[C@@H]2[C@H]1CCCCC2',
                'C(CC1(CN)CCCCC1)(=O)O'

    ]

    
    
    results = predictor.batch_predict(test_smiles)
    for smi, result in results.items():
        # print(f"\nMolecule: {smi}")
        print(f" {result['combined_score']:.2f}")
        # print(f"Binary Class: {result['binary_class']} (0=Hard, 1=Easy)")
        # print(f"ES Score: {result['es_score']:.3f}")
        # print(f"HS Score: {result['hs_score']:.3f}")
        # result = predictor.predict(smi, return_details=True)
    # Predict scores for test_smiles
  
    # # Extract model_scores from results
    # model_scores = []
    # for smi, result in results.items():
    #     if isinstance(result, dict):
    #         model_scores.append(result['combined_score'])  # Assuming 'combined_score' is the key for the predicted score

    # # Chemist scores (ground truth)
    # chemist_scores = [
    #     3.56, 7.00, 3.00, 4.67, 2.33, 7.56, 7.11, 1.56, 9.11, 3.89, 7.33, 1.78, 1.89, 1.11, 8.44, 7.44, 8.44, 8.00,
    #     2.11, 3.78, 1.00, 4.11, 2.00, 8.44, 1.22, 1.33, 6.44, 8.67, 6.89, 9.22, 1.00, 7.22, 8.78, 1.22, 5.22, 4.00,
    #     3.78, 3.78, 8.78, 1.67
    # ]

    # # Ensure model_scores and chemist_scores have the same length
    # if len(model_scores) != len(chemist_scores):
    #     raise ValueError("Length of model_scores and chemist_scores must be the same.")

    # # Calculate R-squared
    # ss_res = sum((chemist_scores[i] - model_scores[i]) ** 2 for i in range(len(model_scores)))
    # ss_tot = sum((y - sum(chemist_scores) / len(chemist_scores)) ** 2 for y in chemist_scores)
    # r_squared = 1 - (ss_res / ss_tot)

    # # Filter complex molecules and calculate average difference
    # complex_molecules = [
    #     {"model_score": model_scores[i], "chemist_score": chemist_scores[i], "diff": chemist_scores[i] - model_scores[i]}
    #     for i in range(len(model_scores)) if chemist_scores[i] > 7
    # ]
    # avg_diff = sum(molecule["diff"] for molecule in complex_molecules) / len(complex_molecules)

    # # Print results
    # print(f"R^2 value: {r_squared:.4f}")
    # print(f"Average difference for complex molecules (chemist - model): {avg_diff:.2f}")

if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     input_path = "best_model.pth"
#     output_path = "best_model_cpu.pth"
#     convert_model_to_cpu('best_model.pth', 'best_model_cpu.pth')
#     model_path = 'best_model_cpu.pth'
    
#     print("Model converted successfully!")