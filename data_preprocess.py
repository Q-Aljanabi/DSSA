import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
import logging
import random
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


class CustomMoleculeTokenizer:
    def __init__(self):
        self.vocab = {
            'PAD': 0, 'UNK': 1, 'CLS': 2, 'SEP': 3,
            'C': 4, 'N': 5, 'O': 6, 'F': 7, 'S': 8, 'Cl': 9, 'Br': 10,
            'I': 11, 'P': 12, 'B': 13, 'Si': 14, 'Se': 15,
            '(': 16, ')': 17, '[': 18, ']': 19, '=': 20, '#': 21,
            '+': 22, '-': 23, '.': 24, '/': 25, '\\': 26,
            '1': 27, '2': 28, '3': 29, '4': 30, '5': 31, '6': 32,
            'c': 33, 'n': 34, 'o': 35, 'p': 36, 's': 37
        }
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.max_length = 128

    def tokenize(self, smiles):
        """Convert SMILES string to tokens"""
        tokens = []
        i = 0
        while i < len(smiles):
            if i + 1 < len(smiles) and smiles[i:i+2] in self.vocab:
                tokens.append(smiles[i:i+2])
                i += 2
            else:
                if smiles[i] in self.vocab:
                    tokens.append(smiles[i])
                else:
                    tokens.append('UNK')
                i += 1
        return tokens

    def encode(self, smiles, max_length=None):
        """Convert SMILES to token IDs with padding"""
        if max_length is None:
            max_length = self.max_length
        tokens = self.tokenize(smiles)
        tokens = ['CLS'] + tokens + ['SEP']
        ids = [self.vocab.get(token, self.vocab['UNK']) for token in tokens]
        padding_length = max_length - len(ids)
        if padding_length > 0:
            ids = ids + [self.vocab['PAD']] * padding_length
        else:
            ids = ids[:max_length]
        attention_mask = [1] * min(len(tokens), max_length) + [0] * max(0, padding_length)
        return {
            'input_ids': torch.tensor([ids], dtype=torch.long),
            'attention_mask': torch.tensor([attention_mask], dtype=torch.long)
        }


class MoleculeProcessor:
    def __init__(self):
        self.atom_types = {
            'C': 0, 'N': 1, 'O': 2, 'F': 3, 'S': 4, 'Cl': 5, 'Br': 6,
            'I': 7, 'P': 8, 'B': 9, 'Si': 10, 'Se': 11, 'other': 12
        }
        self.bond_types = {
            Chem.rdchem.BondType.SINGLE: 0,
            Chem.rdchem.BondType.DOUBLE: 1,
            Chem.rdchem.BondType.TRIPLE: 2,
            Chem.rdchem.BondType.AROMATIC: 3,
            'other': 4
        }

    def get_atom_features(self, atom):
        """Enhanced atom feature extraction"""
        features = [
            atom.GetAtomicNum(),
            atom.GetTotalDegree(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            atom.GetNumRadicalElectrons(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            atom.GetDegree(),
            atom.GetImplicitValence(),
            atom.GetExplicitValence(),
            int(atom.GetChiralTag()),
        ]
        return features

    def get_bond_features(self, bond):
        """Enhanced bond feature extraction"""
        features = [
            float(bond.GetBondTypeAsDouble()),
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
            int(bond.GetStereo()),
            float(bond.GetIsAromatic())
        ]
        return features


class RobustMoleculeDataset(Dataset):
    def __init__(self, dataframe, smiles_column, target_column):
        self.dataframe = dataframe
        self.smiles_column = smiles_column
        self.target_column = target_column
        self.processor = MoleculeProcessor()
        self.tokenizer = CustomMoleculeTokenizer()
        
    def __len__(self):
        return len(self.dataframe)
        
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        smiles = row[self.smiles_column]
        target = row[self.target_column]
        
        try:
            encoded = self.tokenizer.encode(smiles)
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Warning: Could not process SMILES {smiles}, using fallback")
                return self._create_fallback_molecule(target)
            AllChem.Compute2DCoords(mol)
            atom_features = []
            node_types = []
            for atom in mol.GetAtoms():
                atom_features.append(self.processor.get_atom_features(atom))
                node_types.append(self.processor.atom_types.get(atom.GetSymbol(), 
                                                              self.processor.atom_types['other']))
            
            x = torch.tensor(atom_features, dtype=torch.float)
            node_types = torch.tensor(node_types, dtype=torch.long)
            edge_indices = []
            edge_features = []
            edge_types = []
            
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_indices += [[i, j], [j, i]]
                
                bond_feature = self.processor.get_bond_features(bond)
                edge_features += [bond_feature, bond_feature]
                
                bond_type = self.processor.bond_types.get(bond.GetBondType(), 
                                                        self.processor.bond_types['other'])
                edge_types += [bond_type, bond_type]
            
            if not edge_indices: 
                edge_indices = [[0, 0]]
                edge_features = [[1.0, 0.0, 0.0, 0.0, 0.0]]
                edge_types = [self.processor.bond_types['other']]
            
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            edge_types = torch.tensor(edge_types, dtype=torch.long)
            
            return Data(
                x=x, 
                edge_index=edge_index, 
                edge_attr=edge_attr,
                node_types=node_types,
                edge_types=edge_types,
                y=torch.tensor([target], dtype=torch.long),
                smiles_input_ids=input_ids,
                smiles_attention_mask=attention_mask
            )
        except Exception as e:
            print(f"Error processing molecule {idx} ({smiles}): {str(e)}")
            return self._create_fallback_molecule(target, input_ids, attention_mask)

    def _create_fallback_molecule(self, target, input_ids=None, attention_mask=None):
        """Create a fallback molecule with type information"""
        x = torch.tensor([[6, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.float)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float)
        node_types = torch.tensor([self.processor.atom_types['C']], dtype=torch.long)
        edge_types = torch.tensor([self.processor.bond_types['other']], dtype=torch.long)
        if input_ids is None:
            input_ids = torch.zeros(self.tokenizer.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.tokenizer.max_length, dtype=torch.long)
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_types=node_types,
            edge_types=edge_types,
            y=torch.tensor([target], dtype=torch.long),
            smiles_input_ids=input_ids,
            smiles_attention_mask=attention_mask
        )


def balance_dataset(df, target_column):
    print("Original dataset shape:", Counter(df[target_column]))
    under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = under_sampler.fit_resample(df.drop(columns=[target_column]), df[target_column])
    df_balanced = pd.concat([X_resampled, y_resampled], axis=1)
    print("Balanced dataset shape:", Counter(df_balanced[target_column]))
    return df_balanced


class GraphSMILESDataset:
    def __init__(self, graph_dataset, smiles_strings):
        """
        Combined dataset for graph and SMILES data
        
        Args:
            graph_dataset: PyG dataset containing graph data
            smiles_strings: List of SMILES strings corresponding to each graph
        """
        self.graph_dataset = graph_dataset
        self.smiles_strings = smiles_strings
        if len(graph_dataset) != len(smiles_strings):
            raise ValueError(f"Mismatch in dataset lengths: {len(graph_dataset)} graphs vs {len(smiles_strings)} SMILES")
    
    def __len__(self):
        return len(self.graph_dataset)
    
    def __getitem__(self, idx):
        return self.graph_dataset[idx], self.smiles_strings[idx]


def custom_collate_fn(batch):
    graph_data = []
    smiles_input_ids = []
    smiles_attention_masks = []
    labels = []
    
    for data in batch:
        graph_data.append(Data(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            node_types=data.node_types,
            edge_types=data.edge_types
        ))
        smiles_input_ids.append(data.smiles_input_ids)
        smiles_attention_masks.append(data.smiles_attention_mask)
        labels.append(data.y)
    
    graph_batch = Batch.from_data_list(graph_data)
    smiles_input_ids = torch.stack(smiles_input_ids)
    smiles_attention_masks = torch.stack(smiles_attention_masks)
    labels = torch.stack(labels)
    
    return {
        'graph_data': graph_batch,
        'smiles_input_ids': smiles_input_ids,
        'smiles_attention_masks': smiles_attention_masks,
        'labels': labels
    }


def worker_init_fn(worker_id):
    """Initialize seeds for DataLoader workers"""
    np.random.seed(42 + worker_id)
    random.seed(42 + worker_id)


def create_optimized_dataloaders(datasets, batch_size=256, num_workers=4):
    loaders = {}
    dataset_names = ['train', 'val', 'test', 'ts1', 'ts2', 'ts3']
    
    base_params = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
        'persistent_workers': True if num_workers > 0 else False,
        'worker_init_fn': worker_init_fn
    }
    
    for name, dataset in zip(dataset_names, datasets):
        if dataset is not None:
            is_train = (name == 'train')
            loaders[name] = PyGDataLoader(
                dataset,
                shuffle=is_train,
                drop_last=is_train,
                **base_params
            )
            logging.info(f"Created {name} loader with {len(dataset)} samples")
    return loaders


# =============================================================================
# GRBSA COMPATIBLE FUNCTIONS
# =============================================================================

def get_atom_features(atom):
    """
    Convert atom to feature vector - exactly 12 features to match GRBSA model
    
    USED BY: GraphNeuralNetworkPredictor in GRBSA
    """
    try:
        features = []
        
        # Atomic number (one-hot encoded) - 6 types
        atomic_nums = [6, 7, 8, 9, 15, 16]  # C, N, O, F, P, S
        features.extend([1 if atom.GetAtomicNum() == num else 0 for num in atomic_nums])
        
        # Add 6 additional features
        features.extend([
            atom.GetDegree() / 4,           # Normalized degree
            atom.GetTotalNumHs() / 4,       # Normalized number of Hs
            atom.GetImplicitValence() / 4,  # Normalized implicit valence
            atom.GetIsAromatic() * 1.0,     # Aromaticity
            atom.GetFormalCharge() / 2.0,   # Normalized formal charge
            len(atom.GetNeighbors()) / 4.0  # Normalized number of neighbors
        ])
        
        return torch.tensor(features, dtype=torch.float)
    except Exception as e:
        print(f"Error extracting atom features: {str(e)}")
        return torch.tensor([0] * 12, dtype=torch.float)  # Return zero features as fallback


def get_bond_features(bond):
    """
    Convert bond to feature vector - 5 features total
    
    USED BY: GraphNeuralNetworkPredictor in GRBSA
    """
    try:
        features = []
        
        # Bond type (one-hot encoding) - 4 features
        bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                     Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        features.extend([1.0 if bond.GetBondType() == t else 0.0 for t in bond_types])
        
        # Ring membership - 1 feature
        features.append(float(bond.IsInRing()))
        
        return torch.tensor(features, dtype=torch.float)
    except Exception as e:
        print(f"Error extracting bond features: {str(e)}")
        return torch.tensor([0] * 5, dtype=torch.float)  # Return zero features as fallback


def smiles_to_graph(smiles):
    """
    Convert SMILES string to PyTorch Geometric Data object
    
    USED BY: GraphNeuralNetworkPredictor.predict_graph_score() in GRBSA
    This is the MAIN function called by GRBSA for graph conversion
    """
    try:
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
            
        # Get node features
        node_features = []
        for atom in mol.GetAtoms():
            node_features.append(get_atom_features(atom))
        x = torch.stack(node_features)
        
        # Get edge features
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Add edges in both directions
            edge_indices.extend([[i, j], [j, i]])
            
            # Add edge features (same for both directions)
            bond_feature = get_bond_features(bond)
            edge_features.extend([bond_feature, bond_feature])
        
        if edge_indices:  # If molecule has bonds
            edge_index = torch.tensor(edge_indices).t()
            edge_attr = torch.stack(edge_features)
        else:  # Handle molecules with no bonds
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 5), dtype=torch.float)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(node_features)
        )
        
        return data
        
    except Exception as e:
        print(f"Error converting SMILES to graph: {str(e)}")
        raise


def MOlINFo():
    """
    Load model configuration parameters
    
    USED BY: GRBSA initialization to get model dimensions
    Returns configuration for the Graph Neural Network component
    """
    # Note: in_dim is now 12 to match the GRBSA feature dimension
    return (None, None, None, 12, 5, 13, 5, None, {
        'vocab_size': 38,
        'pad_token_id': 0,
        'max_length': 128,
        'vocab': {
            'PAD': 0, 'UNK': 1, 'CLS': 2, 'SEP': 3, 'C': 4, 'N': 5, 'O': 6,
            'F': 7, 'S': 8, 'Cl': 9, 'Br': 10, 'I': 11, 'P': 12, 'B': 13,
            'Si': 14, 'Se': 15, '(': 16, ')': 17, '[': 18, ']': 19, '=': 20,
            '#': 21, '+': 22, '-': 23, '.': 24, '/': 25, '\\': 26, '1': 27,
            '2': 28, '3': 29, '4': 30, '5': 31, '6': 32, 'c': 33, 'n': 34,
            'o': 35, 'p': 36, 's': 37
        }
    })


def dataload():
    print("Starting dataload function...")
    
    # Define the dataset directory relative to the current script
    dataset_dir = os.path.join(os.path.dirname(__file__), "datasets")

    # Load datasets using relative paths
    df_dataset = pd.read_csv(os.path.join(dataset_dir, "dataset.csv"))
    df_ts1 = pd.read_csv(os.path.join(dataset_dir, "TS1.csv"))
    df_ts2 = pd.read_csv(os.path.join(dataset_dir, "TS2.csv"))
    df_ts3 = pd.read_csv(os.path.join(dataset_dir, "TS3.csv"))

    print(f"Dataset shapes: Main: {df_dataset.shape}, TS1: {df_ts1.shape}, TS2: {df_ts2.shape}, TS3: {df_ts3.shape}")
    df_balanced = balance_dataset(df_dataset, 'labels')
    train, temp = train_test_split(df_balanced, test_size=0.2, random_state=42, stratify=df_balanced['labels'])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['labels'])
    train_dataset = RobustMoleculeDataset(train, smiles_column='smiles', target_column='labels')
    val_dataset = RobustMoleculeDataset(val, smiles_column='smiles', target_column='labels')
    test_dataset = RobustMoleculeDataset(test, smiles_column='smiles', target_column='labels')
    ts1_dataset = RobustMoleculeDataset(df_ts1, smiles_column='smiles', target_column='labels')
    ts2_dataset = RobustMoleculeDataset(df_ts2, smiles_column='smiles', target_column='labels')
    ts3_dataset = RobustMoleculeDataset(df_ts3, smiles_column='smiles', target_column='labels')
    sample_data = train_dataset[0]
    num_node_features = sample_data.num_node_features
    num_edge_features = sample_data.num_edge_features
    num_node_types = len(MoleculeProcessor().atom_types)
    num_edge_types = len(MoleculeProcessor().bond_types)
    class_weights = compute_class_weight('balanced', 
                                       classes=np.unique(df_balanced['labels']), 
                                       y=df_balanced['labels'])
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    
    tokenizer = CustomMoleculeTokenizer()
    
    return (train_dataset, val_dataset, test_dataset, 
            ts1_dataset, ts2_dataset, ts3_dataset, 
            num_node_features, num_edge_features,
            num_node_types, num_edge_types, class_weights,
            {
                'vocab_size': len(tokenizer.vocab),
                'pad_token_id': tokenizer.vocab['PAD'],
                'max_length': tokenizer.max_length,
                'vocab': tokenizer.vocab
            })