from rdkit.Chem.rdchem import Mol as RDMol
from rdkit.Chem import Descriptors, Crippen, AllChem, rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumRotatableBonds, CalcFractionCSP3
from rdkit import Chem
from rdkit import DataStructs
import torch
import pickle
import numpy as np
from torch.nn.functional import softplus
from gflownet.utils import sascore
from rdkit.Chem import QED
from typing import List
from utils.maplight import *
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def scale_range(OldValue, OldMax, OldMin, NewMax, NewMin):
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    if OldRange == 0:
        return NewMin
    NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    return NewValue

def calculate_tanimoto_similarity(mol1, mol2):
    """Calculate the Tanimoto similarity between two molecules."""
    if mol1 is None or mol2 is None:
        return 0.0
    fp_gen = rdFingerprintGenerator.GetRDKitFPGenerator()
    fp1 = fp_gen.GetFingerprint(mol1)
    fp2 = fp_gen.GetFingerprint(mol2)
    return DataStructs.FingerprintSimilarity(fp1, fp2)

class Reward():
    def __init__(self, conditional_range_dict, cond_prop_var, reward_aggregation, molenv_dict_path, zinc_rad_scale, hps, wrapped_glidecnn_model=None) -> None:
        self.cond_range = conditional_range_dict
        self.cond_var = cond_prop_var
        self.reward_aggregation = reward_aggregation
        self.zinc_rad_scale = zinc_rad_scale
        with open(molenv_dict_path, "rb") as f:
            self.atomenv_dictionary = pickle.load(f)
        self.glidecnn_model = wrapped_glidecnn_model
        self.hps = hps
        if 'seed_smiles' in self.hps:
            self.seedmol = Chem.MolFromSmiles(hps['seed_smiles'])
        else:
            self.seedmol = None

    def mol_wt(self, mols: List[RDMol]):
        return np.array([Descriptors.MolWt(mol) for mol in mols])
    
    def qed(self, mols: List[RDMol]):
        return np.array([QED.qed(mol) for mol in mols])
    
    def logP(self, mols: List[RDMol]):
        return np.array([Crippen.MolLogP(mol) for mol in mols])
    
    def tpsa(self, mols: List[RDMol]):
        return np.array([CalcTPSA(mol) for mol in mols])
    
    def fsp3(self, mols: List[RDMol]):
        return np.array([CalcFractionCSP3(mol) for mol in mols])
    
    def count_rotatable_bonds(self, mols: List[RDMol]):
        return np.array([CalcNumRotatableBonds(mol) for mol in mols])
    
    def count_num_rings(self, mols: List[RDMol]):
        def count_5_and_6_membered_rings(mol):
            ring_info = mol.GetRingInfo()
            return len([ring for ring in ring_info.AtomRings() if len(ring) in (5, 6)])
        return np.array([count_5_and_6_membered_rings(mol) for mol in mols])
    
    def synthetic_assessibility(self,  mols: List[RDMol]):
        return np.array([sascore.calculateScore(mol) for mol in mols])

    def permeability(self, caco2_model, mols: List[RDMol]):
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        mol_fing = get_fingerprints(pd.Series(smiles_list))
        y_pred = self.Y_scaler.inverse_transform(caco2_model.predict(mol_fing, thread_count=32)).reshape(-1, 1)
        return y_pred
    
    def toxicity(self, tox_model, mols: List[RDMol]):
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        mol_fing = get_fingerprints(pd.Series(smiles_list))
        y_pred = self.Y_scaler.inverse_transform(tox_model.predict(mol_fing, thread_count=32)).reshape(-1, 1)
        return y_pred

    def lipo(self, task_model, mols):
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        mol_fing = get_fingerprints(pd.Series(smiles_list))
        y_pred = self.Y_scaler.inverse_transform(task_model.predict(mol_fing, thread_count=32)).reshape(-1, 1)
        return y_pred

    def sol(self, task_model, mols):
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        mol_fing = get_fingerprints(pd.Series(smiles_list))
        y_pred = self.Y_scaler.inverse_transform(task_model.predict(mol_fing, thread_count=32)).reshape(-1, 1)
        return y_pred

    def bind(self, task_model, mols):
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        mol_fing = get_fingerprints(pd.Series(smiles_list))
        y_pred = self.Y_scaler.inverse_transform(task_model.predict(mol_fing, thread_count=32)).reshape(-1, 1)
        return y_pred

    def mclear(self, task_model, mols):
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        mol_fing = get_fingerprints(pd.Series(smiles_list))
        y_pred = self.Y_scaler.inverse_transform(task_model.predict(mol_fing, thread_count=32)).reshape(-1, 1)
        return y_pred

    def hclear(self, task_model, mols):
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        mol_fing = get_fingerprints(pd.Series(smiles_list))
        y_pred = self.Y_scaler.inverse_transform(task_model.predict(mol_fing, thread_count=32)).reshape(-1, 1)
        return y_pred

    def docking_reward(self, mols, seed_mol):
        if self.glidecnn_model:
            device = next(self.glidecnn_model.parameters()).device
            docking_scores = glide_cnn_scores(self.glidecnn_model, device, mols, batch_size=64)
            # Sigmoid normalization: Squashes scores to (0, 1). Lower (more negative) is better.
            docking_scores_normalized = 1.0 / (1.0 + np.exp((docking_scores + 8.0) / 2.0))
        else:
            docking_scores = np.zeros(len(mols))
            docking_scores_normalized = np.zeros(len(mols))
        
        target_mol = seed_mol if seed_mol is not None else self.seedmol
        if target_mol is not None:
            sim = np.array([calculate_tanimoto_similarity(mol, target_mol) for mol in mols])
            # Diversity reward: penalize identical molecules
            sim_reward = np.where(sim >= 0.95, 0.0, sim)
        else:
            sim = np.zeros(len(mols))
            sim_reward = np.ones(len(mols))

        docking_flatreward = docking_scores_normalized * sim_reward
        return docking_flatreward, docking_scores, sim 
    
    def unidocking_reward(self, mols, seed_mol):
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        docking_scores = np.array(unidock_scores(smiles_list)).astype(np.float64)
        # Using improved sigmoid normalization for stability
        docking_scores_normalized = 1.0 / (1.0 + np.exp((docking_scores + 8.0) / 2.0))
        
        target_mol = seed_mol if seed_mol is not None else self.seedmol
        if target_mol is not None:
            sim = np.array([calculate_tanimoto_similarity(mol, target_mol) for mol in mols])
            sim_reward = np.where(sim >= 0.95, 0.0, sim)
        else:
            sim = np.zeros(len(mols))
            sim_reward = np.ones(len(mols))
            
        docking_flatreward = docking_scores_normalized * sim_reward
        return docking_flatreward, docking_scores, sim

    def energy_reward(self, energy_classfn_model, energy_regressn_model, mols: List[RDMol], batch_size=None):
        # Placeholder compatible with original structure
        return np.zeros(len(mols)), np.zeros(len(mols))

    def searchAtomEnvironments_fraction(self,  mols: List[RDMol], radius=2):
        def per_mol_fraction(mol, radius):
            info = {}
            atomenvs = 0
            AllChem.GetMorganFingerprint(mol, radius, bitInfo=info, includeRedundantEnvironments=True, useFeatures=False)
            for k, v in info.items():
                for e in v:
                    if e[1] == radius and k in self.atomenv_dictionary:
                        atomenvs += 1
            return atomenvs / max(mol.GetNumAtoms(), 1)
        epsilon = 1e-6
        return [per_mol_fraction(mol, radius) + epsilon for mol in mols]

    def _compositional_reward(self, flat_reward, property, range, slope, rate=1):
        if property == 'zinc_radius':
            return flat_reward
        
        composite_reward = []
        for p_x in flat_reward:
            if p_x < range[0]:
                if slope == 1:
                    normalized_reward = 0.5 * np.exp(-(range[0] - p_x) / rate)
                elif slope == -1:
                    normalized_reward = np.exp(-(range[0] - p_x) / rate)
                else:
                    normalized_reward = np.exp(-(range[0] - p_x) / rate)
            elif p_x > range[1]:
                if slope == 1:
                    normalized_reward = np.exp(-(p_x - range[1]) / rate)
                elif slope == -1:
                    normalized_reward = 0.5 * np.exp(-(p_x - range[1]) / rate)
                else:
                    normalized_reward = np.exp(-(p_x - range[1]) / rate)
            else:
                if slope == 1:
                    normalized_reward = 0.5 * ((p_x - range[0]) / (range[1] - range[0])) + 0.5
                elif slope == -1:
                    normalized_reward = -0.5 * ((p_x - range[0]) / (range[1] - range[0])) + 1
                else:
                    normalized_reward = 1.0
            composite_reward.append(normalized_reward)
        return composite_reward

    def _consolidate_rewards(self, flat_rewards):
        lg_rewards = []
        for (flat_reward, property, range, slope) in flat_rewards:
            rate = self.cond_var.get(property, 1.0)
            lg_rewards.append(self._compositional_reward(flat_reward, property, range, slope, rate))
        return np.array(lg_rewards).T

    def _compute_flat_rewards(self, mols):
        flat_rewards = []
        for property in self.cond_range.keys():
            if property == 'Mol_Wt':
                flat_reward = self.mol_wt(mols)
            elif property == 'fsp3':
                flat_reward = self.fsp3(mols)
            elif property == 'logP':
                flat_reward = self.logP(mols)
            elif property == 'num_rot_bonds':
                flat_reward = self.count_rotatable_bonds(mols)
            elif property == 'tpsa':
                flat_reward = self.tpsa(mols)
            elif property == 'num_rings':
                flat_reward = self.count_num_rings(mols)
            elif property == 'sas':
                flat_reward = self.synthetic_assessibility(mols)
            elif property == 'qed':
                flat_reward = self.qed(mols)
            elif property in ['LD50', 'toxicity'] and hasattr(self, "task_model"):
                flat_reward = self.toxicity(self.task_model, mols)
            elif property in ['Caco2', 'permeability'] and hasattr(self, "task_model"):
                flat_reward = self.permeability(self.task_model, mols)
            elif property in ['Lipophilicity', 'logP_pred'] and hasattr(self, "task_model"):
                flat_reward = self.lipo(self.task_model, mols)
            elif property in ['Solubility', 'sol'] and hasattr(self, "task_model"):
                flat_reward = self.sol(self.task_model, mols)
            elif property in ['BindingRate', 'affinity'] and hasattr(self, "task_model"):
                flat_reward = self.bind(self.task_model, mols)
            elif property in ['MicroClearance', 'mclear'] and hasattr(self, "task_model"):
                flat_reward = self.mclear(self.task_model, mols)
            elif property in ['HepatocyteClearance', 'hclear'] and hasattr(self, "task_model"):
                flat_reward = self.hclear(self.task_model, mols)
            else:
                # Dynamic property lookup using RDKit Descriptors (Professional Fallback)
                try:
                    if hasattr(Descriptors, property):
                        func = getattr(Descriptors, property)
                        flat_reward = np.array([func(mol) for mol in mols])
                    else:
                        descriptor_lookup = {d[0].lower(): d[0] for d in Descriptors._descList}
                        if property.lower() in descriptor_lookup:
                            real_name = descriptor_lookup[property.lower()]
                            func = getattr(Descriptors, real_name)
                            flat_reward = np.array([func(mol) for mol in mols])
                        else:
                            logger.warning(f"Property '{property}' not found in hardcoded list or RDKit Descriptors. Skipping.")
                            continue
                except Exception as e:
                    logger.error(f"Error computing dynamic property '{property}': {e}")
                    continue

            flat_rewards.append((flat_reward, property, self.cond_range[property][0], self.cond_range[property][2]))
        
        zinc_radius_flat_reward = self.searchAtomEnvironments_fraction(mols)
        zinc_radius_flat_reward_array = np.array(zinc_radius_flat_reward)
        if self.zinc_rad_scale:
            scaled_zinc_radius_flat_reward_array = zinc_radius_flat_reward_array * self.zinc_rad_scale
        else:
            scaled_zinc_radius_flat_reward_array = zinc_radius_flat_reward_array
            
        if 'zinc_radius' in self.cond_range:
            flat_rewards.append((scaled_zinc_radius_flat_reward_array, "zinc_radius", None, None))
        return flat_rewards, zinc_radius_flat_reward

    def molecular_rewards(self, mols):
        flat_rewards, zinc_rad_flat = self._compute_flat_rewards(mols)
        cons_rew = self._consolidate_rewards(flat_rewards)
        if self.reward_aggregation == "add":
            agg = np.sum(cons_rew, axis=1)
            agg_rew_normal_fn = lambda x: scale_range(x, len(self.cond_range), 0, 1, 0)
            return cons_rew, flat_rewards, agg_rew_normal_fn(agg), zinc_rad_flat
        elif self.reward_aggregation == "mul":
            return cons_rew, flat_rewards, np.prod(cons_rew, axis = 1), zinc_rad_flat
        elif self.reward_aggregation == "add_mul":
            agg = np.sum(softplus(torch.Tensor(cons_rew)).numpy(), axis=1) + np.prod(cons_rew, axis=1)
            agg_rew_normal_fn = lambda x: scale_range(x, 1 + (len(self.cond_range) + 1) * np.log(1+np.exp(1)), (len(self.cond_range)+1) * np.log(2), 1, 0)
            return cons_rew, flat_rewards, agg_rew_normal_fn(agg), zinc_rad_flat
        
class RewardFineTune(Reward):
    def __init__(self, cond_range_dict, ft_cond_dict, cond_prop_var, reward_aggregation, molenv_dict_path, zinc_rad_scale, hps) -> None:
        super().__init__(cond_range_dict, cond_prop_var, reward_aggregation, molenv_dict_path, zinc_rad_scale, hps)
        self.hps = hps
        self.tasks = ['Caco2', 'LD50', 'Lipophilicity', 'Solubility', 'BindingRate', 'MicroClearance', 'HepatocyteClearance']
        if hasattr(hps, 'task') and hps.task in self.tasks:
            if hasattr(hps, 'task_model_path'):
                with open(hps.task_model_path, 'rb') as f:
                    self.task_model, self.Y_scaler = pickle.load(f)
        if ft_cond_dict is not None:
            tmp_cond_range_dict = cond_range_dict.copy()
            tmp_cond_range_dict.update(ft_cond_dict)
            self.cond_range = tmp_cond_range_dict

    def task_reward(self, task, task_model, mols):
        task_model_reward_funcs = {
            'Caco2': lambda x: self.permeability(task_model, x),
            'LD50': lambda x: self.toxicity(task_model, x),
            'Lipophilicity': lambda x: self.lipo(task_model, x),
            'Solubility': lambda x: self.sol(task_model, x),
            'BindingRate': lambda x: self.bind(task_model, x),
            'MicroClearance': lambda x: self.mclear(task_model, x),
            'HepatocyteClearance': lambda x: self.hclear(task_model, x),
        }

        if task == 'Mol_Wt':
            true_task_score = self.mol_wt(mols)
        elif task == 'logP':
            true_task_score = self.logP(mols)
        elif task == 'fsp3':
            true_task_score = self.fsp3(mols)
        elif task in task_model_reward_funcs and task_model is not None:
            true_task_score = task_model_reward_funcs[task](mols)
        elif task is None:
            return np.ones((len(mols), 1)), None
        else:
            # Dynamic lookup for task reward as well
            try:
                if hasattr(Descriptors, task):
                    true_task_score = np.array([getattr(Descriptors, task)(m) for m in mols])
                else:
                    true_task_score = np.zeros(len(mols))
            except:
                true_task_score = np.zeros(len(mols))

        task_range = self.hps.get('task_possible_range', [0, 1])
        rate = self.cond_var.get(task, 1.0)
        composite_reward = []
        
        pref_dir = self.hps.get('pref_dir', 1)
        for p_x in true_task_score:
            if p_x < task_range[0]:
                normalized_reward = (0.5 if pref_dir == 1 else 1.0) * np.exp(-(task_range[0] - p_x) / rate)
            elif p_x > task_range[1]:
                normalized_reward = (1.0 if pref_dir == 1 else 0.5) * np.exp(-(p_x - task_range[1]) / rate)
            else:
                if pref_dir == -1:
                    normalized_reward = 1 - ((p_x - task_range[0]) / (task_range[1] - task_range[0]))
                elif pref_dir == 1:
                    normalized_reward = (p_x - task_range[0]) / (task_range[1] - task_range[0])
                else:
                    normalized_reward = 1.0
            composite_reward.append(normalized_reward)
            
        flat_rewards_task = np.array(composite_reward).reshape(-1, 1)
        return flat_rewards_task, true_task_score
