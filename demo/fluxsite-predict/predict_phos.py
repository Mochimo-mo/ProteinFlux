import os
import sys
import subprocess
import argparse
import json
import logging
import torch
import numpy as np
import pandas as pd
import h5py
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

try:
    from Bio import PDB
    from Bio.PDB import PDBParser, NeighborSearch
    HAS_BIO = True
except ImportError:
    HAS_BIO = False

# Import the FluxSite package from the repository checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fluxsite.data.unified_data_processor import UnifiedDataProcessor, UnifiedPTMDataset
from fluxsite.utils.common_utils import custom_collate_fn, set_seed
from fluxsite.models.acetylation_predictor import DualBranchFusionPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('predict_phos')

# --- Helper Functions from train_with_rl.py ---
def _resolve_branch_hidden_dim(config: dict) -> int:
    branch_dim = config.get('branch_hidden_dim')
    if branch_dim is None:
        branch_dim = config.get('fusion_hidden_dim')
    if branch_dim is None:
        branch_dim = 256
    return max(1, int(branch_dim))

def _positive_int(config: dict, key: str, default: int) -> int:
    value = config.get(key, default)
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, value_int)

def _collect_divisors(value: int) -> list:
    import math
    divisors = set()
    upper = int(math.sqrt(value)) + 1
    for candidate in range(1, upper):
        if value % candidate == 0:
            divisors.add(candidate)
            divisors.add(value // candidate)
    return sorted(divisors)

def adjust_cross_attention_hyperparams(config: dict) -> None:
    branch_dim = _resolve_branch_hidden_dim(config)
    config['branch_hidden_dim'] = branch_dim
    # ... (simplified for prediction, mainly need dimensions) ...
    # Assuming config is already reasonable or trained model dictates it.

# --- Main Script ---

def check_and_generate_features(protein_ids, h5_path, fasta_dir, pdb_dir, output_dir, device_id=0, chain_id=None):
    """Use the precomputed features shipped with this demo."""
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"Feature file not found: {h5_path}")
    with h5py.File(h5_path, 'r') as features:
        if 'proteins' not in features:
            raise ValueError(f"Missing /proteins group in {h5_path}")
        missing = [protein_id for protein_id in protein_ids if protein_id not in features['proteins']]
    if missing:
        raise ValueError(f"Missing precomputed features for: {', '.join(missing)}")
    return h5_path

class PredictNormalizer:
    def __init__(self, stats):
        self.stats = stats
    def normalize(self, batch_dict):
        for key, stat in self.stats.items():
            tensor = None
            if key == 'sequence_features':
                if 'sequence' in batch_dict and isinstance(batch_dict['sequence'], dict):
                    tensor = batch_dict['sequence'].get('window_features')
                elif 'sequence_features' in batch_dict:
                     tensor = batch_dict.get('sequence_features')
            elif key == 'local_features':
                if 'sequence' in batch_dict and isinstance(batch_dict['sequence'], dict):
                    tensor = batch_dict['sequence'].get('local_features')
                elif 'local_features' in batch_dict:
                    tensor = batch_dict.get('local_features')
            elif key == 'global_features':
                if 'sequence' in batch_dict and isinstance(batch_dict['sequence'], dict):
                    tensor = batch_dict['sequence'].get('global_features')
                elif 'global_features' in batch_dict:
                    tensor = batch_dict.get('global_features')
            elif key == 'structure_features':
                tensor = batch_dict.get('structure')
                if tensor is None:
                    tensor = batch_dict.get('structure_features')
            elif key in batch_dict:
                tensor = batch_dict[key]
            
            if tensor is None: continue
            
            mean = torch.tensor(stat['mean'], device=tensor.device, dtype=tensor.dtype)
            std = torch.tensor(stat['std'], device=tensor.device, dtype=tensor.dtype)
            
            if tensor.dim() == 3:
                mean = mean.view(1, 1, -1)
                std = std.view(1, 1, -1)
            elif tensor.dim() == 2:
                mean = mean.view(1, -1)
                std = std.view(1, -1)
            tensor.sub_(mean).div_(std)
            
            if key == 'sequence_features':
                batch_dict['sequence']['window_features'] = tensor
            elif key == 'local_features':
                batch_dict['sequence']['local_features'] = tensor
            elif key == 'global_features':
                batch_dict['sequence']['global_features'] = tensor
            elif key == 'structure_features':
                batch_dict['structure'] = tensor

class StructuralAttentionPlotter:
    def __init__(self, pdb_path):
        self.pdb_path = pdb_path
        if not HAS_BIO:
            logger.warning("Biopython not installed, StructuralAttentionPlotter disabled")
            self.structure = None
            return
            
        parser = PDBParser(QUIET=True)
        try:
            self.structure = parser.get_structure("protein", pdb_path)
            self.model = self.structure[0]
        except Exception as e:
            logger.warning(f"Failed to load structure {pdb_path}: {e}")
            self.structure = None
            
    def calculate_distance_matrix(self, chain_id='A'):
        if not self.structure: return None, None
        
        try:
            chain = self.model[chain_id]
        except KeyError:
            logger.warning(f"Chain {chain_id} not found in {self.pdb_path}")
            # Fallback to first chain
            chain = next(iter(self.model))
            
        residues = [r for r in chain if PDB.is_aa(r)]
        res_ids = [r.get_id()[1] for r in residues]
        coords = []
        valid_indices = []
        
        for i, r in enumerate(residues):
            if 'CA' in r:
                coords.append(r['CA'].get_coord())
                valid_indices.append(i)
            else:
                # Missing CA, handle appropriately (skip or interpolate)
                pass
                
        if not coords: return None, None
        
        coords = np.array(coords)
        # Calculate pairwise distances
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
        
        return dist_matrix, res_ids

    def plot_distance_matrix(self, output_path, chain_id='A'):
        """Generate 3D distance map (contact map)"""
        dist_mat, _ = self.calculate_distance_matrix(chain_id=chain_id)
        if dist_mat is None or plt is None: return
        
        plt.figure(figsize=(10, 8), dpi=400)
        sns.heatmap(dist_mat, cmap="viridis", square=True)
        plt.title(f"C-alpha Distance Matrix: {os.path.basename(self.pdb_path)}")
        plt.xlabel("Residue Index")
        plt.ylabel("Residue Index")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

    def generate_pymol_session_script(self, attention_weights, residue_indices, output_dir, sample_id, chain_id='A'):
        """
        Generate a PyMOL script (.pml) that:
        1. Loads the PDB
        2. Maps attention weights to B-factors
        3. Visualizes the structure with gradient coloring
        4. Saves .pse and .png
        """
        if not self.structure: return
        
        script_path = os.path.join(output_dir, f"{sample_id}_struct_att.pml")
        pdb_name = os.path.basename(self.pdb_path)
        obj_name = f"{sample_id}_struct"
        
        # Create a weights map: residue_index -> weight
        # Attention weights correspond to the window sequence.
        # residue_indices are the PDB residue numbers corresponding to that window.
        
        # Normalize weights for visualization
        w_min, w_max = attention_weights.min(), attention_weights.max()
        norm_weights = (attention_weights - w_min) / (w_max - w_min + 1e-8)
        
        with open(script_path, 'w') as f:
            f.write(f"load {os.path.abspath(self.pdb_path)}, {obj_name}\n")
            f.write("hide all\n")
            f.write(f"show cartoon, {obj_name}\n")
            f.write(f"color white, {obj_name}\n")
            f.write(f"alter {obj_name}, b=0.0\n")
            
            # Map weights
            for res_idx, weight in zip(residue_indices, norm_weights):
                # Check if residue exists in structure to avoid errors? 
                # PyMOL simply ignores if selection is empty usually, but better be safe.
                # Assuming chain A for now as per init default or passed arg
                f.write(f"alter {obj_name} and chain {chain_id} and resi {res_idx}, b={weight:.4f}\n")
            
            f.write("sort\n")
            # Spectrum: Blue (low) -> Red (high) or Custom Gradient
            # User asked for "Gradient color bar identifying attention weight"
            f.write(f"spectrum b, blue_white_red, {obj_name} and chain {chain_id}, minimum=0, maximum=1\n")
            
            # Highlight missing residues? (Gray)
            # Actually we colored everything white initially, then spectrum colored valid ones.
            # Residues not in the window will remain white (or b=0 -> blue). 
            # We should probably color non-window residues gray.
            # Select all, color gray, then spectrum selection.
            f.write(f"color gray80, {obj_name}\n")
            # Create selection for window
            resi_str = "+".join(str(i) for i in residue_indices)
            # Selection string limit in pymol? 
            # If too long, break it up. For window=63 it's fine.
            f.write(f"select window_res, {obj_name} and chain {chain_id} and resi {resi_str}\n")
            f.write(f"spectrum b, blue_white_red, window_res, minimum=0, maximum=1\n")
            
            # Orient and Zoom
            f.write("orient window_res\n")
            f.write("zoom window_res, 10\n")
            
            # Legend? PyMOL CGO for colorbar is complex. 
            # Alternatively, use 'ramp_new'
            f.write(f"ramp_new colorbar, {obj_name}, [0, 0.5, 1], [blue, white, red]\n")
            
            # Save PSE
            pse_path = os.path.join(output_dir, f"{sample_id}_struct_att.pse")
            f.write(f"save {os.path.abspath(pse_path)}\n")
            
            # Save PNG
            png_path = os.path.join(output_dir, f"{sample_id}_struct_att.png")
            f.write(f"png {os.path.abspath(png_path)}, width=1200, height=1200, dpi=300, ray=1\n")
            
            f.write("quit\n")
            
        logger.info(f"Generated PyMOL script: {script_path}")
        
        # Execute PyMOL script to generate PSE and PNG
        try:
            # Check if pymol is available
            env = os.environ.copy()
            env["QT_QPA_PLATFORM"] = "offscreen"
            
            # Run pymol in headless mode
            logger.info("Executing PyMOL script to generate PSE/PNG...")
            subprocess.run(["pymol", "-c", "-q", script_path], env=env, check=True)
            logger.info(f"Successfully generated PSE: {pse_path}")
        except Exception as e:
            logger.warning(f"Failed to execute PyMOL automatically: {e}")
            logger.warning(f"You can manually run: pymol {script_path}")

    def plot_2D_structure_attention(self, output_path, seq_attn_weights, struct_attn_weights, residue_indices, chain_id='A'):
        """
        Plot 2D structure attention: Distance Matrix overlaid with Sequence Attention Weights and Structure Attention Weights.
        """
        if not self.structure or plt is None: return
        
        try:
            chain = self.model[chain_id]
        except KeyError:
            # Fallback to first chain
            chain = next(iter(self.model))
            
        coords = []
        final_seq_weights = []
        final_struct_weights = []
        final_res_ids = []
        
        # Filter residues that exist in PDB structure
        # Assuming seq_attn_weights and struct_attn_weights are aligned with residue_indices
        for i, res_id in enumerate(residue_indices):
            if res_id in chain and 'CA' in chain[res_id]:
                coords.append(chain[res_id]['CA'].get_coord())
                final_seq_weights.append(seq_attn_weights[i])
                if struct_attn_weights is not None and i < len(struct_attn_weights):
                     final_struct_weights.append(struct_attn_weights[i])
                final_res_ids.append(res_id)
        
        if not coords:
            return
            
        coords = np.array(coords)
        final_seq_weights = np.array(final_seq_weights)
        if final_struct_weights:
            final_struct_weights = np.array(final_struct_weights)
        else:
            final_struct_weights = None
        
        # Calculate Distance Matrix
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=-1))

        axis_label_size = 24
        axis_tick_size = 14
        
        # Plotting
        # Reduced height since we removed one bar plot
        fig = plt.figure(figsize=(10, 10), dpi=300)
        fig.patch.set_facecolor('white')
        
        # GridSpec: 
        # Row 0: Attention
        # Row 1: Distance Matrix
        gs = fig.add_gridspec(2, 2,  width_ratios=[1, 0.05], height_ratios=[0.15, 1],
                              wspace=0.05, hspace=0.05)
        
        ax_seq_att = fig.add_subplot(gs[0, 0])
        ax_dist = fig.add_subplot(gs[1, 0])
        ax_cbar = fig.add_subplot(gs[1, 1])
        
        x = np.arange(len(final_seq_weights))
        
        # 1. Sequence Attention Bar Plot (Top)
        norm_seq = plt.Normalize(final_seq_weights.min(), final_seq_weights.max())
        cmap_seq = plt.get_cmap('viridis')
        colors_seq = cmap_seq(norm_seq(final_seq_weights))
        
        ax_seq_att.bar(x, final_seq_weights, color=colors_seq, width=1.0)
        ax_seq_att.set_xlim(-0.5, len(final_seq_weights)-0.5)
        ax_seq_att.set_xticks([])
        ax_seq_att.set_ylabel("Attention", fontsize=axis_label_size)
        ax_seq_att.tick_params(axis='y', labelsize=axis_tick_size)
        ax_seq_att.spines['top'].set_visible(False)
        ax_seq_att.spines['right'].set_visible(False)
        ax_seq_att.spines['bottom'].set_visible(False)
        
        # 3. Distance Matrix Heatmap (Bottom)
        # Using imshow for better alignment control
        im = ax_dist.imshow(dist_mat, cmap='RdYlBu_r', aspect='auto', origin='upper')
        
        ax_dist.set_xlabel("Residue Index (Window)", fontsize=axis_label_size)
        ax_dist.set_ylabel("Residue Index (Window)", fontsize=axis_label_size)
        
        # Ticks
        interval = max(1, len(final_res_ids) // 10)
        ax_dist.set_xticks(x[::interval])
        ax_dist.set_xticklabels(final_res_ids[::interval], rotation=90, fontsize=16)
        ax_dist.set_yticks(x[::interval])
        ax_dist.set_yticklabels(final_res_ids[::interval], fontsize=16)

        # Colorbar for distance
        cbar = plt.colorbar(im, cax=ax_cbar, label='Distance (Å)')
        cbar.set_label('Distance (Å)', fontsize=axis_label_size)
        cbar.ax.tick_params(labelsize=axis_tick_size)

        # Keep figure size unchanged, but reserve extra top margin so large labels
        # (e.g., "Attention") are fully visible after font-size increases.
        fig.subplots_adjust(top=0.90)
        
        # plt.suptitle(f"2D Structure & Attention: {os.path.basename(self.pdb_path)}", y=0.95)
        
        plt.savefig(output_path, transparent=False, facecolor='white')
        plt.close()
        logger.info(f"Generated 2D structure attention plot: {output_path}")

def extract_ptm_hotspots(pdb_path, predictions, chain_id='A', dist_cutoff=12.0):
    """
    Extract PTM 3D Hotspots.
    predictions: List of dicts with keys 'position', 'prediction', 'confidence_score'
    """
    if not HAS_BIO or not os.path.exists(pdb_path): return []
    
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("prot", pdb_path)
        model = structure[0]
        chain = model[chain_id]
    except Exception: return []
    
    # Get all atoms for neighbor search
    atoms = [a for a in chain.get_atoms() if a.name == 'CA']
    ns = NeighborSearch(atoms)
    
    # Identify PTM sites (predicted positive)
    ptm_sites = {p['position']: p for p in predictions if p['prediction'] == 1}
    if not ptm_sites: return []
    
    hotspots = []
    
    # For each PTM residue, find neighbors
    for res_id, info in ptm_sites.items():
        try:
            # PDB res_id might assume 1-based indexing matching position
            center_residue = chain[res_id]
            center_atom = center_residue['CA']
        except KeyError: continue
        
        neighbors = ns.search(center_atom.get_coord(), dist_cutoff, level='R')
        # Filter neighbors: >= 3 residues
        if len(neighbors) < 3: continue
        
        # Check if patch contains >= 2 PTM sites
        patch_res_ids = [r.get_id()[1] for r in neighbors]
        ptm_count = sum(1 for rid in patch_res_ids if rid in ptm_sites)
        
        if ptm_count >= 2:
            # Check linear sequence span >= 30
            span = max(patch_res_ids) - min(patch_res_ids) + 1
            if span >= 30:
                hotspots.append({
                    'pdb_id': os.path.basename(pdb_path),
                    'patch_id': f"P_{res_id}",
                    'center_res': res_id,
                    'neighbor_list': sorted(patch_res_ids),
                    'seq_span': span,
                    'ptm_count': ptm_count,
                    'confidence': info['confidence_score']
                })
                
    return hotspots

def apply_ref_style():
    """
    Apply reference style from compare_phos_y_models_curves.py
    Returns the color palette.
    """
    if plt is None: return []

    # Colors from compare_phos_y_models_curves.py
    # Pink, Green, Orange, Blue, Light Green
    colors = [
        (227/255, 141/255, 179/255), 
        (78/255, 172/255, 151/255),  
        (251/255, 184/255, 142/255), 
        (56/255, 134/255, 194/255),  
        (134/255, 198/255, 184/255), 
    ]
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
        'font.weight': 'regular',
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 400,
        'savefig.dpi': 400,
        'savefig.transparent': True,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'lines.linewidth': 1.5,
    })
    return colors

def plot_modif_attention(sample_id, seq, att_weights, modif_sites=None):
    """
    Plot attention heatmap as a bar plot for publication quality (Nature-style).
    
    Args:
        sample_id: Identifier for the sample (e.g. 's636')
        seq: The amino acid sequence window string
        att_weights: Attention weights array (length of seq)
        modif_sites: The center position index (relative to window) to highlight
    """
    if plt is None:
        logger.warning("matplotlib not installed, skipping plot")
        return

    axis_label_size = 18
    axis_tick_size = 15
        
    # Apply style
    apply_ref_style()
    
    # Ensure directory exists
    save_dir = "./attention_figs"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create figure - 4:3 or 16:10 aspect ratio.
    # Using 8x5 inches (16:10 close)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=400)
    
    x_indices = np.arange(len(seq))
    
    # Match top sequence-bar palette in 2D structure attention plots.
    cmap = plt.get_cmap('viridis')
        
    norm = plt.Normalize(att_weights.min(), att_weights.max())
    bar_colors = cmap(norm(att_weights))
    
    # Plot bars
    bars = ax.bar(x_indices, att_weights, color=bar_colors, width=0.8, edgecolor='none')
    
    # Highlight modification site
    if modif_sites is not None:
        target_color = bar_colors[modif_sites]
        
        # Calculate positions
        x_pos = modif_sites
        y_pos = att_weights[modif_sites]
        y_max = att_weights.max()
        text_y = y_max * 1.25
        
        # Adaptive angle simulation (shift x_text)
        # Shift slightly based on position to demonstrate "angle"
        x_shift = 0
        if modif_sites < len(seq) * 0.2: x_shift = 0.5
        elif modif_sites > len(seq) * 0.8: x_shift = -0.5
        
        # Annotation with Indicator Line
        # Line: 0.75 pt, solid, matching color
        # Start offset (shrinkA): 2px (~0.5 pt)
        # End gap (shrinkB): 1mm (~2.8 pt)
        ax.annotate('Modif', 
                    xy=(x_pos, y_pos), 
                    xytext=(x_pos + x_shift, text_y),
                    fontsize=16,
                    ha='center',
                    va='bottom',
                    arrowprops=dict(
                        arrowstyle='-',
                        color=target_color,
                        linewidth=0.75,
                        shrinkA=0.5, # approx 2px at 400dpi if px=pixel? Using pts for safety
                        shrinkB=2.8, # approx 1mm
                        patchB=None
                    ))
        
        # Add 3 px dot at data end (xy)
        # markersize is in points. 3 points.
        ax.plot(x_pos, y_pos + (y_max * 0.01), marker='o', markersize=3, color=target_color, markeredgecolor='none', zorder=10)

    # Customize axes
    if sns:
        sns.despine(ax=ax, top=True, right=True)
    else:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # X-axis
    ax.set_xticks(x_indices)
    ax.set_xticklabels(list(seq), fontsize=10, fontfamily='monospace')
    ax.tick_params(axis='x', rotation=0)
    ax.set_xlim(-0.5, len(seq) - 0.5)
    
    # Y-axis
    ax.set_ylabel("Attention Weight", fontsize=axis_label_size)
    ax.set_ylim(0, att_weights.max() * 1.4)
    
    # Title
    #  plt.title(f"{sample_id} Phosphorylation Attention", fontsize=16, fontweight='bold', pad=15)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, aspect=20)
    cbar.set_label('Weight', rotation=270, labelpad=26, fontsize=axis_label_size)
    cbar.ax.tick_params(labelsize=axis_tick_size)
    
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{sample_id}_modif_att.png")
    plt.savefig(out_path, dpi=400, bbox_inches='tight', transparent=True)
    
    # Vector PDF
    pdf_path = os.path.join(save_dir, f"{sample_id}_modif_att.pdf")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', transparent=True)
    
    plt.close()
    logger.info(f"Saved attention plots to {out_path} and {pdf_path}")

def get_window_sequence(fasta_dir, uniprot_id, position, window_size):
    try:
        from Bio import SeqIO
        fasta_path = os.path.join(fasta_dir, f"{uniprot_id}.fasta")
        if not os.path.exists(fasta_path):
             for ext in ['.fa', '.txt']:
                if os.path.exists(os.path.join(fasta_dir, f"{uniprot_id}{ext}")):
                    fasta_path = os.path.join(fasta_dir, f"{uniprot_id}{ext}")
                    break
        
        if os.path.exists(fasta_path):
            record = SeqIO.read(fasta_path, "fasta")
            full_seq = str(record.seq)
            
            # Position is 1-based
            # Window logic from UnifiedDataProcessor usually:
            # center at pos-1.
            # radius = (window_size - 1) // 2
            # start = pos - 1 - radius
            # end = pos - 1 + radius + 1
            
            radius = (window_size - 1) // 2
            idx = position - 1
            start = idx - radius
            end = idx + radius + 1
            
            # Pad if needed
            seq_len = len(full_seq)
            pad_left = 0
            pad_right = 0
            
            if start < 0:
                pad_left = -start
                start = 0
            if end > seq_len:
                pad_right = end - seq_len
                end = seq_len
                
            sub_seq = full_seq[start:end]
            final_seq = ('-' * pad_left) + sub_seq + ('-' * pad_right)
            
            if len(final_seq) != window_size:
                logger.warning(f"Window size mismatch: expected {window_size}, got {len(final_seq)}")
                
            return final_seq
    except Exception as e:
        logger.warning(f"Failed to fetch sequence for {uniprot_id}: {e}")
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--pos_file", required=True)
    parser.add_argument("--feature_h5", required=True)
    parser.add_argument("--pdb_dir", required=True)
    parser.add_argument("--fasta_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--chain_id", type=str, default='A', help="Chain ID to use for PDB structure (default: A)")
    parser.add_argument(
        "--micro_env_mode",
        choices=["zero", "proxy"],
        default="zero",
        help="Micro-environment feature mode. Use 'zero' to match the historical training pipeline.",
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.out_dir): os.makedirs(args.out_dir)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info(f"Loading config from {args.config_path}")
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    adjust_cross_attention_hyperparams(config)
    
    logger.info("Checking features...")
    df_pos = pd.read_csv(args.pos_file)
    
    # Filter for S/T residues only as per model focus
    logger.info("Filtering input for target residues (S, T)...")
    original_len = len(df_pos)
    df_pos = df_pos[df_pos['residue'].isin(['S', 'T'])]
    logger.info(f"Filtered {original_len} -> {len(df_pos)} samples (kept only S/T)")
    
    unique_proteins = df_pos['uniprot_id'].unique().tolist()
    final_h5_path = check_and_generate_features(unique_proteins, args.feature_h5, args.fasta_dir, args.pdb_dir, args.out_dir, args.gpu_id, chain_id=args.chain_id)
    
    temp_csv_path = None
    if 'ptm_type' not in df_pos.columns:
        logger.info("Adding missing 'ptm_type' column to input data")
        df_pos['ptm_type'] = 'phosphorylation'
        temp_csv_path = os.path.join(args.out_dir, "temp_input_with_ptm.csv")
        df_pos.to_csv(temp_csv_path, index=False)
        data_input_path = temp_csv_path
    else:
        data_input_path = args.pos_file

    try:
        processor = UnifiedDataProcessor(
            data_path=data_input_path, 
            esm_features_path=final_h5_path,
            pdb_dir=args.pdb_dir,
            fasta_dir=args.fasta_dir,
            window_size=config.get('window_size', 63),
            target_ptm_type='phosphorylation'
        )
        processed_data = processor.prepare_dataset()
        if args.micro_env_mode == "zero":
            logger.info("Using zero micro-environment features to match the historical training pipeline.")
            for item in processed_data:
                item['micro_env_features'] = np.zeros(6, dtype=np.float32)
        else:
            logger.info("Using proxy micro-environment features from the current data pipeline.")
        test_dataset = UnifiedPTMDataset(processed_data, fixed_window_size=config.get('window_size', 63), target_ptm_type='phosphorylation')
    finally:
        if temp_csv_path and os.path.exists(temp_csv_path): os.remove(temp_csv_path)
    
    test_loader = DataLoader(test_dataset, batch_size=config.get('batch_size', 32), shuffle=False, num_workers=4, collate_fn=custom_collate_fn, pin_memory=True)
    
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        # Remap keys if necessary
        new_state_dict[name] = v

    logger.info("Loading model...")
    model = DualBranchFusionPredictor(config=config)
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()
    
    normalizer = None
    if 'normalization_stats' in checkpoint and checkpoint['normalization_stats']:
        logger.info("Loading normalization stats")
        normalizer = PredictNormalizer(checkpoint['normalization_stats'])
    
    logger.info("Starting prediction...")
    results = []
    threshold = 0.50
    total_samples = 0
    positive_samples = 0
    start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            if normalizer: normalizer.normalize(batch)
            
            def to_device(obj):
                if isinstance(obj, torch.Tensor): return obj.to(device)
                elif isinstance(obj, dict): return {k: to_device(v) for k, v in obj.items()}
                elif isinstance(obj, list): return [to_device(v) for v in obj]
                return obj
            batch = to_device(batch)
            
            # Request importance weights for potential visualization.
            model_output = model(batch, return_importance=True)
            if isinstance(model_output, tuple):
                logits, importance = model_output
                attn_weights = importance.get('window_importance') if isinstance(importance, dict) else None
            elif isinstance(model_output, dict):
                logits = model_output['logits']
                attn_weights = None
            else:
                logits = model_output
                attn_weights = None
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs >= threshold).astype(int)
            
            batch_ids = batch['uniprot_ids']
            batch_pos = batch['positions']
            batch_res = batch['residues']
            
            for i in range(len(probs)):
                # Visualization Logic
                current_pid = batch_ids[i]
                current_pos = batch_pos[i].item() if isinstance(batch_pos[i], torch.Tensor) else batch_pos[i]
                current_res = batch_res[i]
                current_pred = int(preds[i])
                
                vis_target_id = None
                if current_pid == 'Q08460':
                    if current_res == 'S' and current_pos == 700:
                        vis_target_id = 's700'
                    elif current_res == 'S' and current_pos == 927:
                        vis_target_id = 'S927'
                    elif current_res == 'S' and current_pos == 636:
                        vis_target_id = 's636'
                
                if vis_target_id and current_pred == 1:
                    logger.info(f"Found target sample {vis_target_id} (Pred=1), generating attention heatmap...")
                    win_size = config.get('window_size', 63)
                    win_seq = get_window_sequence(args.fasta_dir, current_pid, current_pos, win_size)
                    
                    if win_seq:
                        # Attention weights: [SeqLen]
                        weights = attn_weights[i].cpu().numpy()
                        # Center index
                        center_idx = win_size // 2
                        plot_modif_attention(vis_target_id, win_seq, weights, modif_sites=center_idx)
                        
                        valid_struct_weights = None
                        
                        # --- Structural Viz ---
                        if current_pid == 'Q08460':
                            pdb_path = os.path.join(args.pdb_dir, f"{current_pid}.pdb")
                            if os.path.exists(pdb_path):
                                struct_plotter = StructuralAttentionPlotter(pdb_path)
                                radius = win_size // 2
                                residue_indices = []
                                valid_weights = []
                                for w_idx, char in enumerate(win_seq):
                                    if char != '-':
                                        res_num = current_pos - radius + w_idx
                                        if res_num > 0:
                                             residue_indices.append(res_num)
                                             valid_weights.append(weights[w_idx])
                                if residue_indices:
                                    # Generate distance map
                                    dist_map_path = os.path.join(args.out_dir, f"{vis_target_id}_dist_map.png")
                                    struct_plotter.plot_distance_matrix(dist_map_path, chain_id=args.chain_id)
                                    
                                    # Generate 2D Structure Attention Map
                                    struct_att_2d_path = os.path.join(args.out_dir, f"{vis_target_id}_2D_struct_att.pdf")
                                    struct_plotter.plot_2D_structure_attention(
                                        struct_att_2d_path,
                                        np.array(valid_weights),
                                        np.array(valid_struct_weights) if valid_struct_weights is not None else None,
                                        residue_indices,
                                        chain_id=args.chain_id
                                    )
                                    
                                    # Generate PyMOL script
                                    struct_plotter.generate_pymol_session_script(
                                        np.array(valid_weights), 
                                        residue_indices, 
                                        args.out_dir, 
                                        vis_target_id,
                                        chain_id=args.chain_id
                                    )
                    else:
                        logger.warning(f"Could not retrieve sequence for {vis_target_id}")

                results.append({
                    'protein_id': batch_ids[i],
                    'position': batch_pos[i].item() if isinstance(batch_pos[i], torch.Tensor) else batch_pos[i],
                    'residue': batch_res[i],
                    'probability': float(probs[i]),
                    'prediction': int(preds[i]),
                    'confidence_score': float(probs[i] if probs[i] > 0.5 else 1 - probs[i])
                })
                if preds[i] == 1: positive_samples += 1
            total_samples += len(probs)
            
    end_time = time.time()
    df_res = pd.DataFrame(results)
    out_csv = os.path.join(args.out_dir, "prediction_results.csv")
    df_res.to_csv(out_csv, index=False)
    logger.info(f"Predictions saved to {out_csv}")
    
    # Hotspot extraction for Q08460
    q08460_results = [r for r in results if r['protein_id'] == 'Q08460']
    if q08460_results:
        pdb_path = os.path.join(args.pdb_dir, "Q08460.pdb")
        hotspots = extract_ptm_hotspots(pdb_path, q08460_results, chain_id=args.chain_id)
        if hotspots:
             pd.DataFrame(hotspots).to_csv(os.path.join(args.out_dir, "Q08460_hotspots.csv"), index=False)
             logger.info(f"Saved hotspots to {os.path.join(args.out_dir, 'Q08460_hotspots.csv')}")

    summary = {
        'total_samples': total_samples,
        'positive_samples': positive_samples,
        'threshold': threshold,
        'duration_seconds': end_time - start_time,
        'timestamp': datetime.now().isoformat()
    }
    with open(os.path.join(args.out_dir, "prediction_summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)
    logger.info("Done.")

if __name__ == "__main__":
    main()
