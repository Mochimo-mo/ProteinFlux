import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import numpy as np
import os
from argparse import Namespace
import logging
import argparse # Import argparse for add_safe_globals
from pathlib import Path

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("protein_encoders")

# Local model file paths.
ESM_MODEL_PATHS = {
    "esm2_t33_650M_UR50D": "/path/to/esm_weights/esm-2/esm2_t33_650M/esm2_t33_650M_UR50D.pt",
    "esm2_t33_650M_UR50D_alphabet": "/path/to/esm_weights/esm-2/esm2_t33_650M/esm2_t33_650M_UR50D_alphabet.pt",  # Alphabet file path
    "esm2_t33_650M_UR50D_contact": "/path/to/esm_weights/esm-2/esm2_t33_650M/esm2_t33_650M_UR50D-contact-regression.pt",  # Contact regression file path
    "esm2_t36_3B_UR50D": "/path/to/esm_weights/esm-2/models/esm2_t36_3B_UR50D.pt",
    "esm_if1_gvp4_t16_142M_UR50": "/path/to/esm_weights/esm-2/esm_if1_gvp4_t16_142M_UR50.pt",
}

class ProteinSequenceEncoder(nn.Module):
    """
    Protein sequence encoder: extract protein sequence features using ESM-2.

    Inspired by ProteinGPT, with more robust model loading and feature extraction.
    """
    def __init__(self, model_name="esm2_t33_650M_UR50D", freeze=True):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"ESMModel Load Device: {self.device}")
        
        if model_name in ESM_MODEL_PATHS:
            model_path = ESM_MODEL_PATHS[model_name]
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model: {model_path}")
            
            # Load model.
            self.model, self.alphabet = self._load_model_from_path(model_path, model_name)
            self.model = self.model.to(self.device)
            
            # Ensure the model exposes embed_dim.
            if hasattr(self.model, "embed_dim"):
                self.embed_dim = self.model.embed_dim
            elif hasattr(self.model, "args") and hasattr(self.model.args, "embed_dim"):
                self.embed_dim = self.model.args.embed_dim
            elif hasattr(self.model, "dim"):
                self.embed_dim = self.model.dim
            else:
                # Infer embedding dimension from model parameters.
                for name, param in self.model.named_parameters():
                    if 'embed' in name and 'weight' in name:
                        self.embed_dim = param.shape[1]
                        break
                if not hasattr(self, 'embed_dim'):
                    # Use a default inferred from model size.
                    if '3B' in model_name:
                        self.embed_dim = 2560
                    elif '650M' in model_name:
                        self.embed_dim = 1280
                    else:
                        self.embed_dim = 768
                    logger.warning(f" embed_dim,Use: {self.embed_dim}")
            
            # Choose representation layer index.
            if hasattr(self.model, "num_layers"):
                self.repr_layer_index = self.model.num_layers
            else:
                self.repr_layer_index = -1
                
            # Freeze parameters.
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
                logger.info(" ModelParameters ")
        else:
            raise ValueError(f" Model {model_name} Config Path, ESM_MODEL_PATHS ")
    
    def _load_model_from_path(self, model_path, model_name):
        """Helper to load a model from a local path with robust fallbacks."""
        logger.info(f"Load Model: {model_name}: {model_path}")

        try:
            # Strategy 1: try to load via ESM API.
            try:
                if hasattr(esm.pretrained, 'load_model_and_alphabet_local'):
                    model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
                    logger.info(f" UseESM APILoad Model: {model_name}")
                    return model, alphabet
                else:
                    logger.warning(" esm 'load_model_and_alphabet_local',.")
            except Exception as e:
                logger.warning(f"ESM APILoadFailed: {str(e)},.")
            
            # Strategy 2: load raw model data via torch.load.
            logger.info("Usetorch.loadLoadModelData...")
            # Add safe globals to support argparse.Namespace.
            import argparse
            torch.serialization.add_safe_globals([argparse.Namespace])
            try:
                model_data = torch.load(model_path, map_location="cpu", weights_only=True)
            except Exception as e:
                logger.warning(f"weights_only=True LoadFailed: {e}, weights_only=False")
                model_data = torch.load(model_path, map_location="cpu", weights_only=False)
            logger.info("ModelDataLoad ")
            
            # Get model name and directory.
            model_location = Path(model_path)
            model_basename = model_location.stem
            logger.info(f"Model: {model_basename}")
            
            # Check whether contact-regression weights are needed.
            regression_data = None
            if not ("esm1v" in model_basename or "esm_if" in model_basename or "270K" in model_basename or "500K" in model_basename):
                try:
                    regression_location = str(model_location.with_suffix("")) + "-contact-regression.pt"
                    if os.path.exists(regression_location):
                        try:
                            regression_data = torch.load(regression_location, map_location="cpu", weights_only=True)
                        except Exception:
                            regression_data = torch.load(regression_location, map_location="cpu", weights_only=False)
                        logger.info(" DataLoad ")
                except Exception as re:
                    logger.warning(f" Load Data: {re}, ")
            
            # Strategy 3: load via ESM core loader.
            try:
                if hasattr(esm.pretrained, 'load_model_and_alphabet_core'):
                    model, alphabet = esm.pretrained.load_model_and_alphabet_core(
                        model_basename, model_data, regression_data
                    )
                    logger.info("UseESM LoadModel")
                    return model, alphabet
            except Exception as core_e:
                logger.warning(f"Use LoadFailed: {core_e},.")
            
            # Strategy 4: extract model directly from model_data.
            if isinstance(model_data, dict):
                if "model" in model_data:
                    model_obj = model_data["model"]
                    alphabet = model_data.get("alphabet", esm.data.Alphabet.from_architecture("ESM-1b"))
                    logger.info(" Model Load Model ")
                    return model_obj, alphabet
                elif "args" in model_data and "state_dict" in model_data:
                    args = model_data["args"]
                    state_dict = model_data["state_dict"]
                    if hasattr(esm, 'model') and hasattr(esm.model, 'ProteinBertModel'):
                        model = esm.model.ProteinBertModel(args)
                        model.load_state_dict(state_dict)
                        alphabet = esm.data.Alphabet.from_architecture(args.arch)
                        logger.info(" state_dict argsCreate Model ")
                        return model, alphabet
                    else:
                        raise ValueError(" esm 'model.ProteinBertModel'")
            elif isinstance(model_data, nn.Module):
                logger.info(" Load Model ")
                return model_data, esm.data.Alphabet.from_architecture("ESM-1b")
            
            # Strategy 5: fallback to a pretrained model.
            logger.warning(f" LoadModel, Use TrainingModel.")
            if "esm2_t33_650M" in model_basename:
                model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                logger.info("Use Training esm2_t33_650M_UR50DModel")
                return model, alphabet
            elif "esm2_t36_3B" in model_basename:
                model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
                logger.info("Use Training esm2_t36_3B_UR50DModel")
                return model, alphabet
            else:
                raise ValueError(f" TrainingModel: {model_basename}")
                
        except Exception as e:
            logger.error(f" Load Failed: {str(e)}")
            raise ValueError(f" Load Model {model_name}: {str(e)}")

    def tokenize(self, sequences):
        """Convert protein sequences into token IDs for the model input."""
        if not hasattr(self, 'alphabet'):
            raise ValueError("Model alphabet, ")
            
        batch_tokens = []
        for seq in sequences:
            tokens = self.alphabet.encode(seq)
            batch_tokens.append(tokens)
        
        # Pad to the same length.
        max_len = max(len(t) for t in batch_tokens)
        batch_tokens_padded = []
        for tokens in batch_tokens:
            padding = torch.full((max_len - len(tokens),), self.alphabet.padding_idx)
            batch_tokens_padded.append(torch.cat([tokens, padding]))
        
        return torch.stack(batch_tokens_padded)

    def extract_features(self, tokens, return_contacts=False):
        """Extract features from sequences, optionally returning contact maps."""
        if hasattr(self, 'device'):
            tokens = tokens.to(self.device)
            
        with torch.no_grad():
            # Use the standard ESM-2 API.
            if hasattr(self.model, 'extract_features'):
                out = self.model.extract_features(
                    tokens,
                    repr_layers=[self.repr_layer_index],
                    return_contacts=return_contacts
                )
                sequence_repr = out['representations'][self.repr_layer_index]
                
                # Split CLS token as global representation; the rest are per-residue representations.
                global_repr = sequence_repr[:, 0]
                per_residue_repr = sequence_repr[:, 1:]
                
                if return_contacts:
                    return global_repr, per_residue_repr, out['contacts']
                return global_repr, per_residue_repr
            
            # If extract_features is not available, use a standard forward pass.
            return self._forward_standard(tokens)
                
    def forward(self, tokens):
        """
        Forward pass. Prefer extract_features when available; otherwise fall back to a standard forward.
        
        Args:
            tokens: ESM-tokenized sequences.
        
        Returns:
            tuple: (global_repr, per_residue_repr)
        """
        if hasattr(self, 'device'):
            tokens = tokens.to(self.device)
            
        # Prefer extract_features for more consistent handling across ESM model versions.
        if hasattr(self.model, 'extract_features'):
            return self.extract_features(tokens, return_contacts=False)
            
        return self._forward_standard(tokens)
        
    def _forward_standard(self, tokens):
        """Standard forward path handling multiple possible output formats."""
        with torch.no_grad():
            try:
                # Try the standard ESM-2 interface.
                try:
                    logger.debug(f" Use ESM-2 shape: {tokens.shape}")
                    repr_layers = [self.repr_layer_index]
                    output = self.model(tokens, repr_layers=repr_layers)
                    
                    if isinstance(output, dict) and "representations" in output:
                        layer_key = list(output["representations"].keys())[-1]
                        sequence_repr = output["representations"][layer_key][:, 0, :]
                        per_residue_repr = output["representations"][layer_key][:, 1:, :]
                        logger.debug(f", shape: {sequence_repr.shape}, Residue shape: {per_residue_repr.shape}")
                        return sequence_repr, per_residue_repr
                except Exception as e:
                    logger.warning(f" ESM-2 Failed: {e}")
                
                # Try a simplified interface.
                try:
                    output = self.model(tokens)
                    
                    # Handle multiple possible output formats.
                    if isinstance(output, dict) and "representations" in output:
                        layer_key = list(output["representations"].keys())[-1]
                        sequence_repr = output["representations"][layer_key][:, 0, :]
                        per_residue_repr = output["representations"][layer_key][:, 1:, :]
                    elif isinstance(output, dict) and "last_hidden_state" in output:
                        sequence_repr = output["last_hidden_state"][:, 0, :]
                        per_residue_repr = output["last_hidden_state"][:, 1:, :]
                    elif isinstance(output, tuple) and len(output) > 0:
                        sequence_repr = output[0][:, 0, :]
                        per_residue_repr = output[0][:, 1:, :]
                    else:
                        sequence_repr = output[:, 0, :]
                        per_residue_repr = output[:, 1:, :]
                    
                    logger.debug(f", shape: {sequence_repr.shape}, Residue shape: {per_residue_repr.shape}")
                    return sequence_repr, per_residue_repr
                except Exception as e:
                    logger.warning(f" Failed: {e}")
                
                # Try model-specific feature methods.
                if hasattr(self.model, "forward_features"):
                    features = self.model.forward_features(tokens)
                    if isinstance(features, torch.Tensor):
                        sequence_repr = features[:, 0, :]
                        per_residue_repr = features[:, 1:, :]
                        return sequence_repr, per_residue_repr
                
                # Last resort: direct call.
                logger.warning(" Model,Use.")
                output = self.model(tokens)
                if isinstance(output, torch.Tensor):
                    sequence_repr = output[:, 0, :]
                    per_residue_repr = output[:, 1:, :]
                    return sequence_repr, per_residue_repr
                
                raise RuntimeError(" Model ")
                
            except Exception as e:
                raise RuntimeError(f" Model Error: {e}")

class GVP(nn.Module):
    """
    Geometric Vector Perceptron (GVP).

    Processes scalar and vector features in protein structures while preserving rotational equivariance.
    Based on the ProteinGPT-style implementation.
    """
    def __init__(self, scalar_dim, vector_dim, scalar_out_dim=None, vector_out_dim=None, activation=F.relu):
        super(GVP, self).__init__()
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.scalar_out_dim = scalar_out_dim or scalar_dim
        self.vector_out_dim = vector_out_dim or vector_dim
        self.activation = activation
        
        # Scalar-to-scalar/vector projections.
        self.w_s2s = nn.Linear(scalar_dim, self.scalar_out_dim)
        self.w_s2v = nn.Linear(scalar_dim, 3*self.vector_out_dim) if vector_dim > 0 else None
        
        # Vector-to-scalar/vector projections.
        self.w_v2s = nn.Linear(vector_dim*3, self.scalar_out_dim) if vector_dim > 0 else None
        self.w_v2v = nn.Linear(vector_dim*3, 3*self.vector_out_dim) if vector_dim > 0 else None
        
    def forward(self, s, v=None):
        """
        Forward pass.
        
        Args:
            s: Scalar features [batch, ..., scalar_dim]
            v: Vector features [batch, ..., vector_dim, 3]
            
        Returns:
            s_out: Output scalar features [batch, ..., scalar_out_dim]
            v_out: Output vector features [batch, ..., vector_out_dim, 3]
        """
        s_out = self.w_s2s(s)
        
        if self.vector_dim > 0 and v is not None:
            v = v.reshape(v.shape[:-2] + (3*self.vector_dim,))
            
            # Vector features to scalar features.
            v2s = self.w_v2s(v)
            s_out = s_out + v2s
            
            # Scalar features to vector features.
            s2v = self.w_s2v(s).reshape(s.shape[:-1] + (self.vector_out_dim, 3))
            
            # Vector features to vector features.
            v2v = self.w_v2v(v).reshape(v.shape[:-1] + (self.vector_out_dim, 3))
            
            # Combine vector features.
            v_out = s2v + v2v
            
            # Apply activation to the scalar part.
            s_out = self.activation(s_out)
            
            # Modulate the vector part by its norm to preserve equivariance.
            v_norm = torch.norm(v_out, dim=-1, keepdim=True)
            v_normalized = self.activation(v_norm) * (v_out / (v_norm + 1e-8))
            
            return s_out, v_normalized
        else:
            # Scalar-only features.
            s_out = self.activation(s_out)
            return s_out, None

class GVPConvLayer(nn.Module):
    """
    Geometric vector convolution layer for modeling spatial relationships in protein structures.
    """
    def __init__(self, scalar_dim, vector_dim, edge_dim=None, n_message=3):
        super(GVPConvLayer, self).__init__()
        
        # Edge feature dimension.
        edge_dim = edge_dim or scalar_dim
        
        # Message passing modules.
        self.message_funcs = nn.ModuleList([
            GVP(scalar_dim + edge_dim, vector_dim, scalar_dim, vector_dim)
            for _ in range(n_message)
        ])
        
    def forward(self, s, v, edge_index, edge_attr=None):
        """
        Args:
            s: Node scalar features [n_nodes, scalar_dim]
            v: Node vector features [n_nodes, vector_dim, 3]
            edge_index: Edge indices [2, n_edges]
            edge_attr: Edge attributes [n_edges, edge_dim]
        """
        src, dst = edge_index
        
        for message_func in self.message_funcs:
            # Collect source node features.
            s_src = s[src]
            v_src = v[src] if v is not None else None
            
            # Merge edge features.
            if edge_attr is not None:
                s_src = torch.cat([s_src, edge_attr], dim=-1)
            
            # Pass messages through GVP.
            ds, dv = message_func(s_src, v_src)
            
            # Aggregate messages.
            ds = scatter_add(ds, dst, dim=0, dim_size=s.size(0))
            if v is not None:
                dv = scatter_add(dv, dst, dim=0, dim_size=v.size(0))
            
            # Update node representations.
            s = s + ds
            if v is not None:
                v = v + dv
        
        return s, v

def scatter_add(src, index, dim, dim_size):
    """Simplified scatter_add compatible with torch_scatter.scatter_add behavior."""
    result = torch.zeros(dim_size, *src.shape[1:], device=src.device)
    result.index_add_(dim, index, src)
    return result

class ProteinStructureEncoder(nn.Module):
    """
    Protein structure encoder: process protein structure using ESM-IF1 and GVP-style geometry modules.

    Inspired by ProteinGPT, with improved handling of 3D coordinates and spatial relationships.
    """
    
    def __init__(self, model_name="esm_if1_gvp4_t16_142M_UR50", freeze=True):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"ESM-IF1Model Load Device: {self.device}")
        
        if model_name in ESM_MODEL_PATHS:
            model_path = ESM_MODEL_PATHS[model_name]
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model: {model_path}")
            
            # Load model.
            self.model, self.alphabet = self._load_model_from_path(model_path, model_name)
            self.model = self.model.to(self.device)
            
            # Ensure the model exposes embed_dim.
            if hasattr(self.model, "embed_dim"):
                self.embed_dim = self.model.embed_dim
            elif hasattr(self.model, "args") and hasattr(self.model.args, "embed_dim"):
                self.embed_dim = self.model.args.embed_dim
            else:
                # Default embedding dimension for ESM-IF1.
                self.embed_dim = 512
                logger.warning(f"Model {model_name} embed_dim,Use: {self.embed_dim}")
            
            # Device validation.
            model_device = next(self.model.parameters()).device
            if str(model_device) != str(self.device):
                logger.warning(f"Model Device {model_device}, {self.device}")
                self.model = self.model.to(self.device)
            
            # Additional GVP layers to enhance geometric processing.
            self.gvp_encoder = self._create_gvp_encoder()
            
            # Freeze parameters.
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
                logger.info(" ModelParameters ")
        else:
            raise ValueError(f" Model {model_name} Config Path, ESM_MODEL_PATHS ")
    
    def _create_gvp_encoder(self):
        """Create additional geometric/vector processing layers to enhance structure representations."""
        return nn.ModuleDict({
            'node_embedding': GVP(self.embed_dim, 1, self.embed_dim, 16),
            'edge_embedding': GVP(self.embed_dim, 1, self.embed_dim, 4), 
            'conv_layers': nn.ModuleList([
                GVPConvLayer(self.embed_dim, 16, self.embed_dim)
                for _ in range(3)
            ]),
            'final_projection': nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim)
            )
        })
    
    def _load_model_from_path(self, model_path, model_name):
        """Helper to load a structure model from a local path."""
        logger.info(f"Load Model: {model_name}: {model_path}")
        # Structure checkpoints may not include argparse.Namespace; add it for compatibility.
        import argparse
        torch.serialization.add_safe_globals([argparse.Namespace])

        try:
            # Prefer weights_only=True.
            model_data = torch.load(model_path, map_location="cpu", weights_only=True)

            if isinstance(model_data, dict) and "model" in model_data:
                model_obj = model_data["model"]
                if isinstance(model_obj, nn.Module):
                    logger.info(" Load Model (weights_only=True)")
                    # Use esm.data.Alphabet.
                    return model_obj, model_data.get("alphabet", esm.data.Alphabet.from_architecture("ESM-1b"))
            elif isinstance(model_data, nn.Module):
                logger.info(" Load Model (weights_only=True)")
                # Use esm.data.Alphabet.
                return model_data, esm.data.Alphabet.from_architecture("ESM-1b")
            else:
                # Try loading via the official ESM API.
                try:
                    if hasattr(esm, 'pretrained') and hasattr(esm.pretrained, 'load_model_and_alphabet_local'):
                        model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
                        logger.info(f"UseESM API Load Model: {model_name}")
                        return model, alphabet
                    else:
                        raise AttributeError("ESM API Load Unavailable")
                except Exception as esm_err:
                    logger.warning(f"ESM APILoadFailed: {esm_err}, weights_only=False")
                    raise ValueError(" ModelData (weights_only=True)")
        except Exception as e:
            logger.error(f"Load ModelFailed (weights_only=True): {e}")
            logger.warning(" Use weights_only=False Load.")
            try:
                model_data = torch.load(model_path, map_location="cpu", weights_only=False)
                if isinstance(model_data, dict) and "model" in model_data:
                    model_obj = model_data["model"]
                    if isinstance(model_obj, nn.Module):
                        logger.info(" Load Model (weights_only=False)")
                        # Use esm.data.Alphabet.
                        return model_obj, model_data.get("alphabet", esm.data.Alphabet.from_architecture("ESM-1b"))
                elif isinstance(model_data, nn.Module):
                    logger.info(" Load Model (weights_only=False)")
                    # Use esm.data.Alphabet.
                    return model_data, esm.data.Alphabet.from_architecture("ESM-1b")
                else:
                    raise ValueError(" ModelData (weights_only=False)")
            except Exception as fallback_e:
                logger.error(f"Use weights_only=False Load Model Failed: {fallback_e}")
                raise ValueError(f" Load Model {model_name}: {fallback_e}")

    def _prepare_structure_input(self, coords):
        """
        Convert protein structure coordinates into the input format expected by ESM-IF1.
        
        Args:
            coords: Atom coordinate tensor [batch_size, seq_length, n_atoms, 3].
                Typically n_atoms=4 for backbone atoms N, CA, C, O.
        
        Returns:
            dict: Input dictionary for ESM-IF1.
        """
        batch_size, seq_length = coords.shape[0], coords.shape[1]
        
        # Ensure coordinates are on the correct device.
        if hasattr(self, 'device') and coords.device != self.device:
            coords = coords.to(self.device)
        
        # Build the input structure expected by ESM-IF1.
        structure_input = {
            "coords": {
                "N": coords[:, :, 0, :],    # Nitrogen atom coordinates
                "CA": coords[:, :, 1, :],   # Alpha-carbon coordinates
                "C": coords[:, :, 2, :],    # Carbon atom coordinates
            }
        }
        
        # If a fourth atom (oxygen) is provided, include it as well.
        if coords.shape[2] >= 4:
            structure_input["coords"]["O"] = coords[:, :, 3, :]
            
        # Add a mask (all positions are valid here).
        structure_input["mask"] = torch.ones(batch_size, seq_length, device=self.device).bool()
        
        # Additional metadata for GVP processing.
        structure_input["seq_length"] = seq_length
        
        # Compute residue-residue distance matrix using CA atoms.
        ca_coords = structure_input["coords"]["CA"]  # [batch, seq_len, 3]
        
        # Distance computation with batch dimension.
        distances = []
        for b in range(batch_size):
            # Pairwise Euclidean distances between CA atoms.
            ca_b = ca_coords[b]  # [seq_len, 3]
            dist_mat = torch.cdist(ca_b, ca_b)  # [seq_len, seq_len]
            distances.append(dist_mat)
        
        structure_input["distances"] = torch.stack(distances)  # [batch, seq_len, seq_len]
        
        return structure_input

    def _compute_structure_features(self, structure_input):
        """
        Compute additional geometric features to enrich structure representations.
        
        Args:
            structure_input: Dictionary containing atom coordinates.
            
        Returns:
            torch.Tensor: Enhanced structure features.
        """
        batch_size = structure_input["coords"]["CA"].shape[0]
        seq_length = structure_input["coords"]["CA"].shape[1]
        
        # Extract coordinates.
        ca_coords = structure_input["coords"]["CA"]  # [batch, seq_len, 3]
        n_coords = structure_input["coords"]["N"]    # [batch, seq_len, 3]
        c_coords = structure_input["coords"]["C"]    # [batch, seq_len, 3]
        
        all_features = []
        
        for b in range(batch_size):
            ca_b = ca_coords[b]  # [seq_len, 3]
            n_b = n_coords[b]    # [seq_len, 3]
            c_b = c_coords[b]    # [seq_len, 3]
            
            # Bond angle: N-CA-C
            v1 = n_b - ca_b  # N->CA vector
            v2 = c_b - ca_b  # C->CA vector
            
            # Normalize vectors.
            v1_norm = torch.norm(v1, dim=1, keepdim=True)
            v2_norm = torch.norm(v2, dim=1, keepdim=True)
            v1 = v1 / (v1_norm + 1e-7)
            v2 = v2 / (v2_norm + 1e-7)
            
            # Cosine of the bond angle via dot product.
            cos_angles = torch.sum(v1 * v2, dim=1)  # [seq_len]
            
            # Sine magnitude via cross product.
            cross_products = torch.cross(v1, v2, dim=1)  # [seq_len, 3]
            sin_norms = torch.norm(cross_products, dim=1)  # [seq_len]
            
            # Build a local coordinate frame to capture residue relationships.
            local_frames = torch.zeros(seq_length, 3, 3, device=self.device)
            
            # X axis: normalized CA->C vector.
            x_axis = c_b - ca_b
            x_axis = x_axis / (torch.norm(x_axis, dim=1, keepdim=True) + 1e-7)
            
            # Temporary Y axis: normalized CA->N vector.
            temp_y = n_b - ca_b
            temp_y = temp_y / (torch.norm(temp_y, dim=1, keepdim=True) + 1e-7)
            
            # Z axis: cross product of X and temporary Y.
            z_axis = torch.cross(x_axis, temp_y, dim=1)
            z_axis = z_axis / (torch.norm(z_axis, dim=1, keepdim=True) + 1e-7)
            
            # True Y axis: cross product of Z and X.
            y_axis = torch.cross(z_axis, x_axis, dim=1)
            
            # Assemble local frame.
            local_frames[:, :, 0] = x_axis
            local_frames[:, :, 1] = y_axis
            local_frames[:, :, 2] = z_axis
            
            # Concatenate geometric features.
            geom_features = torch.cat([
                cos_angles.unsqueeze(1),     # cosine of bond angle
                sin_norms.unsqueeze(1),      # cross-product norm
                torch.flatten(local_frames, 1, 2)  # local frame (flattened)
            ], dim=1)
            
            all_features.append(geom_features)
        
        # Stack across batches.
        return torch.stack(all_features)  # [batch, seq_len, feature_dim]

    def forward(self, structure_data):
        """
        Forward pass.
        
        Args:
            structure_data: Protein structure data, either a coordinate tensor [batch, seq_len, n_atoms, 3]
                or a preprocessed input dictionary.
        
        Returns:
            tuple: (structure_repr, per_residue_repr) - global structure representation and per-residue features
        """
        # Ensure inputs are on the correct device.
        if hasattr(self, 'device'):
            if isinstance(structure_data, torch.Tensor):
                structure_data = structure_data.to(self.device)
            elif isinstance(structure_data, dict):
                for key in structure_data:
                    if isinstance(structure_data[key], torch.Tensor):
                        structure_data[key] = structure_data[key].to(self.device)
                    elif isinstance(structure_data[key], dict):
                        for subkey in structure_data[key]:
                            if isinstance(structure_data[key][subkey], torch.Tensor):
                                structure_data[key][subkey] = structure_data[key][subkey].to(self.device)
        
        with torch.no_grad():
            try:
                # Prepare structure input.
                if isinstance(structure_data, torch.Tensor):
                    structure_input = self._prepare_structure_input(structure_data)
                    logger.debug(f", shape: {structure_data.shape}")
                else:
                    structure_input = structure_data
                    logger.debug("Use ")
                
                # Compute extra geometric features.
                geom_features = self._compute_structure_features(structure_input)
                
                # Run ESM-IF1 on the structure.
                try:
                    # Support multiple ESM-IF1 API variants.
                    if hasattr(self.model, 'forward_structure'):
                        # Newer ESM-IF1 interface.
                        output = self.model.forward_structure(structure_input)
                    else:
                        # Standard interface.
                        repr_layers = [self.model.num_layers] if hasattr(self.model, "num_layers") else [-1]
                        output = self.model(structure_input, repr_layers=repr_layers)
                    
                    if isinstance(output, dict):
                        # Extract representations from the output.
                        if "representations" in output:
                            layer_key = list(output["representations"].keys())[-1]
                            structure_repr = output["representations"][layer_key][:, 0, :]
                            per_residue_repr = output["representations"][layer_key][:, 1:, :]
                        elif "s" in output and "z" in output:
                            # ESM-IF1 may return scalar and vector representations.
                            structure_repr = output["s"][:, 0, :]
                            per_residue_repr = output["s"][:, 1:, :]
                        else:
                            raise ValueError(" ModelOutput ")
                    else:
                        # Tensor-returning output.
                        structure_repr = output[:, 0, :]
                        per_residue_repr = output[:, 1:, :]
                    
                    # Enhance per-residue representations with GVP.
                    per_residue_repr = self._enhance_with_gvp(per_residue_repr, structure_input, geom_features)
                    
                    logger.debug(f", shape: {structure_repr.shape}, Residue shape: {per_residue_repr.shape}")
                    return structure_repr, per_residue_repr
                    
                except Exception as e:
                    logger.error(f" Failed: {e}")
                    raise RuntimeError(f"ESM-IF1 Error: {e}")
                    
            except Exception as e:
                logger.error(f" Error: {e}")
                raise RuntimeError(f" Error: {e}")
                
    def _enhance_with_gvp(self, per_residue_repr, structure_input, geom_features):
        """
        Enhance per-residue representations with GVP and fuse local micro-environment features.
        
        Args:
            per_residue_repr: Per-residue representations from ESM-IF1 [batch, seq_len, embed_dim]
            structure_input: Structure input dictionary.
            geom_features: Additional geometric features [batch, seq_len, feature_dim]
            
        Returns:
            torch.Tensor: Enhanced per-residue representations.
        """
        batch_size = per_residue_repr.shape[0]
        seq_length = per_residue_repr.shape[1]
        
        # Compute local micro-environment features (e.g., solvent accessibility proxies and concavity).
        micro_env_features = self._compute_micro_environment_features(structure_input)
        
        # Project micro-environment features to the embedding dimension.
        if not hasattr(self, 'micro_env_projection'):
            self.micro_env_projection = nn.Sequential(
                nn.Linear(3, self.embed_dim // 4),
                nn.ReLU(),
                nn.Linear(self.embed_dim // 4, self.embed_dim),
                nn.LayerNorm(self.embed_dim)
            ).to(per_residue_repr.device)
            
        micro_env_emb = self.micro_env_projection(micro_env_features)

        # For simplicity, merge geometric features directly into the representation.
        enhanced_repr = per_residue_repr.clone()
        
        # Fuse micro-environment features (enhance local awareness).
        enhanced_repr = enhanced_repr + 0.2 * micro_env_emb
        
        # If available, use distance information to further enhance representations.
        if "distances" in structure_input:
            # Create distance-based context features for each residue.
            distances = structure_input["distances"]  # [batch, seq_len, seq_len]
            
            for b in range(batch_size):
                # Extract distance matrix for this batch.
                dist_mat = distances[b]  # [seq_len, seq_len]
                
                # Compute distance-weighted context features.
                # Weights decay with distance: exp(-dist^2 / 10^2)
                weights = torch.exp(-(dist_mat ** 2) / 100).unsqueeze(-1)  # [seq_len, seq_len, 1]
                
                # Weighted average via broadcasted multiplication: [seq_len, seq_len, 1] * [seq_len, embed_dim]
                context_features = torch.sum(
                    weights * enhanced_repr[b].unsqueeze(0).expand(seq_length, -1, -1),
                    dim=1
                )  # [seq_len, embed_dim]
                
                # Update representations for this batch.
                enhanced_repr[b] = enhanced_repr[b] + 0.1 * context_features
        
        return enhanced_repr

    def _compute_micro_environment_features(self, structure_input):
        """
        Compute micro-environment geometric features (e.g., solvent accessibility proxy, local concavity).
        
        Args:
            structure_input: Dictionary containing atom coordinates.
            
        Returns:
            torch.Tensor: Micro-environment features [batch, seq_len, 3]
                - Channel 0: Local density / SASA proxy
                - Channel 1: Local concavity / protrusion
                - Channel 2: Mean neighbor distance
        """
        ca_coords = structure_input["coords"]["CA"] # [batch, seq_len, 3]
        batch_size, seq_len, _ = ca_coords.shape
        device = ca_coords.device

        micro_env_features = []

        for b in range(batch_size):
            coords = ca_coords[b] # [seq_len, 3]
            # Distance matrix.
            dist_mat = torch.cdist(coords, coords) # [seq_len, seq_len]
            
            # 1. Local density (SASA proxy): number of neighbors within 10 Å.
            radius = 10.0
            # Exclude self (distance > 0).
            mask = (dist_mat < radius) & (dist_mat > 1e-6)
            local_density = mask.sum(dim=1).float().unsqueeze(1) # [seq_len, 1]
            
            # 2. Local concavity / protrusion.
            # Compute neighbor centroid.
            neighbor_sum = torch.matmul(mask.float(), coords) # [seq_len, 3]
            neighbor_count = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
            centroids = neighbor_sum / neighbor_count # [seq_len, 3]
            
            # Distance from residue to neighbor centroid.
            # Protruding residues tend to be farther from their neighbor centroid.
            dist_to_centroid = torch.norm(coords - centroids, dim=1).unsqueeze(1) # [seq_len, 1]
            
            # 3. Mean neighbor distance.
            avg_dist = (dist_mat * mask.float()).sum(dim=1, keepdim=True) / neighbor_count # [seq_len, 1]
            
            # Rough normalization for easier optimization.
            features = torch.cat([
                local_density / 20.0,      # density is typically in ~[0, 50]
                dist_to_centroid / 5.0,    # centroid distance is typically in ~[0, 10] Å
                avg_dist / 10.0            # mean distance is typically in ~[0, 10] Å
            ], dim=1)
            
            micro_env_features.append(features)
            
        return torch.stack(micro_env_features)

class AcetylationInfoEncoder(nn.Module):
    """
    Acetylation information encoder for acetylation-site prediction.

    Inspired by ProteinGPT-style design, specialized for acetylation-related inputs.
    """
    
    def __init__(self, input_dim=21, hidden_dim=128, output_dim=512):
        """
        Initialize the acetylation information encoder.
        
        Args:
            input_dim: Input dimension (number of amino-acid types + 1).
            hidden_dim: Hidden dimension.
            output_dim: Output dimension; should match other encoder output dimensions.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Amino-acid embedding layer.
        self.aa_embedding = nn.Embedding(input_dim, hidden_dim)
        
        # Positional encoding.
        self.max_seq_len = 2048
        self.position_encoding = nn.Embedding(self.max_seq_len, hidden_dim)
        
        # Site feature extractor.
        self.site_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Local context encoder.
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim*4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=2
        )
        
        # Output projection.
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, aa_sequence, positions=None):
        """
        Forward pass.
        
        Args:
            aa_sequence: Integer-encoded amino-acid sequence [batch_size, seq_len]
            positions: Candidate acetylation positions [batch_size]; defaults to the sequence midpoint.
            
        Returns:
            torch.Tensor: Acetylation-related features [batch_size, output_dim]
        """
        batch_size, seq_len = aa_sequence.shape
        device = aa_sequence.device
        
        # Use default positions (sequence center) if not provided.
        if positions is None:
            positions = torch.tensor([seq_len // 2] * batch_size, device=device)
        
        # Amino-acid embeddings.
        aa_embeddings = self.aa_embedding(aa_sequence)  # [batch, seq_len, hidden_dim]
        
        # Add positional encodings.
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_ids = torch.clamp(position_ids, max=self.max_seq_len-1)
        pos_embeddings = self.position_encoding(position_ids)  # [batch, seq_len, hidden_dim]
        
        # Combine embeddings.
        embeddings = aa_embeddings + pos_embeddings  # [batch, seq_len, hidden_dim]
        
        # Extract center-site features.
        site_indices = torch.arange(batch_size, device=device)
        center_features = embeddings[site_indices, positions]  # [batch, hidden_dim]
        
        # Encode center-site features.
        site_features = self.site_encoder(center_features)  # [batch, hidden_dim]
        
        # Build an attention-like mask to emphasize the center position.
        attn_mask = torch.ones(batch_size, seq_len, device=device)
        for i, pos in enumerate(positions):
            # Center importance is 1; others decay with distance.
            distances = torch.abs(torch.arange(seq_len, device=device) - pos)
            attn_mask[i] = torch.exp(-distances.float() / 10)
        
        # Apply the context encoder.
        context_features = self.context_encoder(embeddings)  # [batch, seq_len, hidden_dim]
        
        # Weighted aggregation of context features.
        weighted_context = context_features * attn_mask.unsqueeze(-1)  # [batch, seq_len, hidden_dim]
        context_sum = weighted_context.sum(dim=1)  # [batch, hidden_dim]
        context_weight_sum = attn_mask.sum(dim=1, keepdim=True)  # [batch, 1]
        aggregated_context = context_sum / context_weight_sum  # [batch, hidden_dim]
        
        # Merge site and context features.
        combined_features = site_features + aggregated_context  # [batch, hidden_dim]
        
        # Project to the output dimension.
        output = self.output_projection(combined_features)  # [batch, output_dim]
        
        return output

class ProjectionLayer(nn.Module):
    """
    Projection layer: map outputs from different encoders into a shared representation space.

    Designed to support fusion of sequence, structure, and acetylation information.
    """
    
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        # [Modified] Enhanced projection with intermediate layer to avoid information bottleneck
        # and increase non-linearity as requested.
        # Strategy: Retain high-dimensional features longer by projecting to an intermediate dimension
        # instead of directly to the small output dimension.
        mid_dim = max(output_dim, input_dim // 2)
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """Project input features into a shared representation space."""
        return self.projection(x)

class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion layer for sequence, structure, and acetylation information.

    Uses attention to weight and fuse different modalities.
    """
    
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Multi-head self-attention.
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization.
        self.layer_norm1 = nn.LayerNorm(feature_dim)
        self.layer_norm2 = nn.LayerNorm(feature_dim)
        
        # Feed-forward network.
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(0.1)
        )
        
        # Attention pooling layer (for weighted fusion).
        self.attn_pool = nn.Linear(feature_dim, 1)
        
    def forward(self, *features):
        """
        Forward pass: fuse multiple feature tensors.
        
        Args:
            *features: Variable number of feature tensors, each of shape [batch_size, seq_len, feature_dim].
            
        Returns:
            torch.Tensor: Fused features [batch_size, seq_len, feature_dim]
        """
        # Validate inputs.
        valid_features = [f for f in features if f is not None]
        if not valid_features:
            raise ValueError(" ")
        
        if len(valid_features) == 1:
            # If only one feature is provided, return it directly.
            return valid_features[0]
            
        # Stack features: [batch_size, num_features, seq_len, feature_dim]
        stacked = torch.stack(valid_features, dim=1)
        batch_size, num_features, seq_len, feature_dim = stacked.shape
        
        # Reshape for attention: [batch_size * seq_len, num_features, feature_dim]
        reshaped = stacked.transpose(1, 2).reshape(batch_size * seq_len, num_features, feature_dim)
        
        # Self-attention fusion (cross-modality interaction).
        attn_output, _ = self.attention(reshaped, reshaped, reshaped)
        
        # Residual connection + layer norm.
        norm_output = self.layer_norm1(reshaped + attn_output)
        
        # Feed-forward network.
        ffn_output = self.ffn(norm_output)
        
        # Residual connection + layer norm.
        output = self.layer_norm2(norm_output + ffn_output)
        
        # [Modified] Use attention pooling instead of mean pooling.
        # Compute modality weights: [batch*seq, num_features, 1]
        scores = self.attn_pool(output)
        weights = F.softmax(scores, dim=1)
        
        # Weighted sum.
        fused = (output * weights).sum(dim=1)
        
        # Reshape back to [batch_size, seq_len, feature_dim]
        fused = fused.reshape(batch_size, seq_len, feature_dim)
        
        return fused
