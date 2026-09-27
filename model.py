import torch
import torch.nn.functional as F
from torch import nn


class SymmetricInfoNCE(nn.Module):

    def __init__(self, temperature: float = 0.2):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.ndim != 2 or second.ndim != 2 or first.shape != second.shape:
            raise ValueError("InfoNCE inputs must have the same shape [batch, features]")
        if first.shape[0] == 0:
            raise ValueError("InfoNCE does not support an empty batch")
        first = F.normalize(first, dim=-1)
        second = F.normalize(second, dim=-1)
        logits = first @ second.T / self.temperature
        labels = torch.arange(first.shape[0], device=first.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


class MolecularGINE(nn.Module):

    def __init__(self, atom_dim: int, bond_dim: int, hidden_dim: int = 256, layers: int = 3):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        self.atom_projection = nn.Linear(atom_dim, hidden_dim)
        self.bond_projections = nn.ModuleList(nn.Linear(bond_dim, hidden_dim) for _ in range(layers))
        self.mlps = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_dim, 2 * hidden_dim), nn.ReLU(), nn.Linear(2 * hidden_dim, hidden_dim))
            for _ in range(layers)
        )
        self.eps = nn.Parameter(torch.zeros(layers))

    def forward(
        self,
        atoms: torch.Tensor,
        bonds: torch.Tensor,
        bond_index: torch.Tensor,
        molecule_index: torch.Tensor,
        num_molecules: int,
    ) -> torch.Tensor:
        if bond_index.dtype != torch.long or bond_index.shape != (2, bonds.shape[0]):
            raise ValueError("bond_index must be [2, num_directed_bonds] with dtype long")
        if molecule_index.dtype != torch.long or molecule_index.shape != (atoms.shape[0],):
            raise ValueError("molecule_index must contain one long index per atom")
        if num_molecules < 1 or atoms.shape[0] < num_molecules:
            raise ValueError("each molecule must contain at least one atom")
        if molecule_index.numel() and (molecule_index.min() < 0 or molecule_index.max() >= num_molecules):
            raise ValueError("molecule_index is out of range")
        if bond_index.numel() and (bond_index.min() < 0 or bond_index.max() >= atoms.shape[0]):
            raise ValueError("bond_index is out of range")
        source, destination = bond_index
        if bond_index.numel() and not torch.equal(molecule_index[source], molecule_index[destination]):
            raise ValueError("bonds must connect atoms of the same molecule")
        h = self.atom_projection(atoms)
        for layer, bond_projection, mlp in zip(self.eps, self.bond_projections, self.mlps):
            messages = F.relu(h[source] + bond_projection(bonds))
            aggregated = torch.zeros_like(h).index_add(0, destination, messages)
            h = mlp((1 + layer) * h + aggregated)
        pooled = h.new_zeros((num_molecules, h.shape[-1])).index_add(0, molecule_index, h)
        counts = h.new_zeros((num_molecules, 1)).index_add(
            0, molecule_index, h.new_ones((atoms.shape[0], 1))
        )
        if (counts == 0).any():
            raise ValueError("every molecule must contain at least one atom")
        return pooled / counts


class ViewFusion(nn.Module):

    def __init__(self, first_dim: int, second_dim: int, feature_dim: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.first_projection = nn.Linear(first_dim, feature_dim)
        self.second_projection = nn.Linear(second_dim, feature_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.type_embedding = nn.Parameter(torch.randn(1, 3, feature_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            feature_dim, heads, 4 * feature_dim, dropout=dropout, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, layers)
        self.norm = nn.LayerNorm(feature_dim)

    def forward_with_views(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if first.ndim != 2 or second.ndim != 2 or first.shape[0] != second.shape[0]:
            raise ValueError("both views must contain the same number of entities")
        first_view = self.first_projection(first)
        second_view = self.second_projection(second)
        tokens = torch.stack(
            (self.cls.expand(first.shape[0], -1, -1).squeeze(1), first_view, second_view), dim=1
        )
        fused = self.norm(self.transformer(tokens + self.type_embedding)[:, 0])
        return F.normalize(first_view, dim=-1), F.normalize(second_view, dim=-1), F.normalize(fused, dim=-1)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return self.forward_with_views(first, second)[-1]


class MultiViewEntityRepresentation(nn.Module):

    def __init__(
        self, atom_dim: int, bond_dim: int, drug_descriptor_dim: int,
        esm_dim: int, feature_dim: int = 256, heads: int = 4,
        transformer_layers: int = 1, gine_layers: int = 3, dropout: float = 0.2,
        edge_drop_prob: float = 0.2, atom_mask_prob: float = 0.15,
        contrastive_temperature: float = 0.2,
    ):
        super().__init__()
        if not 0 <= edge_drop_prob < 1 or not 0 <= atom_mask_prob < 1:
            raise ValueError("augmentation probabilities must be in [0, 1)")
        self.edge_drop_prob = edge_drop_prob
        self.atom_mask_prob = atom_mask_prob
        self.molecular_encoder = MolecularGINE(atom_dim, bond_dim, feature_dim, gine_layers)
        self.molecule_contrastive_projection = nn.Linear(feature_dim, feature_dim)
        self.drug_fusion = ViewFusion(feature_dim, drug_descriptor_dim, feature_dim, heads, transformer_layers, dropout)
        self.target_fusion = ViewFusion(2 * esm_dim, 567, feature_dim, heads, transformer_layers, dropout)
        self.infonce = SymmetricInfoNCE(contrastive_temperature)

    @staticmethod
    def augment_molecular_graph(
        atoms: torch.Tensor,
        bonds: torch.Tensor,
        bond_index: torch.Tensor,
        edge_drop_prob: float,
        atom_mask_prob: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if not 0 <= edge_drop_prob < 1 or not 0 <= atom_mask_prob < 1:
            raise ValueError("augmentation probabilities must be in [0, 1)")
        atom_view = atoms.clone()
        if atom_view.numel():
            atom_view = atom_view.masked_fill(
                torch.rand(atom_view.shape[0], 1, device=atoms.device) < atom_mask_prob, 0
            )
        if bonds.shape[0] == 0 or edge_drop_prob == 0:
            return atom_view, bonds, bond_index
        # Directed rows for the same undirected bond must be dropped together.
        endpoints = torch.sort(bond_index, dim=0).values.T
        _, groups = torch.unique(endpoints, dim=0, return_inverse=True)
        keep_group = torch.rand(int(groups.max()) + 1, device=bonds.device) >= edge_drop_prob
        keep = keep_group[groups]
        return atom_view, bonds[keep], bond_index[:, keep]

    @staticmethod
    def fusion_contrastive_loss(
        first: torch.Tensor, second: torch.Tensor, fused: torch.Tensor, infonce: SymmetricInfoNCE
    ) -> torch.Tensor:

        return (
            infonce(first, second)
            + 0.5 * infonce(fused, first)
            + 0.5 * infonce(fused, second)
        )

    def _encode_triplets(
        self,
        molecules: torch.Tensor,
        drug_descriptors: torch.Tensor,
        target_plm: torch.Tensor,
        target_descriptors: torch.Tensor,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return {
            "drug": self.drug_fusion.forward_with_views(molecules, drug_descriptors),
            "target": self.target_fusion.forward_with_views(target_plm, target_descriptors),
        }

    def pretraining_loss(
        self,
        atoms: torch.Tensor,
        bonds: torch.Tensor,
        bond_index: torch.Tensor,
        molecule_index: torch.Tensor,
        drug_descriptors: torch.Tensor,
        target_plm: torch.Tensor,
        target_descriptors: torch.Tensor,
        *,
        include_graph_contrastive: bool = True,
        graph_contrastive_weight: float = 1.0,
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:

        if graph_contrastive_weight < 0:
            raise ValueError("graph_contrastive_weight must be non-negative")
        if drug_descriptors.shape[0] < 2 or target_descriptors.shape[0] < 2:
            raise ValueError("pretraining requires at least two drugs and two targets per batch")
        view_args = [
            self.augment_molecular_graph(
                atoms, bonds, bond_index, self.edge_drop_prob, self.atom_mask_prob
            )
            for _ in range(2)
        ]
        molecules = [
            self.molecular_encoder(atom_view, bond_view, edge_view, molecule_index, drug_descriptors.shape[0])
            for atom_view, bond_view, edge_view in view_args
        ]
        triplets = [
            self._encode_triplets(molecule, drug_descriptors, target_plm, target_descriptors)
            for molecule in molecules
        ]
        drug_fusion = torch.stack(
            [self.fusion_contrastive_loss(*triplet["drug"], self.infonce) for triplet in triplets]
        ).mean()
        target_fusion = self.fusion_contrastive_loss(*triplets[0]["target"], self.infonce)
        if include_graph_contrastive and graph_contrastive_weight > 0:
            graph_loss = self.infonce(
                self.molecule_contrastive_projection(molecules[0]),
                self.molecule_contrastive_projection(molecules[1]),
            )
        else:
            graph_loss = drug_fusion.new_zeros(())
        total = drug_fusion + target_fusion + graph_contrastive_weight * graph_loss
        if return_components:
            return {
                "loss": total,
                "drug_fusion_loss": drug_fusion,
                "target_fusion_loss": target_fusion,
                "graph_contrastive_loss": graph_loss,
            }
        return total

    @staticmethod
    def pool_esm_chunks(chunks: list[list[torch.Tensor]]) -> torch.Tensor:
        representations = []
        for sequence in chunks:
            if not sequence or any(chunk.ndim != 2 or chunk.shape[0] == 0 or chunk.shape[0] > 1000 for chunk in sequence):
                raise ValueError("each target requires nonempty ESM-2 chunks of at most 1000 residues")
            length = sum(chunk.shape[0] for chunk in sequence)
            mean = sum(chunk.sum(dim=0) for chunk in sequence) / length
            maximum = torch.stack([chunk.amax(dim=0) for chunk in sequence]).amax(dim=0)
            representations.append(torch.cat((mean, maximum)))
        return torch.stack(representations)

    def forward(
        self, atoms: torch.Tensor, bonds: torch.Tensor, bond_index: torch.Tensor,
        molecule_index: torch.Tensor, drug_descriptors: torch.Tensor,
        target_plm: torch.Tensor, target_descriptors: torch.Tensor,
    ) -> torch.Tensor:
        molecules = self.molecular_encoder(atoms, bonds, bond_index, molecule_index, drug_descriptors.shape[0])
        drug = self.drug_fusion(molecules, drug_descriptors)
        target = self.target_fusion(target_plm, target_descriptors)
        return torch.cat((drug, target), dim=0)


class LatentSemanticMultiFrequencyGuidance(nn.Module):
    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 64,
        bands: int = 4,
        dropout: float = 0.2,
        band_dropout: float = 0.3,
        global_entropy_floor: float = 0.8,
        node_entropy_ceiling: float = 0.5,
    ):
        super().__init__()
        if bands < 2:
            raise ValueError("bands must be at least two")
        if not 0 <= global_entropy_floor <= 1 or not 0 <= node_entropy_ceiling <= 1:
            raise ValueError("entropy thresholds must be in [0, 1]")
        self.bands = bands
        self.band_dropout = band_dropout
        self.global_entropy_floor = global_entropy_floor
        self.node_entropy_ceiling = node_entropy_ceiling
        self.projection = nn.Linear(feature_dim, hidden_dim)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.structure_projection = nn.Linear(hidden_dim, hidden_dim)
        self.signal_projection = nn.Linear(hidden_dim, hidden_dim)
        self.band_embedding = nn.Parameter(torch.randn(bands, hidden_dim) * 0.02)
        self.scorer = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.guidance_projection = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.latest_entropy: dict[str, torch.Tensor] | None = None
        self.latest_band_weights: torch.Tensor | None = None

    @staticmethod
    def propagate(structure: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        degree = structure @ structure.sum(dim=0) - 1
        inv_sqrt = degree.clamp_min(1e-3).rsqrt().unsqueeze(1)
        return inv_sqrt * (structure @ (structure.T @ (inv_sqrt * signal))) - inv_sqrt.square() * signal

    def routing_regularization(self, weights: torch.Tensor | None = None) -> torch.Tensor:

        if weights is None:
            weights = self.latest_band_weights
        if weights is None:
            raise RuntimeError("routing weights are unavailable; run the LSMFG forward pass first")
        if weights.ndim != 2 or weights.shape[1] != self.bands or weights.shape[0] == 0:
            raise ValueError("weights must have shape [num_nodes, bands]")
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("routing weights must be finite and non-negative")
        if not torch.allclose(weights.sum(dim=1), weights.new_ones(weights.shape[0]), atol=1e-5):
            raise ValueError("routing weights must sum to one for every node")
        eps = torch.finfo(weights.dtype).eps
        global_weights = weights.mean(dim=0)
        log_bands = torch.log(weights.new_tensor(float(self.bands)))
        global_entropy = -(global_weights * global_weights.clamp_min(eps).log()).sum() / log_bands
        node_entropy = -(weights * weights.clamp_min(eps).log()).sum(dim=1).mean() / log_bands
        regularizer = (
            F.relu(weights.new_tensor(self.global_entropy_floor) - global_entropy).square()
            + F.relu(node_entropy - weights.new_tensor(self.node_entropy_ceiling)).square()
        )
        self.latest_entropy = {
            "global_entropy": global_entropy.detach(),
            "node_entropy": node_entropy.detach(),
            "regularizer": regularizer.detach(),
        }
        return regularizer

    def routing_statistics(self) -> dict[str, torch.Tensor]:

        if self.latest_entropy is None:
            raise RuntimeError("routing statistics are unavailable; run the LSMFG forward pass first")
        return dict(self.latest_entropy)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("features must have shape [num_nodes, feature_dim] and be non-empty")
        base = self.dropout(F.relu(self.norm(self.projection(features))))
        structure = F.normalize(torch.sigmoid(self.structure_projection(base)), dim=1)
        signal = self.signal_projection(base)
        frequency_bands = []
        for k in range(self.bands):
            response = signal
            for _ in range(self.bands - 1 - k):
                response = 0.5 * (response + self.propagate(structure, response))
            for _ in range(k):
                response = 0.5 * (response - self.propagate(structure, response))
            frequency_bands.append(F.dropout(self.bands / 2 * response, self.band_dropout, self.training))
        frequency_bands = torch.stack(frequency_bands, dim=1)
        identity = self.band_embedding.unsqueeze(0).expand(features.shape[0], -1, -1)
        scores = self.scorer(torch.cat((frequency_bands, identity), dim=-1)).squeeze(-1)
        weights = scores.softmax(dim=1)
        guidance = self.guidance_projection((weights.unsqueeze(-1) * frequency_bands).sum(dim=1))
        gate = torch.sigmoid(self.gate(torch.cat((base, guidance), dim=1)))
        self.latest_band_weights = weights
        self.routing_regularization(weights)
        return base + gate * guidance, base, weights


class RelationalDiffusionTrajectory(nn.Module):

    def __init__(self, hidden_dim: int = 64, steps: int = 3, restart_alpha: float = 0.1):
        super().__init__()
        if steps < 1 or not 0 < restart_alpha < 1:
            raise ValueError("steps must be positive and restart_alpha must be in (0, 1)")
        self.steps = steps
        self.restart_alpha = restart_alpha
        self.positions = nn.Parameter(torch.randn(steps + 1, hidden_dim) * 0.02)

    def forward(self, features: torch.Tensor, positive_train_pairs: torch.Tensor, num_drugs: int) -> torch.Tensor:
        if positive_train_pairs.dtype != torch.long or positive_train_pairs.ndim != 2 or positive_train_pairs.shape[1] != 2:
            raise ValueError("positive_train_pairs must be [E, 2] with dtype long and local drug/target indices")
        num_nodes = features.shape[0]
        if not 0 < num_drugs < num_nodes:
            raise ValueError("num_drugs must leave at least one target")
        if positive_train_pairs.numel() and (
            positive_train_pairs[:, 0].min() < 0 or positive_train_pairs[:, 0].max() >= num_drugs
            or positive_train_pairs[:, 1].min() < 0 or positive_train_pairs[:, 1].max() >= num_nodes - num_drugs
        ):
            raise ValueError("training pairs contain out-of-range indices")
        drug, target = positive_train_pairs.unbind(dim=1)
        target = target + num_drugs
        nodes = torch.arange(num_nodes, device=features.device)
        row = torch.cat((drug, target, nodes))
        col = torch.cat((target, drug, nodes))
        edges = torch.stack((row, col))
        adjacency = torch.sparse_coo_tensor(edges, torch.ones_like(row, dtype=features.dtype), (num_nodes, num_nodes)).coalesce()
        row, col = adjacency.indices()
        values = torch.ones_like(adjacency.values())
        degree = features.new_zeros(num_nodes).index_add(0, row, values)
        normalized = torch.sparse_coo_tensor(
            adjacency.indices(), values * degree[row].rsqrt() * degree[col].rsqrt(),
            (num_nodes, num_nodes)
        ).coalesce()
        states = [features]
        state = features
        for _ in range(self.steps):
            state = (1 - self.restart_alpha) * torch.sparse.mm(normalized, state) + self.restart_alpha * features
            states.append(state)
        return torch.stack(states, dim=1) + self.positions


class SelectiveTrajectoryEncoding(nn.Module):

    def __init__(self, hidden_dim: int = 64, state_dim: int = 32, dt_rank: int = 16, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.norm = nn.LayerNorm(hidden_dim)
        self.input_projection = nn.Linear(hidden_dim, hidden_dim)
        self.parameter_projection = nn.Linear(hidden_dim, dt_rank + 2 * state_dim)
        self.delta_projection = nn.Linear(dt_rank, hidden_dim)
        self.transition_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1).float()).expand(hidden_dim, -1).clone())
        self.skip = nn.Parameter(torch.ones(hidden_dim))
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.dt_rank = dt_rank

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        if trajectory.ndim != 3 or trajectory.shape[2] != self.hidden_dim:
            raise ValueError("trajectory must have shape [nodes, steps + 1, hidden_dim]")
        values = self.input_projection(self.norm(trajectory))
        parameters = F.dropout(F.silu(self.parameter_projection(values)), self.dropout, self.training)
        delta_raw, write, read = parameters.split((self.dt_rank, self.state_dim, self.state_dim), dim=-1)
        delta = F.softplus(self.delta_projection(delta_raw))
        transition = -self.transition_log.exp()
        state = values.new_zeros((values.shape[0], self.hidden_dim, self.state_dim))
        output = None
        for step in range(values.shape[1]):
            scale = delta[:, step].unsqueeze(-1)
            state = torch.exp(scale * transition) * state + scale * write[:, step].unsqueeze(1) * values[:, step].unsqueeze(-1)
            state = F.dropout(state, self.dropout, self.training)
            result = (state * read[:, step].unsqueeze(1)).sum(dim=-1) + self.skip * values[:, step]
            output = trajectory[:, step] + self.output_projection(result)
        return output


class DTSGDTI(nn.Module):

    def __init__(
        self, atom_dim: int, bond_dim: int, drug_descriptor_dim: int, esm_dim: int,
        feature_dim: int = 256, hidden_dim: int = 64, bands: int = 4,
        steps: int = 3, restart_alpha: float = 0.1, state_dim: int = 32,
        dt_rank: int = 16, dropout: float = 0.2,
        global_entropy_floor: float = 0.8, node_entropy_ceiling: float = 0.5,
        band_regularization_weight: float = 0.001,
    ):
        super().__init__()
        if band_regularization_weight < 0:
            raise ValueError("band_regularization_weight must be non-negative")
        self.band_regularization_weight = band_regularization_weight
        self.mver = MultiViewEntityRepresentation(atom_dim, bond_dim, drug_descriptor_dim, esm_dim, feature_dim, dropout=dropout)
        self.lsmfg = LatentSemanticMultiFrequencyGuidance(
            feature_dim, hidden_dim, bands, dropout,
            global_entropy_floor=global_entropy_floor, node_entropy_ceiling=node_entropy_ceiling,
        )
        self.rdt = RelationalDiffusionTrajectory(hidden_dim, steps, restart_alpha)
        self.ste = SelectiveTrajectoryEncoding(hidden_dim, state_dim, dt_rank, dropout)
        self.decoder = nn.Sequential(nn.Linear(4 * hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.band_weights = None

    def training_loss(
        self, probabilities: torch.Tensor, labels: torch.Tensor, *, return_components: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Eq. (L_DTI), using routing weights from the same forward pass."""
        if self.band_weights is None or self.band_weights is not self.lsmfg.latest_band_weights:
            raise RuntimeError("run DTSGDTI.forward before computing the training loss")
        if probabilities.ndim != 1 or labels.shape != probabilities.shape or probabilities.numel() == 0:
            raise ValueError("probabilities and labels must be nonempty vectors of equal length")
        bce = F.binary_cross_entropy(probabilities, labels.to(probabilities.dtype))
        band_loss = self.lsmfg.routing_regularization(self.band_weights)
        total = bce + self.band_regularization_weight * band_loss
        if return_components:
            return {"loss": total, "bce_loss": bce, "band_regularization": band_loss}
        return total

    def pretraining_loss(self, *args, **kwargs):
 
        return self.mver.pretraining_loss(*args, **kwargs)

    def encode(
        self, atoms: torch.Tensor, bonds: torch.Tensor, bond_index: torch.Tensor,
        molecule_index: torch.Tensor, drug_descriptors: torch.Tensor,
        target_plm: torch.Tensor, target_descriptors: torch.Tensor,
        positive_train_pairs: torch.Tensor,
    ) -> torch.Tensor:
        features = self.mver(atoms, bonds, bond_index, molecule_index, drug_descriptors, target_plm, target_descriptors)
        guided, attributes, self.band_weights = self.lsmfg(features)
        trajectory = self.rdt(guided, positive_train_pairs, drug_descriptors.shape[0])
        return self.ste(trajectory) + attributes

    def forward(
        self, atoms: torch.Tensor, bonds: torch.Tensor, bond_index: torch.Tensor,
        molecule_index: torch.Tensor, drug_descriptors: torch.Tensor,
        target_plm: torch.Tensor, target_descriptors: torch.Tensor,
        positive_train_pairs: torch.Tensor, candidate_pairs: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_pairs.dtype != torch.long or candidate_pairs.ndim != 2 or candidate_pairs.shape[1] != 2:
            raise ValueError("candidate_pairs must be [pairs, 2] with dtype long and local drug/target indices")
        num_drugs = drug_descriptors.shape[0]
        num_targets = target_descriptors.shape[0]
        if candidate_pairs.numel() and (
            candidate_pairs[:, 0].min() < 0 or candidate_pairs[:, 0].max() >= num_drugs
            or candidate_pairs[:, 1].min() < 0 or candidate_pairs[:, 1].max() >= num_targets
        ):
            raise ValueError("candidate_pairs contain out-of-range indices")
        nodes = self.encode(
            atoms, bonds, bond_index, molecule_index, drug_descriptors,
            target_plm, target_descriptors, positive_train_pairs
        )
        drug = nodes[candidate_pairs[:, 0]]
        target = nodes[candidate_pairs[:, 1] + num_drugs]
        pair_features = torch.cat((drug, target, (drug - target).abs(), drug * target), dim=-1)
        return torch.sigmoid(self.decoder(pair_features).squeeze(-1))
