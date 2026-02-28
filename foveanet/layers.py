import torch
import torch.nn as nn
import torch.nn.functional as F

class RetinalContrastGraphBuilder(nn.Module):
    """
    Stage 0 & 1: Retinal_Contrast_Event_Encoder & Dynamic_Sparse_Graph_Builder
    (Generation D "FoveaNet-Δ" Standard)
    
    1) Extracts a 64x64 cortical magnification (factor 4.2) at the image centroid.
    2) Applies a Difference of Gaussians (DoG) filter bank.
    3) Converts non-zero events into a sparse graph with k-NN (k=6) topology.
    """
    def __init__(self, distortion_factor=4.2, foveal_crop_limit=4.05, k_neighbors=6, max_nodes=196):
        super().__init__()
        self.r_distortion = distortion_factor
        self.crop_limit = foveal_crop_limit
        self.k = k_neighbors
        self.max_nodes = max_nodes
        
        # 3 Scales of ON-center/OFF-center filters
        # Approximated here as 3x3, 5x5, 9x9 convs over grayscale
        self.dog_1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.dog_2 = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False)
        self.dog_3 = nn.Conv2d(1, 1, kernel_size=9, padding=4, bias=False)
        
        # Initialize fixed biological DoG weights
        with torch.no_grad():
            self._init_dog_kernel(self.dog_1, sigma_center=0.5, sigma_surround=1.0)
            self._init_dog_kernel(self.dog_2, sigma_center=1.0, sigma_surround=2.0)
            self._init_dog_kernel(self.dog_3, sigma_center=2.0, sigma_surround=4.0)
            
    def _init_dog_kernel(self, conv_layer, sigma_center, sigma_surround):
        k = conv_layer.kernel_size[0]
        center = k // 2
        
        y, x = torch.meshgrid(torch.arange(k, dtype=torch.float32), torch.arange(k, dtype=torch.float32), indexing='ij')
        dist_sq = (x - center)**2 + (y - center)**2
        
        g_center = torch.exp(-dist_sq / (2 * sigma_center**2)) / (2 * torch.pi * sigma_center**2)
        g_surround = torch.exp(-dist_sq / (2 * sigma_surround**2)) / (2 * torch.pi * sigma_surround**2)
        
        # DoG = Center - Surround
        dog = g_center - g_surround
        # Normalize sum to essentially 0
        dog = dog - dog.mean() 
        conv_layer.weight.copy_(dog.unsqueeze(0).unsqueeze(0))

        # Freeze the weights (retinal pre-processing)
        conv_layer.weight.requires_grad = False

    def forward(self, image):
        B, C, H, W = image.shape
        device = image.device
        
        # 1. Cortical Magnification
        # Extract 64x64 crop focused on centroid
        theta = torch.zeros(B, 2, 3, device=device)
        scale = 16.0 / H  # 16x16 field of view mapped -> 64x64
        theta[:, 0, 0] = scale
        theta[:, 1, 1] = scale
        
        grid = F.affine_grid(theta, size=[B, C, 64, 64], align_corners=False)
        crop = F.grid_sample(image, grid, align_corners=False)
        gray = crop.mean(dim=1, keepdim=True) # (B, 1, 64, 64)
        
        # 2. DoG Filter Bank
        r1 = self.dog_1(gray)
        r2 = self.dog_2(gray)
        r3 = self.dog_3(gray)
        
        responses = torch.cat([r1, r2, r3], dim=1) # (B, 3, 64, 64)
        
        # 3. Dynamic Sparse Graph Construction
        # Since graph sizes vary per batch element, we compile lists
        all_node_feats = []
        all_edge_indices = []
        all_edge_feats = []
        all_batch_idx = []
        
        global_node_offset = 0
        
        for b in range(B):
            resp = responses[b] # (3, 64, 64)
            
            # Find significant events (magnitude > 0.1)
            mag = resp.abs()
            mask = mag > 0.1
            
            c_idx, y_idx, x_idx = torch.where(mask)
            
            num_events = len(c_idx)
            if num_events == 0:
                continue
                
            # Cap at max nodes by picking the strongest responses
            if num_events > self.max_nodes:
                vals = mag[c_idx, y_idx, x_idx]
                _, top_k = torch.topk(vals, self.max_nodes)
                c_idx, y_idx, x_idx = c_idx[top_k], y_idx[top_k], x_idx[top_k]
                num_events = self.max_nodes
                
            # Node Features: [x_cortical, y_cortical, polarity, scale_index, eccentricity]
            y_cortical = (y_idx.float() / 63.0) * 2.0 - 1.0
            x_cortical = (x_idx.float() / 63.0) * 2.0 - 1.0
            eccentricity = torch.sqrt(x_cortical**2 + y_cortical**2)
            scale_idx = c_idx.float()
            
            # Polarity: +1 if ON, -1 if OFF
            polarity = torch.sign(resp[c_idx, y_idx, x_idx])
            
            nodes = torch.stack([x_cortical, y_cortical, polarity, scale_idx, eccentricity], dim=1)
            
            # Edges via k-NN
            if num_events > 1:
                # Pairwise distance
                coords = nodes[:, :2] # (N, 2)
                dist_matrix = torch.cdist(coords, coords) # (N, N)
                
                # Mask self
                dist_matrix.fill_diagonal_(float('inf'))
                
                # Get k nearest
                k = min(self.k, num_events - 1)
                dists, neighbors = torch.topk(dist_matrix, k=k, dim=1, largest=False)
                
                # Flatten into source/target arrays
                src = torch.arange(num_events, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
                dst = neighbors.reshape(-1)
                valid_mask = dists.reshape(-1) < (self.crop_limit / self.r_distortion) # Foveal crop masking
                
                src = src[valid_mask]
                dst = dst[valid_mask]
                
                if len(src) > 0:
                    edge_idx = torch.stack([src, dst], dim=0)
                    
                    # Edge Features: [delta_x, delta_y, cortical_distance]
                    dx = coords[dst, 0] - coords[src, 0]
                    dy = coords[dst, 1] - coords[src, 1]
                    cdist = torch.sqrt(dx**2 + dy**2)
                    edge_f = torch.stack([dx, dy, cdist], dim=1)
                else:
                    edge_idx = torch.empty((2, 0), dtype=torch.long, device=device)
                    edge_f = torch.empty((0, 3), dtype=torch.float32, device=device)
            else:
                edge_idx = torch.empty((2, 0), dtype=torch.long, device=device)
                edge_f = torch.empty((0, 3), dtype=torch.float32, device=device)
            
            all_node_feats.append(nodes)
            all_edge_indices.append(edge_idx + global_node_offset)
            all_edge_feats.append(edge_f)
            
            b_idx = torch.full((num_events,), b, dtype=torch.long, device=device)
            all_batch_idx.append(b_idx)
            
            global_node_offset += num_events
            
        # Compile final sparse batched graph (Fallback if completely empty image)
        if len(all_node_feats) == 0:
            return (
                torch.empty((0, 5), device=device),
                torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty((0, 3), device=device),
                torch.empty((0,), dtype=torch.long, device=device)
            )
            
        return (
            torch.cat(all_node_feats, dim=0),
            torch.cat(all_edge_indices, dim=1),
            torch.cat(all_edge_feats, dim=0),
            torch.cat(all_batch_idx, dim=0)
        )


class PredictiveCodingErrorRouter(nn.Module):
    """
    Stage 2: Predictive_Coding_Error_Router
    Implements Friston's Free Energy Principle for routing. Uses a tiny Prior
    network to predict the Expected Node features given the current sequence belief.
    Routes only nodes that exhibit high L1 residuals (Surprise) above a dynamic
    threshold, separating them into ON (+1) and OFF (-1) processing channels.
    """
    def __init__(self, num_classes=100, node_dim=5):
        super().__init__()
        # Tiny Generative Prior: ~3.5K params
        self.prior_net = nn.Sequential(
            nn.Linear(num_classes, 32),
            nn.GELU(),
            nn.Linear(32, node_dim)
        )
        
    def forward(self, nodes, edge_idx, edge_feat, batch_idx, current_belief, routing_enabled=True, fixed_threshold=None):
        """
        nodes: (N, 5)
        edge_idx: (2, E)
        edge_feat: (E, 3)
        batch_idx: (N,)
        current_belief: (B, 100)
        routing_enabled: Controls if we apply sparsity, or route everything (Phase 1)
        """
        device = nodes.device
        
        # If the image was completely blank/uniform, return empty structures
        if nodes.size(0) == 0:
            return (
                nodes, edge_idx, edge_feat, batch_idx, # ON channel
                nodes, edge_idx, edge_feat, batch_idx, # OFF channel
                torch.zeros(current_belief.size(0), 5, device=device), # predictions
                torch.tensor(0.0, device=device) # mean error scalar
            )
            
        N = nodes.size(0)
        
        # 1. Generate Top-Down Prediction per batch element
        # (B, 5)
        expected_nodes = self.prior_net(current_belief)
        
        # Broadcast expectations to all N nodes based on their batch ID
        # (N, 5)
        node_expectations = expected_nodes[batch_idx]
        
        # 2. Compute Surprise (L1 Error)
        # (N,)
        errors = torch.norm(nodes - node_expectations, p=1, dim=1)
        
        mean_error_per_batch = torch.zeros(current_belief.size(0), device=device)
        mean_error_per_batch.scatter_add_(0, batch_idx, errors)
        
        counts = torch.bincount(batch_idx, minlength=current_belief.size(0))
        mean_error_per_batch = mean_error_per_batch / counts.clamp(min=1).float()
        
        if not routing_enabled:
            # Phase 1: Route everything
            route_mask = torch.ones(N, dtype=torch.bool, device=device)
        else:
            # Phase 2 & 3: Dynamic routing threshold: theta = mean + 0.5 * std
            # Computed uniquely per batch element
            b_means = []
            b_stds = []
            
            # Using loop for clarity on subset statistics per batch item
            route_mask = torch.zeros(N, dtype=torch.bool, device=device)
            for b in range(current_belief.size(0)):
                b_mask = (batch_idx == b)
                b_errs = errors[b_mask]
                
                if len(b_errs) == 0: continue
                
                if fixed_threshold is not None:
                    # e.g., Phase 2 asks for top 50%
                    median = torch.median(b_errs)
                    route_mask[b_mask] = (b_errs >= median)
                else:
                    # Phase 3: mu + 0.5 * sigma
                    mu = b_errs.mean()
                    if len(b_errs) > 1:
                        sigma = b_errs.std()
                    else:
                        sigma = 0.0
                        
                    theta = mu + 0.5 * sigma
                    route_mask[b_mask] = (b_errs > theta)
        
        # 3. Channel Split (ON vs OFF polarity)
        # node_feat[:, 2] is polarity. +1 for ON, -1 for OFF
        polarity = nodes[:, 2]
        
        on_mask = route_mask & (polarity > 0)
        off_mask = route_mask & (polarity < 0)
        
        # Helper function to extract subgraphs efficiently
        def extract_channel(mask):
            sub_nodes = nodes[mask]
            sub_batch = batch_idx[mask]
            
            if len(sub_nodes) == 0:
                 return sub_nodes, torch.empty((2,0), dtype=torch.long, device=device), torch.empty((0,3), device=device), sub_batch
            
            # Subgraph edge extraction: keep edge only if both source and dest are routed
            src, dst = edge_idx
            edge_mask = mask[src] & mask[dst]
            
            sub_edge_idx = edge_idx[:, edge_mask]
            sub_edge_feat = edge_feat[edge_mask]
            
            # We must re-index the edges since the node array shrank
            # Create a mapping from old global index to new local index
            remap = torch.full((N,), -1, dtype=torch.long, device=device)
            remap[mask] = torch.arange(mask.sum(), device=device)
            
            new_src = remap[sub_edge_idx[0]]
            new_dst = remap[sub_edge_idx[1]]
            new_edge_idx = torch.stack([new_src, new_dst], dim=0)
            
            return sub_nodes, new_edge_idx, sub_edge_feat, sub_batch
            
        on_subgraph = extract_channel(on_mask)
        off_subgraph = extract_channel(off_mask)
        
        return on_subgraph, off_subgraph, expected_nodes, mean_error_per_batch


class GINConv(nn.Module):
    """ Simple Graph Isomorphism Network Convolution """
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp
        self.eps = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, edge_idx):
        if x.size(0) == 0:
            return x
        src, dst = edge_idx
        # Message passing: sum features of neighbors
        out = torch.zeros_like(x)
        if src.numel() > 0:
            out.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.size(1)), x[src])
        out = (1 + self.eps) * x + out
        return self.mlp(out)


class ONChannelGINNode(nn.Module):
    """
    Stage 3: ON_Channel_Graph_Processor
    Processes ON-polarity (+1) surprise events via a 2-layer Graph Isomorphism Network.
    """
    def __init__(self, in_dim=5):
        super().__init__()
        # GIN Layer 1: [5 -> 64 -> 64]
        self.gin1 = GINConv(nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, 64)
        ))
        # GIN Layer 2: [64 -> 128 -> 128]
        self.gin2 = GINConv(nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 128)
        ))
        
        # Output Pooler
        self.out_proj = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU()
        )
        
    def forward(self, nodes, edge_idx, batch_idx, B):
        """
        nodes: (N, 5)
        edge_idx: (2, E)
        B: Batch size
        """
        device = nodes.device
        
        if nodes.size(0) == 0:
            return torch.zeros(B, 128, device=device)
            
        x = self.gin1(nodes, edge_idx)
        x = self.gin2(x, edge_idx)
        
        # Global Add Pool
        pooled = torch.zeros(B, 128, device=device)
        if nodes.size(0) > 0:
             pooled.scatter_add_(0, batch_idx.unsqueeze(1).expand(-1, 128), x)
        
        return self.out_proj(pooled)


class GINConvWithEdge(nn.Module):
    """ Graph Isomorphism Network Convolution integrating Edge Features """
    def __init__(self, in_node_dim, in_edge_dim, out_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_edge_dim, 16),
            nn.GELU(),
            nn.Linear(16, 16)
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_node_dim + 16, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
        self.eps = nn.Parameter(torch.zeros(1))
        # Projection for the ego node (which doesn't have an edge feature to itself in this step)
        self.ego_proj = nn.Linear(in_node_dim, in_node_dim + 16, bias=False)
        
    def forward(self, x, edge_idx, edge_feat):
        if x.size(0) == 0:
            return x
            
        src, dst = edge_idx
        out = torch.zeros(x.size(0), x.size(1) + 16, device=x.device)
        
        if src.numel() > 0:
            # Embed edge features
            e_emb = self.edge_mlp(edge_feat) # (E, 16)
            # Concat neighbor node features with the edge connecting them 
            msg = torch.cat([x[src], e_emb], dim=-1) # (E, in_node_dim + 16)
            out.scatter_add_(0, dst.unsqueeze(1).expand(-1, msg.size(1)), msg)
            
        # Ego node integration
        ego = self.ego_proj(x)
        out = (1 + self.eps) * ego + out
        
        return self.mlp(out)


class OFFChannelGINNode(nn.Module):
    """
    Stage 4: OFF_Channel_Graph_Processor
    Processes OFF-polarity (-1) surprise events. Critically incorporates Edge
    geometric features (delta_x, delta_y, dist) to detect oriented contrast shapes.
    """
    def __init__(self, in_node_dim=5, in_edge_dim=3):
        super().__init__()
        # GIN Layer 1 (With Edges): [5 node + 16 edge -> 64 -> 64]
        self.gin1 = GINConvWithEdge(in_node_dim, in_edge_dim, 64)
        
        # GIN Layer 2 (Standard): [64 -> 128 -> 128]
        self.gin2 = GINConv(nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 128)
        ))
        
        # Output Pooler
        self.out_proj = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU()
        )
        
    def forward(self, nodes, edge_idx, edge_feat, batch_idx, B):
        device = nodes.device
        
        if nodes.size(0) == 0:
            return torch.zeros(B, 128, device=device)
            
        x = self.gin1(nodes, edge_idx, edge_feat)
        x = self.gin2(x, edge_idx)
        
        # Global Add Pool
        pooled = torch.zeros(B, 128, device=device)
        if nodes.size(0) > 0:
             pooled.scatter_add_(0, batch_idx.unsqueeze(1).expand(-1, 128), x)
        
        return self.out_proj(pooled)


class BindingPredictionHead(nn.Module):
    """
    Stage 5: Binding_Prediction_Update_Head
    Fuses the GIN ON-channel and OFF-channel embeddings with the Prior Network's
    top-down prediction and the scalar surprise metric.
    Input dims: 128 (ON) + 128 (OFF) + 5 (Prediction) + 1 (Error Scalar) = 262.
    """
    def __init__(self, num_classes=100):
        super().__init__()
        # Input: 262-dim concatenation
        self.c1 = nn.Linear(262, 512)
        
        # Dual BatchNorm stabilization to prevent uniform-distribution collapse
        self.bn1 = nn.BatchNorm1d(512)
        self.c2 = nn.Linear(512, 256)
        
        self.bn2 = nn.BatchNorm1d(256)
        self.c3 = nn.Linear(256, 128)
        
        self.dropout = nn.Dropout(p=0.25)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, emb_ON, emb_OFF, expected_nodes, mean_error):
        """
        emb_ON: (B, 128)
        emb_OFF: (B, 128)
        expected_nodes: (B, 5) from Prior
        mean_error: (B,) Scalar tracking total L1 surprise
        """
        B = emb_ON.size(0)
        
        # Concat everything
        combined = torch.cat([emb_ON, emb_OFF, expected_nodes, mean_error.unsqueeze(1)], dim=-1) # (B, 262)
        
        out = F.silu(self.c1(combined))
        
        if out.size(0) > 1:
            out = self.bn1(out)
            
        out = F.gelu(self.c2(out))
        
        if out.size(0) > 1:
            out = self.bn2(out)
            
        out = F.gelu(self.c3(out))
        out = self.dropout(out)
        
        logits = F.log_softmax(self.classifier(out), dim=-1)
        return logits
