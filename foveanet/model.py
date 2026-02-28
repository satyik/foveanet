import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import (
    RetinalContrastGraphBuilder,
    PredictiveCodingErrorRouter,
    ONChannelGINNode,
    OFFChannelGINNode,
    BindingPredictionHead
)

class FoveaNetDelta(nn.Module):
    """
    FoveaNet-Δ: Predictive Coding Error-Routed Graph Network
    Generation D. Purely event-driven sparse computational network.
    
    1) Raw image is converted to a sparse k-NN graph of contrast events.
    2) A Generative Prior network predicts expected node properties.
    3) Nodes with high L1 Error (prediction mismatch) are routed.
    4) Routed nodes are split into ON/OFF channels and processed by GIN nodes.
    5) Top-down Belief vector is updated recursively (T=2).
    """
    def __init__(self, num_classes=100):
        super().__init__()
        
        # Stage 0/1: Retinal DoG filter + kNN sparse graph
        self.encoder = RetinalContrastGraphBuilder(max_nodes=196)
        
        # Stage 2: Friston Free-Energy predictive router
        self.router = PredictiveCodingErrorRouter(num_classes=num_classes)
        
        # Stage 3/4: Polarity-separated Graph Convolutions
        self.node_A_ON = ONChannelGINNode()
        self.node_B_OFF = OFFChannelGINNode()
        
        # Stage 5: Prediction Head
        self.cortical_binding = BindingPredictionHead(num_classes=num_classes)
        
        # Training Curriculum State
        self.current_phase = 1

    def set_training_phase(self, phase):
        self.current_phase = phase

    def forward_step(self, nodes, edge_idx, edge_feat, batch_idx, current_belief):
        """ Executes a single recurrent pass through the Delta architecture """
        B = current_belief.size(0)
        
        # 1. Predictive Coding Error Routing
        routing_enabled = (self.current_phase > 1)
        fixed_threshold = None
        if self.current_phase == 2:
            fixed_threshold = 'median' # Top 50%
            
        on_graph, off_graph, expected_nodes, mean_error = self.router(
            nodes, edge_idx, edge_feat, batch_idx, current_belief, 
            routing_enabled=routing_enabled, fixed_threshold=fixed_threshold
        )
        
        # 2. Graph Isomorphism Network Processing
        # Node A handles ON-channel (Luminance increases)
        emb_ON = self.node_A_ON(on_graph[0], on_graph[1], on_graph[3], B)
        
        # Node B handles OFF-channel (Luminance decreases) + shape directionality
        emb_OFF = self.node_B_OFF(off_graph[0], off_graph[1], off_graph[2], off_graph[3], B)
        
        # 3. Cortical Binding Prediction
        logits = self.cortical_binding(emb_ON, emb_OFF, expected_nodes, mean_error)
        return logits, expected_nodes

    def forward(self, image):
        B = image.size(0)
        device = image.device
        
        # Initial bottom-up graph extraction (Runs once)
        nodes, edge_idx, edge_feat, batch_idx = self.encoder(image)
        
        # Prior Belief Initialization (Uniform)
        current_belief = torch.ones(B, 100, device=device) / 100.0
        
        # T=2 Inference Steps
        T_steps = 2 if self.current_phase == 3 else 1
        
        logits_history = []
        
        for t in range(T_steps):
            # Forward execution
            logits, expected_nodes = self.forward_step(nodes, edge_idx, edge_feat, batch_idx, current_belief)
            logits_history.append(logits)
            
            # Belief EMA update (tau = 0.7) for next timestep
            if t < T_steps - 1:
                # Early Exit logic checking maximum probability confidence
                max_probs = torch.max(torch.exp(logits), dim=1)[0]
                if (max_probs > 0.94).all():
                    break
                    
                new_belief = torch.exp(logits).detach() # Probability space
                current_belief = (0.7 * current_belief) + (0.3 * new_belief)
                
        # Confidence-weighted ensembling (if multi-step ran)
        if len(logits_history) == 1:
            final_logits = logits_history[0]
        else:
            final_logits = torch.stack(logits_history).mean(dim=0)
            
        # Optional Context Return: In train.py we need the expected_nodes to calculate
        # the auxiliary `prior_reconstruction_loss`. 
        if self.training:
            # Reconstruct loss vs unmasked mean of graph nodes
            b_means = []
            for b in range(B):
                b_mask = (batch_idx == b)
                if b_mask.any():
                    b_means.append(nodes[b_mask].mean(dim=0))
                else:
                    b_means.append(torch.zeros(5, device=device))
            # Average unmasked node parameters that the Generative Prior *should* have predicted
            actual_node_means = torch.stack(b_means) # (B, 5)
            
            return final_logits, expected_nodes, actual_node_means
            
        return final_logits
