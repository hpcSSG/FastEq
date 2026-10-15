"""Composed CPU message passing, with a materialized edge-message tensor."""
from .cwtp import fast_cwtp
from .scatter import fast_scatter_sum

def fast_mptp(node_features, edge_attrs, weight, sender, receiver, metadata, num_nodes=None):
    """uvu tensor product followed by scatter; NOT a fused MPTP kernel."""
    messages = fast_cwtp(node_features[sender], edge_attrs, weight, metadata)
    return fast_scatter_sum(messages, receiver, node_features.shape[0] if num_nodes is None else num_nodes)
