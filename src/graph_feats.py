# src/graph_feats.py


from __future__ import annotations
import pandas as pd
import networkx as nx

class GraphState:
    def __init__(self, deg, comp_size):
        self.deg = deg
        self.comp_size = comp_size

def build_graph(train: pd.DataFrame) -> GraphState:
    card = 'card_' + train['card_number'].astype(str)
    dev  = 'dev_'  + train['device_fingerprint'].astype(str)
    ip   = 'ip_'   + train['ip_address'].astype(str)
    G = nx.Graph()
    G.add_edges_from(zip(card, dev))
    G.add_edges_from(zip(card, ip))
    deg = dict(G.degree())
    comp_id = {}
    for i, comp in enumerate(nx.connected_components(G)):
        for n in comp: comp_id[n] = i
    comp_size = pd.Series(comp_id).value_counts().to_dict()
    return GraphState(deg, comp_size)

def graph_features(df: pd.DataFrame, state: GraphState) -> pd.DataFrame:
    card = 'card_' + df['card_number'].astype(str)
    dev  = 'dev_'  + df['device_fingerprint'].astype(str)
    ip   = 'ip_'   + df['ip_address'].astype(str)
    out = pd.DataFrame(index=df.index)
    out['deg_card'] = card.map(state.deg).fillna(0).astype(float)
    out['deg_device'] = dev.map(state.deg).fillna(0).astype(float)
    out['deg_ip'] = ip.map(state.deg).fillna(0).astype(float)
    out['comp_size_card'] = card.map(lambda x: state.comp_size.get(x, 1)).astype(float)
    out['comp_size_device'] = dev.map(lambda x: state.comp_size.get(x, 1)).astype(float)
    out['comp_size_ip'] = ip.map(lambda x: state.comp_size.get(x, 1)).astype(float)
    return out
