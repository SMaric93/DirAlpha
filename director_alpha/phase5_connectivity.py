import pandas as pd
import networkx as nx
from . import config

def analyze_connectivity(df):
    """
    Identify the largest connected set of directors and firms.
    """
    # Create a bipartite graph
    # Nodes: Directors (D_xxx), Firms (F_xxx)
    # Edges: Participation in selection
    
    # Create unique IDs for graph
    df['node_director'] = 'D_' + df['directorid'].astype(str)
    df['node_firm'] = 'F_' + df['gvkey'].astype(str)
    
    G = nx.Graph()
    
    # Add edges
    # We only need unique pairs to establish connectivity
    edges = df[['node_director', 'node_firm']].drop_duplicates().values
    G.add_edges_from(edges)
    
    print(f"Graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    # Find connected components
    components = list(nx.connected_components(G))
    print(f"Found {len(components)} connected components.")
    
    # Identify largest component
    largest_component = max(components, key=len)
    print(f"Largest component size: {len(largest_component)} nodes.")
    
    # Filter data to this component
    # Keep rows where BOTH director and firm are in the largest component
    # (If one is, the other must be, by definition of component, unless we have disconnected rows? 
    # No, if an edge exists, both nodes are in the same component.)
    
    # Create a set for fast lookup
    connected_nodes = set(largest_component)
    
    # Filter
    is_connected = df['node_director'].isin(connected_nodes) # Sufficient to check one node of the edge
    
    connected_df = df[is_connected].copy()
    
    return connected_df

def run_phase5():
    print("Starting Phase 5: Connectivity Analysis...")
    
    try:
        df = pd.read_parquet(config.ANALYSIS_HDFE_PATH)
    except FileNotFoundError:
        print("Analysis file not found.")
        return

    if df.empty:
        print("Analysis file is empty.")
        return

    # 1. Analyze Network & Restrict
    final_df = analyze_connectivity(df)
    
    print(f"Phase 5 Complete. Retained {len(final_df)} observations in the connected set.")
    
    # Save Final Dataset
    final_path = config.DATA_DIR / "director_alpha_final.parquet"
    final_df.to_parquet(final_path)
    
    return final_df

if __name__ == "__main__":
    run_phase5()
