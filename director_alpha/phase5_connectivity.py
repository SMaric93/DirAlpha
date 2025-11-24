import pandas as pd
import networkx as nx
from . import config, log

logger = log.logger

def analyze_connectivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify the largest connected set of directors and firms.
    """
    # Create a bipartite graph
    # Nodes: Directors (D_xxx), Firms (F_xxx)
    # Edges: Participation in selection
    
    df = df.copy()
    
    # Create unique IDs for graph
    df['node_director'] = 'D_' + df['directorid'].astype(str)
    df['node_firm'] = 'F_' + df['gvkey'].astype(str)
    
    G = nx.Graph()
    
    # Add edges
    edges = df[['node_director', 'node_firm']].drop_duplicates().values
    G.add_edges_from(edges)
    
    logger.info(f"Graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    if G.number_of_nodes() == 0:
        return df
        
    # Find connected components
    components = list(nx.connected_components(G))
    logger.info(f"Found {len(components)} connected components.")
    
    if not components:
        return df
        
    # Identify largest component
    largest_component = max(components, key=len)
    logger.info(f"Largest component size: {len(largest_component)} nodes.")
    
    # Filter data to this component
    connected_nodes = set(largest_component)
    is_connected = df['node_director'].isin(connected_nodes) 
    
    connected_df = df[is_connected].copy()
    
    return connected_df

def run_phase5():
    logger.info("Starting Phase 5: Connectivity Analysis...")
    
    if not config.ANALYSIS_HDFE_PATH.exists():
        logger.error(f"Analysis file not found: {config.ANALYSIS_HDFE_PATH}")
        return

    try:
        df = pd.read_parquet(config.ANALYSIS_HDFE_PATH)
    except Exception as e:
        logger.error(f"Failed to read analysis file: {e}")
        return

    if df.empty:
        logger.warning("Analysis file is empty.")
        return

    # 1. Analyze Network & Restrict
    final_df = analyze_connectivity(df)
    
    logger.info(f"Phase 5 Complete. Retained {len(final_df)} observations in the connected set.")
    
    # Save Final Dataset
    final_path = config.DATA_DIR / "director_alpha_final.parquet"
    final_df.to_parquet(final_path)
    
    return final_df

if __name__ == "__main__":
    run_phase5()