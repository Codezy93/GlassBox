import os
import joblib
import pandas as pd
import numpy as np
import networkx as nx
from causallearn.search.ScoreBased.GES import ges
from loguru import logger

class CausalEngine:
    def __init__(self, data_path, model_path=None):
        logger.info(f"Initializing CausalEngine with data {data_path}")
        splits = joblib.load(data_path)
        self.df = splits['X_train']
        if not isinstance(self.df, pd.DataFrame):
            # If it's a numpy array, we need feature names
            feature_names = ["credit_limit", "sex", "education", "marriage", "age", 
                             "repayment_sep", "repayment_aug", "repayment_jul", "repayment_jun", "repayment_may", "repayment_apr",
                             "bill_sep", "bill_aug", "bill_jul", "bill_jun", "bill_may", "bill_apr",
                             "pay_sep", "pay_aug", "pay_jul", "pay_jun", "pay_may", "pay_apr"]
            self.df = pd.DataFrame(self.df, columns=feature_names)
            
        self.feature_names = self.df.columns.tolist()
        self.dag = None
        self.adj_matrix = None
        
        # Load or Discover DAG
        self.dag_save_path = data_path.replace("raw_splits.pkl", "causal_dag.pkl")
        if os.path.exists(self.dag_save_path):
            self.load_dag()
        else:
            self.discover_dag()

    def discover_dag(self):
        """
        Use GES (Greedy Equivalence Search) to discover the causal DAG from data.
        """
        logger.info("🔭 Discovering Causal DAG via GES (this may take a minute)…")
        # GES expects a numpy array. We limit samples for speed if needed.
        data_arr = self.df.values
        try:
            # max_p: maximum number of parents for each node
            record = ges(data_arr, score_func='local_score_BIC')
            self.adj_matrix = record['G'].graph # Graph object
            
            # Convert causal-learn graph to NetworkX for easier processing
            G = nx.DiGraph()
            G.add_nodes_from(self.feature_names)
            
            # In causal-learn, edges are represented in the adjacency matrix
            # We iterate through the graph and extract directed edges.
            matrix = record['G'].graph
            for i in range(len(self.feature_names)):
                for j in range(len(self.feature_names)):
                    if matrix[i, j] == -1 and matrix[j, i] == 1:
                        # Directed edge: j -> i
                        G.add_edge(self.feature_names[j], self.feature_names[i])
            
            self.dag = G
            self.save_dag()
            logger.success("✅ Causal DAG discovered and saved.")
        except Exception as e:
            logger.error(f"❌ Causal discovery failed: {e}")
            self.dag = nx.DiGraph()

    def save_dag(self):
        joblib.dump({"dag": self.dag, "adj": self.adj_matrix}, self.dag_save_path)

    def load_dag(self):
        data = joblib.load(self.dag_save_path)
        self.dag = data["dag"]
        self.adj_matrix = data["adj"]
        logger.info("📖 Causal DAG loaded from disk.")

    def get_graph_json(self):
        """
        Returns nodes and edges for D3.js visualization.
        """
        if self.dag is None:
            return {"nodes": [], "links": []}
            
        nodes = [{"id": name} for name in self.feature_names]
        links = [{"source": u, "target": v} for u, v in self.dag.edges()]
        
        return {"nodes": nodes, "links": links}

    def get_causal_parents(self, node_name):
        if self.dag and node_name in self.dag:
            return list(self.dag.predecessors(node_name))
        return []

    def get_causal_children(self, node_name):
        if self.dag and node_name in self.dag:
            return list(self.dag.successors(node_name))
        return []
