import pandas as pd
import networkx as nx
import random
import os

class NetworkManager:
    def __init__(self):
        self.graph = None
        self.demands = []

    def load_from_csv(self, node_file, edge_file, demand_file=None):
        """
        Hocanın verdiği CSV dosyalarını (noktalı virgül ve virgül ondalık formatlı) okur.
        """
        print(f"--- MOD: Dosyadan Yükleme ---")
        self.graph = nx.Graph()

        # 1. Düğümleri (Nodes) Yükle
        # Beklenen Sütunlar: node_id, s_ms, r_node
        try:
            df_nodes = pd.read_csv(node_file, sep=';', dtype=str) # String okuyoruz ki virgülleri düzeltelim
            for _, row in df_nodes.iterrows():
                node_id = int(row['node_id'])
                # Virgülleri noktaya çevirip float yapıyoruz
                s_ms = float(row['s_ms'].replace(',', '.'))
                r_node = float(row['r_node'].replace(',', '.'))
                
                self.graph.add_node(node_id, proc_delay=s_ms, reliability=r_node)
                
        except Exception as e:
            print(f"HATA (Node Dosyası): {e}")
            return None, None

        # 2. Kenarları (Edges) Yükle
        # Beklenen Sütunlar: src, dst, capacity_mbps, delay_ms, r_link
        try:
            df_edges = pd.read_csv(edge_file, sep=';', dtype=str)
            for _, row in df_edges.iterrows():
                src = int(row['src'])
                dst = int(row['dst'])
                cap = int(row['capacity_mbps'])
                delay = int(row['delay_ms'])
                # Güvenilirlik ondalık sayı (0,99 gibi)
                r_link = float(row['r_link'].replace(',', '.'))
                
                self.graph.add_edge(src, dst, bandwidth=cap, link_delay=delay, reliability=r_link)
                
        except Exception as e:
            print(f"HATA (Edge Dosyası): {e}")
            return None, None

        # 3. Talepleri (Demands) Yükle
        # Beklenen Sütunlar: src, dst, demand_mbps
        if demand_file:
            try:
                df_demands = pd.read_csv(demand_file, sep=';')
                self.demands = []
                for i, row in df_demands.iterrows():
                    self.demands.append({
                        'id': i,
                        'src': int(row['src']),
                        'dst': int(row['dst']),
                        'bandwidth_needed': int(row['demand_mbps'])
                    })
            except Exception as e:
                print(f"HATA (Demand Dosyası): {e}")

        print(f"Başarılı! Node: {len(self.graph.nodes)}, Edge: {len(self.graph.edges)}, Talep: {len(self.demands)}")
        return self.graph, self.demands

    def generate_random(self, num_nodes=50, connection_prob=0.15, num_demands=10):
        """
        Rastgele ağ üretir ancak veri yapısını (attributes) hocanın dosyalarıyla aynı tutar.
        """
        print(f"--- MOD: Rastgele Üretim (N={num_nodes}) ---")
        
        # 1. Topoloji
        while True:
            self.graph = nx.erdos_renyi_graph(n=num_nodes, p=connection_prob)
            if nx.is_connected(self.graph):
                break
        
        # 2. Düğüm Özellikleri (Node Attributes)
        for node in self.graph.nodes():
            self.graph.nodes[node]['proc_delay'] = round(random.uniform(0.5, 2.0), 2)
            self.graph.nodes[node]['reliability'] = round(random.uniform(0.90, 0.99), 3)

        # 3. Kenar Özellikleri (Edge Attributes)
        for (u, v) in self.graph.edges():
            self.graph.edges[u, v]['bandwidth'] = random.randint(100, 1000) # Mbps
            self.graph.edges[u, v]['link_delay'] = random.randint(5, 50)         # ms
            self.graph.edges[u, v]['reliability'] = round(random.uniform(0.90, 0.99), 3)

        # 4. Rastgele Talepler
        nodes = list(self.graph.nodes())
        self.demands = []
        for i in range(num_demands):
            try:
                src, dst = random.sample(nodes, 2)
                self.demands.append({
                    'id': i,
                    'src': src,
                    'dst': dst,
                    'bandwidth_needed': random.randint(50, 200)
                })
            except ValueError:
                pass # Yeterli node yoksa atla

        print(f"Rastgele ağ hazır. Node: {len(self.graph.nodes)}, Talep: {len(self.demands)}")
        return self.graph, self.demands

# Test Bloğu (Kodu çalıştırıp denemek için)
if __name__ == "__main__":
    nm = NetworkManager()
    
    # 1. TEST: Rastgele Mod
    G_rnd, d_rnd = nm.generate_random(num_nodes=20)
    
    # 2. TEST: Dosya Modu (Dosyaların proje ana dizininde 'data' klasöründe olduğunu varsayıyoruz)
    # G_file, d_file = nm.load_from_csv(
    #     'data/BSM307_317_Guz2025_TermProject_NodeData.csv',
    #     'data/BSM307_317_Guz2025_TermProject_EdgeData.csv',
    #     'data/BSM307_317_Guz2025_TermProject_DemandData.csv'
    # )