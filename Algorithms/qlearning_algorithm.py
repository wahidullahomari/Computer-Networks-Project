"""
Q-LEARNING NETWORK ROUTING ALGORITHM
=====================================
Arayüze entegre edilebilir, modüler Q-Learning yönlendirme algoritması.

Kullanım:
    from qlearning_algorithm import QLearningRouter
    
    router = QLearningRouter()
    router.load_network_from_csv('node_file.csv', 'edge_file.csv')
    
    result = router.calculate_path(
        source=8, 
        target=44, 
        w_delay=0.4, 
        w_reliability=0.4, 
        w_resource=0.2,
        bandwidth_demand=200
    )
"""

import networkx as nx
import pandas as pd
import numpy as np
import random
import math
from collections import defaultdict


class QLearningRouter:
    """
    Q-Learning tabanlı ağ yönlendirme algoritması.
    Mevcut arayüz yapısına uyumlu, entegre edilebilir tasarım.
    """
    
    def __init__(self, alpha=0.1, gamma=0.9, epsilon_start=0.9, epsilon_end=0.05):
        """
        Q-Learning Router başlatma.
        
        Args:
            alpha (float): Öğrenme oranı (0-1)
            gamma (float): İskonto faktörü (0-1)
            epsilon_start (float): Başlangıç keşif oranı
            epsilon_end (float): Minimum keşif oranı
        """
        self.G = None
        self.node_data = {}
        self.edge_data = {}
        
        # Q-Learning parametreleri
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon_start
        
        # Q-Table
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        print("[Q-Learning] Router hazır.")
    
    def load_network_from_csv(self, node_file, edge_file):
        """
        CSV dosyalarından ağ verilerini yükler.
        
        Args:
            node_file (str): Düğüm CSV dosyası
            edge_file (str): Bağlantı CSV dosyası
        
        Returns:
            bool: Başarı durumu
        """
        try:
            self.G = nx.Graph()
            
            # Node dosyasını yükle
            df_nodes = pd.read_csv(node_file, sep=';', decimal=',', encoding='utf-8-sig')
            
            for _, row in df_nodes.iterrows():
                node_id = int(row['node_id'])
                proc_delay = float(str(row['s_ms']).replace(',', '.'))
                reliability = float(str(row['r_node']).replace(',', '.'))
                
                self.G.add_node(node_id)
                self.node_data[node_id] = {
                    'proc_delay': proc_delay,
                    'reliability': reliability
                }
            
            # Edge dosyasını yükle
            df_edges = pd.read_csv(edge_file, sep=';', decimal=',', encoding='utf-8-sig')
            
            for _, row in df_edges.iterrows():
                src = int(row['src'])
                dst = int(row['dst'])
                capacity = float(str(row['capacity_mbps']).replace(',', '.'))
                delay = float(str(row['delay_ms']).replace(',', '.'))
                reliability = float(str(row['r_link']).replace(',', '.'))
                
                self.G.add_edge(src, dst)
                self.edge_data[(src, dst)] = {
                    'bandwidth': capacity,
                    'link_delay': delay,
                    'reliability': reliability
                }
                # Çift yönlü
                self.edge_data[(dst, src)] = {
                    'bandwidth': capacity,
                    'link_delay': delay,
                    'reliability': reliability
                }
            
            print(f"[Q-Learning] Ag yuklendi: {len(self.node_data)} dugum, {len(self.edge_data)} baglanti")
            return True
            
        except Exception as e:
            print(f"[Q-Learning] Yukleme hatasi: {e}")
            return False
    
    def load_network_from_graph(self, G, node_data=None, edge_data=None):
        """
        Mevcut bir NetworkX grafından ağı yükler.
        (Arayüzde hazır graf varsa bu metod kullanılabilir)
        
        Args:
            G (nx.Graph): NetworkX graf objesi
            node_data (dict): Düğüm özellikleri (opsiyonel)
            edge_data (dict): Bağlantı özellikleri (opsiyonel)
        """
        self.G = G
        
        if node_data:
            self.node_data = node_data
        else:
            # Graftan bilgileri çıkar
            for node in G.nodes():
                self.node_data[node] = {
                    'proc_delay': G.nodes[node].get('proc_delay', 1.0),
                    'reliability': G.nodes[node].get('reliability', 0.99)
                }
        
        if edge_data:
            self.edge_data = edge_data
        else:
            # Graftan bilgileri çıkar
            for u, v in G.edges():
                self.edge_data[(u, v)] = {
                    'bandwidth': G.edges[u, v].get('bandwidth', 500),
                    'link_delay': G.edges[u, v].get('link_delay', 5),
                    'reliability': G.edges[u, v].get('reliability', 0.98)
                }
        
        print(f"[Q-Learning] Graf yuklendi: {len(self.node_data)} dugum")
    
    def get_neighbors(self, node):
        """Komşu düğümleri döndürür."""
        if self.G is None:
            return []
        return list(self.G.neighbors(node))
    
    def get_q_value(self, state, action):
        """Q-değerini döndürür."""
        return self.q_table[state][action]
    
    def get_max_q_value(self, state):
        """Bir durumdan yapılabilecek en iyi eylemin Q-değerini döndürür."""
        neighbors = self.get_neighbors(state)
        if not neighbors:
            return 0.0
        return max([self.get_q_value(state, n) for n in neighbors])
    
    def choose_action(self, state, destination, visited, epsilon):
        """
        Epsilon-greedy politikası ile eylem seçer.
        
        Args:
            state (int): Mevcut düğüm
            destination (int): Hedef düğüm
            visited (set): Ziyaret edilen düğümler
            epsilon (float): Keşif oranı
        
        Returns:
            int or None: Seçilen komşu düğüm
        """
        neighbors = self.get_neighbors(state)
        unvisited = [n for n in neighbors if n not in visited]
        
        if not unvisited:
            return None
        
        # Hedefe direkt bağlantı varsa öncelik ver
        if destination in unvisited:
            if random.random() > epsilon / 2:
                return destination
        
        # Epsilon-greedy
        if random.random() < epsilon:
            return random.choice(unvisited)
        else:
            # En yüksek Q-değerine sahip komşu
            q_values = [(n, self.get_q_value(state, n)) for n in unvisited]
            return max(q_values, key=lambda x: x[1])[0]
    
    def calculate_path_cost(self, path, w_delay, w_rel, w_resource):
        """
        Bir yolun toplam maliyetini hesaplar.
        
        Args:
            path (list): Düğüm yolu
            w_delay (float): Gecikme ağırlığı
            w_rel (float): Güvenilirlik ağırlığı
            w_resource (float): Kaynak ağırlığı
        
        Returns:
            float: Toplam maliyet
        """
        if len(path) < 2:
            return float('inf')
        
        total_delay = 0.0
        reliability_cost = 0.0
        resource_cost = 0.0
        
        # Bağlantı metrikleri
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_info = self.edge_data.get((u, v))
            
            if edge_info:
                total_delay += edge_info['link_delay']
                
                if edge_info['reliability'] > 0:
                    reliability_cost += -math.log(edge_info['reliability'])
                else:
                    reliability_cost += 100
                
                if edge_info['bandwidth'] > 0:
                    resource_cost += 1000.0 / edge_info['bandwidth']
                else:
                    resource_cost += 1000.0
        
        # Düğüm metrikleri (kaynak ve hedef hariç)
        for i in range(1, len(path) - 1):
            node_info = self.node_data.get(path[i])
            if node_info:
                total_delay += node_info['proc_delay']
                
                if node_info['reliability'] > 0:
                    reliability_cost += -math.log(node_info['reliability'])
                else:
                    reliability_cost += 100
        
        # Başlangıç ve bitiş düğümlerinin güvenilirliği
        for node in [path[0], path[-1]]:
            node_info = self.node_data.get(node)
            if node_info and node_info['reliability'] > 0:
                reliability_cost += -math.log(node_info['reliability'])
        
        # Ağırlıklı toplam
        total_cost = (w_delay * total_delay + 
                     w_rel * reliability_cost * 100 +  # Scale factor
                     w_resource * resource_cost)
        
        return total_cost
    
    def calculate_reward(self, path, destination, w_delay, w_rel, w_resource, 
                        bandwidth_demand, reached_destination):
        """
        Episode sonundaki ödülü hesaplar.
        
        Args:
            path (list): Bulunan yol
            destination (int): Hedef düğüm
            w_delay (float): Gecikme ağırlığı
            w_rel (float): Güvenilirlik ağırlığı
            w_resource (float): Kaynak ağırlığı
            bandwidth_demand (float): Bant genişliği talebi
            reached_destination (bool): Hedefe ulaşıldı mı?
        
        Returns:
            float: Ödül değeri
        """
        if not reached_destination or path[-1] != destination:
            return -1000.0
        
        # Bant genişliği kontrolü
        for i in range(len(path) - 1):
            edge_info = self.edge_data.get((path[i], path[i+1]))
            if edge_info and edge_info['bandwidth'] < bandwidth_demand:
                return -500.0
        
        # Yol maliyeti
        total_cost = self.calculate_path_cost(path, w_delay, w_rel, w_resource)
        
        # Ödül fonksiyonu
        reward = 1000.0 / (1.0 + total_cost)
        
        # Yol uzunluğu cezası
        length_penalty = len(path) * 0.1
        reward -= length_penalty
        
        return reward
    
    def train_episode(self, source, destination, w_delay, w_rel, w_resource, 
                     bandwidth_demand, max_steps=100):
        """
        Tek bir eğitim episode'u çalıştırır.
        
        Args:
            source (int): Kaynak düğüm
            destination (int): Hedef düğüm
            w_delay (float): Gecikme ağırlığı
            w_rel (float): Güvenilirlik ağırlığı
            w_resource (float): Kaynak ağırlığı
            bandwidth_demand (float): Bant genişliği talebi
            max_steps (int): Maksimum adım sayısı
        
        Returns:
            tuple: (path, reward, reached_destination)
        """
        current_state = source
        path = [source]
        visited = {source}
        reached_destination = False
        
        for step in range(max_steps):
            if current_state == destination:
                reached_destination = True
                break
            
            action = self.choose_action(current_state, destination, visited, self.epsilon)
            
            if action is None:
                break
            
            next_state = action
            path.append(next_state)
            visited.add(next_state)
            
            # Q-değer güncelleme
            step_reward = -0.1
            old_q = self.get_q_value(current_state, action)
            
            if next_state == destination:
                final_reward = self.calculate_reward(path, destination, w_delay, w_rel, 
                                                     w_resource, bandwidth_demand, True)
                new_q = old_q + self.alpha * (final_reward - old_q)
            else:
                max_next_q = self.get_max_q_value(next_state)
                new_q = old_q + self.alpha * (step_reward + self.gamma * max_next_q - old_q)
            
            self.q_table[current_state][action] = new_q
            current_state = next_state
        
        total_reward = self.calculate_reward(path, destination, w_delay, w_rel, 
                                            w_resource, bandwidth_demand, reached_destination)
        
        return path, total_reward, reached_destination
    
    def calculate_path(self, source, target, w_delay, w_reliability, w_resource, 
                      bandwidth_demand=100, num_episodes=500, verbose=False):
        """
        Q-Learning ile optimal yolu hesaplar.
        ARAYÜZ ENTEGRASYONU İÇİN ANA METOD!
        
        Args:
            source (int): Kaynak düğüm
            target (int): Hedef düğüm
            w_delay (float): Gecikme ağırlığı (0-1)
            w_reliability (float): Güvenilirlik ağırlığı (0-1)
            w_resource (float): Kaynak ağırlığı (0-1)
            bandwidth_demand (float): Bant genişliği talebi (Mbps)
            num_episodes (int): Eğitim episode sayısı
            verbose (bool): Detaylı çıktı
        
        Returns:
            dict: {
                'path': [node_list],
                'total_delay': float,
                'final_reliability': float (yüzde),
                'resource_cost': float,
                'success': bool
            }
            veya None (hata durumunda)
        """
        if self.G is None:
            print("[Q-Learning] Hata: Ag yuklenmemis!")
            return None
        
        if source not in self.G.nodes() or target not in self.G.nodes():
            print(f"[Q-Learning] Hata: Gecersiz dugum ({source} veya {target})")
            return None
        
        if verbose:
            print(f"\n[Q-Learning] Egitim baslatiliyor...")
            print(f"  Kaynak: {source} -> Hedef: {target}")
            print(f"  Bant Genisligi: {bandwidth_demand} Mbps")
            print(f"  Episode: {num_episodes}")
        
        # Q-table'ı sıfırla (yeni hesaplama için)
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        best_path = []
        best_reward = float('-inf')
        successful_episodes = 0
        
        # Epsilon decay
        epsilon_decay = (self.epsilon_start - self.epsilon_end) / num_episodes
        
        for episode in range(num_episodes):
            self.epsilon = max(self.epsilon_end, self.epsilon_start - episode * epsilon_decay)
            
            path, reward, reached = self.train_episode(
                source, target, w_delay, w_reliability, w_resource, 
                bandwidth_demand
            )
            
            if reached:
                successful_episodes += 1
                if reward > best_reward:
                    best_reward = reward
                    best_path = path.copy()
            
            # İlerleme
            if verbose and (episode + 1) % (num_episodes // 5) == 0:
                success_rate = successful_episodes / (episode + 1) * 100
                print(f"  Episode {episode + 1}/{num_episodes} | Basari: {success_rate:.1f}%")
        
        if not best_path:
            if verbose:
                print("[Q-Learning] HATA: Uygun yol bulunamadi!")
            return None
        
        # Metrikleri hesapla (arayüz formatında)
        total_delay = 0.0
        reliability_log = 0.0
        resource_cost = 0.0
        
        # Bağlantı metrikleri
        for i in range(len(best_path) - 1):
            u, v = best_path[i], best_path[i+1]
            edge_info = self.edge_data.get((u, v))
            
            if edge_info:
                total_delay += edge_info['link_delay']
                reliability_log += -math.log(edge_info['reliability'])
                resource_cost += 1000.0 / edge_info['bandwidth']
        
        # Düğüm gecikmeleri (ara düğümler)
        for i in range(1, len(best_path) - 1):
            node_info = self.node_data.get(best_path[i])
            if node_info:
                total_delay += node_info['proc_delay']
        
        # Tüm düğümlerin güvenilirliği
        for node in best_path:
            node_info = self.node_data.get(node)
            if node_info:
                reliability_log += -math.log(node_info['reliability'])
        
        # Güvenilirliği yüzdeye çevir
        final_reliability = math.exp(-reliability_log) * 100
        
        if verbose:
            print(f"\n[Q-Learning] BASARILI!")
            print(f"  Yol: {' -> '.join(map(str, best_path))}")
            print(f"  Uzunluk: {len(best_path)} dugum")
            print(f"  Gecikme: {total_delay:.2f} ms")
            print(f"  Guvenilirlik: %{final_reliability:.4f}")
            print(f"  Kaynak Maliyeti: {resource_cost:.2f}")
        
        # Arayüzün beklediği formatta döndür
        return {
            'path': best_path,
            'total_delay': total_delay,
            'final_reliability': final_reliability,
            'resource_cost': resource_cost,
            'success': True
        }
    
    def reset_q_table(self):
        """Q-table'ı sıfırlar."""
        self.q_table = defaultdict(lambda: defaultdict(float))
        print("[Q-Learning] Q-table sifirlandi")


# =============================================================================
# ÖRNEK KULLANIM (Test için)
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*20 + "Q-LEARNING TEST")
    print("="*70 + "\n")
    
    # Router oluştur
    router = QLearningRouter(alpha=0.1, gamma=0.9)
    
    # CSV'den yükle
    success = router.load_network_from_csv(
        'BSM307_317_Guz2025_TermProject_NodeData.csv',
        'BSM307_317_Guz2025_TermProject_EdgeData.csv'
    )
    
    if not success:
        print("CSV dosyalari yuklenemedi!")
        exit(1)
    
    # Test senaryoları
    test_cases = [
        (8, 44, 200),
        (53, 19, 57),
        (24, 221, 172)
    ]
    
    # Ağırlıklar
    weights = {
        'delay': 0.4,
        'reliability': 0.4,
        'resource': 0.2
    }
    
    print("Test Senaryolari:")
    for i, (s, d, bw) in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}: S={s} -> D={d}, BW={bw} Mbps")
        print('='*70)
        
        result = router.calculate_path(
            source=s,
            target=d,
            w_delay=weights['delay'],
            w_reliability=weights['reliability'],
            w_resource=weights['resource'],
            bandwidth_demand=bw,
            num_episodes=300,
            verbose=True
        )
        
        if result:
            print(f"\nSONUC:")
            print(f"  Yol: {result['path']}")
            print(f"  Gecikme: {result['total_delay']:.2f} ms")
            print(f"  Guvenilirlik: %{result['final_reliability']:.4f}")
            print(f"  Kaynak: {result['resource_cost']:.2f}")
        else:
            print("\nYOL BULUNAMADI!")
    
    print("\n" + "="*70)
    print("Test tamamlandi!")
    print("="*70 + "\n")
