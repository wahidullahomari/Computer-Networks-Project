"""
===================================================================================
                    PARTICLE SWARM OPTIMIZATION FOR QoS ROUTING
===================================================================================
Bu modül, Parçacık Sürü Optimizasyonu (PSO) kullanarak çoklu QoS kısıtlı rota
optimizasyonu gerçekleştirir. Sürü zekası prensipleriyle global optimuma yakınsama sağlar.

Temel Özellikler:
------------------
1. Bant genişliği kısıtı altında uygun yolları bulma
2. Ağırlıklı fitness fonksiyonu (gecikme, güvenilirlik, kaynak)
3. Hız ve konum güncelleme ile arama uzayını keşfetme
4. Personal best (pbest) ve Global best (gbest) takibi
5. Inertia weight (w), cognitive (c1), social (c2) parametreleriyle ayarlama

Algoritma:
----------
1. Rastgele parçacık sürüsü oluştur (her parçacık = düğüm öncelikleri)
2. Her parçacık için öncelik sırasına göre yol üret
3. Fitness değerini hesapla (düşük = iyi)
4. Personal ve Global best'leri güncelle
5. Hız ve konum vektörlerini güncelle (PSO formülü)
6. Iterasyon sayısı kadar tekrarla

Kullanım:
---------
    pso = QoS_PSO_Solver(G=network_graph)
    path, metrics = pso.solve_demand(src=0, dst=10, demand_value=200, weights=(0.4, 0.3, 0.3))

Author: [Proje Ekibi]
Date: Aralık 2024
"""

import numpy as np    # Sayısal hesaplamalar ve parçacık vektörleri
import networkx as nx # Ağ grafiği ve yol bulma
import math           # Logaritmik güvenilirlik hesaplamaları

class Particle:
    """
    PSO Parçacığı (Sürüdeki bir aday çözüm)
    
    Her parçacık, düğümlere atanmış öncelik ağırlıklarını temsil eder.
    Bu ağırlıklar, yol üretiminde düğüm seçim sırasını belirler.
    
    Attributes:
        position (np.array): Düğüm öncelikleri [0, 1] arası (size kadar)
        velocity (np.array): Hız vektörü (arama yönü ve büyüklüğü)
        pbest_pos (np.array): Kişisel en iyi pozisyon
        pbest_score (float): Kişisel en iyi fitness değeri
    """
    def __init__(self, size):
        """
        Parçacığı rastgele başlat.
        
        Args:
            size (int): Ağdaki toplam düğüm sayısı
        """
        # Her düğüm için rastgele öncelik ağırlığı [0, 1]
        self.position = np.random.rand(size)
        # Başlangıç hızı küçük rastgele değerler
        self.velocity = np.random.uniform(-0.1, 0.1, size)
        # İlk pozisyon aynı zamanda personal best
        self.pbest_pos = self.position.copy()
        self.pbest_score = float('inf')  # Minimum aranıyor

    def update(self, gbest_pos, w=0.7, c1=1.5, c2=2.0):
        """
        PSO hız ve konum güncelleme formülü.
        
        v(t+1) = w*v(t) + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        x(t+1) = x(t) + v(t+1)
        
        Args:
            gbest_pos (np.array): Global en iyi pozisyon (sürünün lideri)
            w (float): Inertia weight (관성 ağırlığı) - mevcut hızın etkisi
            c1 (float): Cognitive coefficient - kişisel deneyimin etkisi
            c2 (float): Social coefficient - sosyal bilginin etkisi
        """
        r1, r2 = np.random.rand(), np.random.rand()  # Stokastiklik
        
        # Bilişsel bileşen: Kendi en iyi deneyimine doğru çekim
        cognitive = c1 * r1 * (self.pbest_pos - self.position)
        
        # Sosyal bileşen: Sürünün en iyisine doğru çekim
        social = c2 * r2 * (gbest_pos - self.position)
        
        # Hız güncelleme (관성 + bilişsel + sosyal)
        self.velocity = w * self.velocity + cognitive + social
        
        # Konum güncelleme ve sınırlandırma [0.001, 1.0]
        self.position = np.clip(self.position + self.velocity, 0.001, 1.0)

class QoS_PSO_Solver:
    """
    Parçacık Sürü Optimizasyonu Tabanlı QoS Rota Çözücü
    
    Attributes:
        G (nx.Graph): Ağ topolojisi grafiği
        nodes_list (list): Düğüm ID'leri listesi (parçacık indeks eşleştirmesi için)
    """
    def __init__(self, G=None):
        """
        PSO çözücüyü başlat.
        
        Args:
            G (nx.Graph, optional): NetworkX graf. None ise boş graf oluşturulur.
        """
        # Arayüz ile entegrasyon için TopologyManager'dan gelen G grafını kullan
        self.G = G if G is not None else nx.Graph()
        self.nodes_list = list(self.G.nodes())  # Düğüm listesi

    def calculate_fitness(self, path, weights):
        """
        Verilen yolun fitness değerini hesaplar (düşük = iyi).
        
        Weighted Sum Method ile çoklu QoS kriterlerini tek değere dönüştürür:
        fitness = w_delay*Delay + w_rel*ReliabilityCost + w_res*ResourceCost
        
        Args:
            path (list): Düğüm listesi [src, ..., dst]
            weights (tuple): (w_delay, w_reliability, w_resource)
        
        Returns:
            dict: {
                'score': Toplam ağırlıklı maliyet,
                'delay': Toplam gecikme (ms),
                'reliability': Yol güvenilirliği (0-1),
                'resource': Toplam kaynak maliyeti
            }
        """
        w_delay, w_rel, w_res = weights
        
        # Metrik değişkenleri
        total_delay = 0.0         # Toplam gecikme (link + node)
        reliability_cost = 0.0    # Logaritmik güvenilirlik maliyeti
        resource_cost = 0.0       # Kaynak kullanım maliyeti
        total_reliability = 1.0   # Çarpımsal güvenilirlik

        # Yol boyunca kenarları gez
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge = self.G[u][v]
            
            # 1. Gecikme: Link Gecikmesi + Ara Düğüm İşlem Süresi
            total_delay += edge.get('link_delay', 0)
            if i > 0:  # Sadece ara düğümler (kaynak ve hedef hariç)
                total_delay += self.G.nodes[u].get('proc_delay', 0)
            
            # 2. Güvenilirlik: Çarpımsal metrik (tüm bileşenlerin güvenilirliği)
            # Log toplamı kullanarak hesaplama kolaylığı (PDF formülü)
            link_rel = edge.get('reliability', 1.0)
            node_rel = self.G.nodes[v].get('reliability', 1.0)
            total_reliability *= (link_rel * node_rel)
            
            reliability_cost += -math.log(max(link_rel, 1e-6))  # Sıfır log engelleme
            reliability_cost += -math.log(max(node_rel, 1e-6))

            # 3. Kaynak Kullanımı: Bant genişliği ile ters orantılı
            # Yüksek bant → düşük maliyet
            bw = max(edge.get('bandwidth', 1), 1e-6)
            resource_cost += (1000.0 / bw)

        # Ağırlıklı Toplam Maliyet (Weighted Sum Method)
        score = (w_delay * total_delay) + \
                (w_rel * reliability_cost * 10) + \
                (w_res * resource_cost)
        
        return {
            'score': score,                  # Minimize edilecek değer
            'delay': total_delay,            # ms cinsinden
            'reliability': total_reliability, # 0-1 arası
            'resource': resource_cost        # Kaynak maliyeti
        }

    def solve_demand(self, src, dst, demand_value, weights=(0.4, 0.3, 0.3)):
        if self.G is None or len(self.G.edges()) == 0:
            return None, None

        # Kapasite Kısıtı Filtreleme: Demand <= Capacity 
        available_edges = [(u, v, d) for u, v, d in self.G.edges(data=True) 
                          if d.get('bandwidth', 0) >= demand_value]
        
        temp_G = nx.Graph()
        temp_G.add_nodes_from(self.G.nodes(data=True))
        temp_G.add_edges_from(available_edges)

        # Fiziksel yol kontrolü
        if not nx.has_path(temp_G, src, dst):
            return None, None

        # PSO Parametreleri 
        swarm_size = 30
        iterations = 25
        num_nodes = len(self.nodes_list)
        node_to_idx = {node: i for i, node in enumerate(self.nodes_list)}
        
        swarm = [Particle(num_nodes) for _ in range(swarm_size)]
        gbest_pos = np.random.rand(num_nodes)
        gbest_path, gbest_metrics, gbest_score = None, None, float('inf')

        for _ in range(iterations):
            for particle in swarm:
                # Kenar ağırlıklarını parçacığın konumuna (düğüm önceliğine) göre ata
                for u, v in temp_G.edges():
                    # v düğümünün konum değeri kenarın pso_ağırlığı olur
                    temp_G[u][v]['pso_w'] = particle.position[node_to_idx[v]]
                
                try:
                    # Dijkstra ile parçacığın önceliklerine göre bir yol bul
                    path = nx.shortest_path(temp_G, src, dst, weight='pso_w')
                    m = self.calculate_fitness(path, weights)
                    
                    # Pbest (Kişisel en iyi) güncelleme
                    if m['score'] < particle.pbest_score:
                        particle.pbest_score = m['score']
                        particle.pbest_pos = particle.position.copy()
                        
                    # Gbest (Sürü en iyisi) güncelleme 
                    if m['score'] < gbest_score:
                        gbest_score = m['score']
                        gbest_path = path
                        gbest_metrics = m
                        gbest_pos = particle.position.copy()
                except (nx.NetworkXNoPath, KeyError):
                    continue
            
            # Sürüyü hareket ettir (Hız ve Konum Güncelleme)
            for particle in swarm:
                particle.update(gbest_pos)

        return gbest_path, gbest_metrics