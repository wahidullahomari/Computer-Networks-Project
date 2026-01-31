"""
===================================================================================
PROFESYONEL BENZETİMLİ TAVLAMA (SIMULATED ANNEALING) ALGORİTMASI
Bilgisayar Ağları QoS Rotalama Problemi için Optimize Edilmiş Versiyon
===================================================================================

Özellikler:
- Adaptif Soğutma Stratejisi (Alpha değişimi)
- Üçlü Komşuluk Stratejisi (2-opt, Swap, Segment Reversal)
- Tabu List ile Yerel Minimum Önleme
- Restart Mekanizması
- Detaylı Metrik Takibi

Kaynaklar: NotebookLM Makale Analizi, NetworkX Optimizasyonları
Yazar: Engin Tekşüt Takımı
Tarih: 29 Aralık 2025
===================================================================================
"""

import pandas as pd
import networkx as nx
import math
import random
import copy
import matplotlib.pyplot as plt
from collections import deque
import time

# --------------------------------------------------------------------------------
# 1. VERİ YÜKLEME VE HAZIRLIK
# --------------------------------------------------------------------------------
def load_and_clean_data(node_file='BSM307_317_Guz2025_TermProject_NodeData.csv',
                        edge_file='BSM307_317_Guz2025_TermProject_EdgeData.csv',
                        demand_file='BSM307_317_Guz2025_TermProject_DemandData.csv'):
    """
    CSV dosyalarını yükler ve sayısal dönüşümleri yapar (, -> . dönüşümü).
    Dosya yolları parametre olarak verilebilir (arayüz entegrasyonu için).
    """
    try:
        nodes_df = pd.read_csv(node_file, delimiter=';', encoding='utf-8-sig')
        edges_df = pd.read_csv(edge_file, delimiter=';', encoding='utf-8-sig')
        demand_df = pd.read_csv(demand_file, delimiter=';', encoding='utf-8-sig')
    except FileNotFoundError as e:
        print(f"Hata: CSV dosyaları bulunamadı: {e}")
        return None, None, None

    # Virgüllü sayıları float'a çevirme fonksiyonu
    def to_float(x):
        try:
            return float(str(x).replace(',', '.'))
        except (ValueError, AttributeError):
            return 0.0

    # Node verilerini temizle
    nodes_df['s_ms'] = nodes_df['s_ms'].apply(to_float)
    nodes_df['r_node'] = nodes_df['r_node'].apply(to_float)

    # Edge verilerini temizle
    edges_df['r_link'] = edges_df['r_link'].apply(to_float)
    edges_df['capacity_mbps'] = edges_df['capacity_mbps'].apply(lambda x: float(x) if isinstance(x, (int, float)) else to_float(x))
    edges_df['delay_ms'] = edges_df['delay_ms'].apply(lambda x: float(x) if isinstance(x, (int, float)) else to_float(x))
    
    return nodes_df, edges_df, demand_df

def build_graph(nodes_df, edges_df):
    """
    NetworkX grafiğini oluşturur. 
    Node ve Edge özelliklerini (Gecikme, Güvenilirlik Maliyeti vb.) ekler.
    """
    G = nx.DiGraph() # Yönlü graf

    # Düğümleri ekle
    for _, row in nodes_df.iterrows():
        node_id = int(row['node_id'])
        rel = row['r_node']
        # Güvenilirlik maliyeti (-log(R)) hesapla (Minimizasyon için)
        rel_cost = -math.log(rel) if rel > 0 else 1000.0 
        
        G.add_node(node_id, 
                   proc_delay=row['s_ms'], 
                   reliability=rel,
                   reliability_cost=rel_cost)

    # Kenarları ekle
    for _, row in edges_df.iterrows():
        src, dst = int(row['src']), int(row['dst'])
        bw = row['capacity_mbps']
        delay = row['delay_ms']
        rel = row['r_link']
        
        # Güvenilirlik maliyeti
        rel_cost = -math.log(rel) if rel > 0 else 1000.0
        # Kaynak kullanım maliyeti (1000 / Bandwidth)
        res_cost = 1000.0 / bw if bw > 0 else 1000.0

        G.add_edge(src, dst, 
                   capacity=bw, 
                   link_delay=delay, 
                   reliability=rel,
                   reliability_cost=rel_cost,
                   resource_cost=res_cost)
    
    return G

# --------------------------------------------------------------------------------
# 2. PROFESYONEL BENZETİMLİ TAVLAMA (SIMULATED ANNEALING) ALGORİTMASI
# --------------------------------------------------------------------------------
class SimulatedAnnealingRouting:
    """
    Profesyonel Benzetimli Tavlama Rotalama Algoritması
    
    Özellikler:
    - Adaptif soğutma stratejisi (çok fazlı alpha)
    - Üçlü komşuluk mekanizması (2-opt, swap, segment reversal)
    - Tabu list ile döngü önleme
    - Restart mekanizması
    - Detaylı performans izleme
    """
    
    def __init__(self, graph, source, target, bandwidth_demand, weights, 
                 initial_temp=1000.0, final_temp=0.1, alpha_phase1=0.9, 
                 alpha_phase2=0.85, markov_length=200, tabu_size=30, 
                 max_no_improve=50, enable_restart=True, verbose=False):
        """
        Parametreler:
        ----------
        graph : NetworkX DiGraph
            Ağ topoloji grafiği
        source : int
            Başlangıç düğümü
        target : int
            Hedef düğüm
        bandwidth_demand : float
            İstenen minimum bant genişliği (Mbps)
        weights : dict
            QoS ağırlıkları {'delay': 0.33, 'reliability': 0.33, 'resource': 0.34}
        initial_temp : float, optional
            Başlangıç sıcaklığı (default: 1000.0)
        final_temp : float, optional
            Bitiş sıcaklığı (default: 0.1)
        alpha_phase1 : float, optional
            İlk faz soğutma katsayısı (default: 0.9)
        alpha_phase2 : float, optional
            İkinci faz soğutma katsayısı (default: 0.85)
        markov_length : int, optional
            Her sıcaklık seviyesinde iterasyon sayısı (default: 200)
        tabu_size : int, optional
            Tabu listesi boyutu (default: 30)
        max_no_improve : int, optional
            Restart için iyileşme olmayan iterasyon limiti (default: 50)
        enable_restart : bool, optional
            Restart mekanizması aktif mi? (default: True)
        verbose : bool, optional
            Detaylı log yazdır (default: False)
        """
        self.G = graph
        self.source = source
        self.target = target
        self.bandwidth_demand = bandwidth_demand
        self.weights = weights
        
        # Adaptif Sıcaklık Parametreleri (NotebookLM Q1 ve Q4)
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha_phase1 = alpha_phase1  # İlk 15 soğutma için
        self.alpha_phase2 = alpha_phase2  # Sonrası için (daha hızlı)
        self.phase_threshold = 15  # Faz geçişi iterasyonu
        
        # İterasyon Parametreleri (NotebookLM Q2)
        self.markov_length = markov_length
        
        # Tabu List (Döngü önleme) - DAHA ESNEK
        self.tabu_list = deque(maxlen=tabu_size)
        self.use_tabu = tabu_size > 0  # Tabu listesini kullan mı?
        
        # Restart Mekanizması - DAHA TOLERANSLı
        self.max_no_improve = max_no_improve
        self.enable_restart = enable_restart
        self.no_improve_counter = 0
        self.restart_count = 0
        self.max_restarts = 3  # Maksimum 3 restart
        
        # İzleme ve Log
        self.verbose = verbose
        self.iteration_count = 0
        self.cooling_step = 0
        self.best_history = []  # En iyi maliyet geçmişi
        self.acceptance_history = []  # Kabul oranı geçmişi
        self.neighbor_strategy_used = []  # Hangi komşuluk kullanıldı
        
        # Performans Metrikleri
        self.start_time = None
        self.computation_time = 0
        
        # Bant genişliği filtresi uygula
        self.filtered_G = self._filter_graph_by_bandwidth()

    def _filter_graph_by_bandwidth(self):
        """Talep edilen bant genişliğini sağlamayan kenarları filtreler."""
        # Gelen grafın tipine göre uygun alt graf oluştur
        if isinstance(self.G, nx.DiGraph):
            subgraph = nx.DiGraph()
        else:
            subgraph = nx.Graph()
        
        subgraph.add_nodes_from(self.G.nodes(data=True))
        
        valid_edges = [
            (u, v, d) for u, v, d in self.G.edges(data=True) 
            if d.get('capacity', d.get('bandwidth', float('inf'))) >= self.bandwidth_demand
        ]
        subgraph.add_edges_from(valid_edges)
        
        if self.verbose:
            print(f"[BW Filter] Orijinal Kenarlar: {self.G.number_of_edges()}, "
                  f"Geçerli Kenarlar: {subgraph.number_of_edges()}")
        
        return subgraph

    def calculate_total_cost(self, path):
        """
        Verilen bir yolun (path) Fitness değerini hesaplar.
        
        Formül (PDF'ye göre):
        Fitness = W_d × TotalDelay + W_r × (-log(R_total)) + W_res × (1000/BW_avg)
        
        Returns:
        --------
        tuple : (fitness_value, detailed_metrics_dict)
        """
        if not path or len(path) < 2:
            return float('inf'), None

        total_delay = 0.0
        total_rel_cost = 0.0
        total_res_cost = 0.0
        reliability_product = 1.0  # Gerçek güvenilirlik çarpımı (% için)

        # Kenar Maliyetleri
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            
            if not self.filtered_G.has_edge(u, v):
                return float('inf'), None
            
            edge_data = self.filtered_G[u][v]
            
            # link_delay (yeni format) veya delay (eski format)
            total_delay += edge_data.get('link_delay', edge_data.get('delay', 0))
            
            # reliability_cost varsa kullan, yoksa hesapla
            if 'reliability_cost' in edge_data:
                total_rel_cost += edge_data['reliability_cost']
            else:
                rel = edge_data.get('reliability', 1.0)
                total_rel_cost += -math.log(max(rel, 1e-6))
            
            # resource_cost varsa kullan, yoksa hesapla
            if 'resource_cost' in edge_data:
                total_res_cost += edge_data['resource_cost']
            else:
                bw = edge_data.get('capacity', edge_data.get('bandwidth', 1))
                total_res_cost += 1000.0 / max(bw, 1e-6)
            
            reliability_product *= edge_data.get('reliability', 1.0)

        # Düğüm Maliyetleri (Ara düğümler için işlem gecikmesi ve güvenilirlik)
        for node in path[1:-1]:  # Kaynak ve hedef hariç
            node_data = self.filtered_G.nodes[node]
            total_delay += node_data.get('proc_delay', node_data.get('processing_delay', 0))
            
            # reliability_cost varsa kullan, yoksa hesapla
            if 'reliability_cost' in node_data:
                total_rel_cost += node_data['reliability_cost']
            else:
                rel = node_data.get('reliability', 1.0)
                total_rel_cost += -math.log(max(rel, 1e-6))
            
            reliability_product *= node_data.get('reliability', 1.0)

        # Ağırlıklı Toplam (Fitness)
        fitness = (self.weights['delay'] * total_delay +
                   self.weights['reliability'] * total_rel_cost +
                   self.weights['resource'] * total_res_cost)
        
        detailed_metrics = {
            "total_delay": total_delay,
            "reliability_cost": total_rel_cost,
            "resource_cost": total_res_cost,
            "final_reliability": reliability_product * 100,  # Yüzde olarak
            "fitness": fitness,
            "path_length": len(path)
        }
        
        return fitness, detailed_metrics

    def get_initial_solution(self):
        """
        Başlangıç çözümünü bulur.
        Strategi: Dijkstra ile en kısa yolu bul (geçerli bir çözümle başla).
        """
        try:
            path = nx.shortest_path(self.filtered_G, self.source, self.target)
            if self.verbose:
                cost, _ = self.calculate_total_cost(path)
                print(f"[Initial Solution] Path: {path[:5]}...{path[-3:]} (Length: {len(path)}, Cost: {cost:.2f})")
            return path
        except nx.NetworkXNoPath:
            if self.verbose:
                print(f"[ERROR] Kaynak {self.source} ile Hedef {self.target} arasında yol bulunamadı!")
            return None

    def _path_to_hashable(self, path):
        """Yolu hashable formata çevir (Tabu list için)."""
        return tuple(path)

    def _is_in_tabu(self, path):
        """Yol tabu listesinde mi kontrol et."""
        return self._path_to_hashable(path) in self.tabu_list

    def _add_to_tabu(self, path):
        """Yolu tabu listesine ekle."""
        self.tabu_list.append(self._path_to_hashable(path))

    def generate_neighbor(self, current_path, temperature_ratio):
        """
        Komşu Çözüm Üretme - Üçlü Adaptif Strateji (NotebookLM Q3)
        
        Stratejiler:
        1. Node Swap: İki ara düğümün yerini değiştir
        2. 2-opt: Rotadaki iki kenarı çaprazla
        3. Segment Reversal: Alt rotayı ters çevir
        
        Adaptif Seçim: Sıcaklık oranına göre strateji seçimi
        - Yüksek T: Swap (hızlı keşif)
        - Orta T: 2-opt (dengeli)
        - Düşük T: Reversal (ince ayar)
        
        Parameters:
        -----------
        current_path : list
            Mevcut rota
        temperature_ratio : float
            T_current / T_initial (0.0 - 1.0 arası)
            
        Returns:
        --------
        tuple : (new_path, strategy_name)
        """
        if len(current_path) < 3:
            return current_path, "none"
        
        # Adaptif strateji seçimi
        if temperature_ratio > 0.6:
            strategy = "swap"  # Yüksek sıcaklık: Hızlı keşif
        elif temperature_ratio > 0.3:
            strategy = "2-opt"  # Orta sıcaklık: Dengeli arama
        else:
            strategy = "reversal"  # Düşük sıcaklık: Hassas ayar
        
        # Rastgele seçim yap (çeşitlilik için)
        if random.random() < 0.1:  # %10 ihtimalle farklı strateji
            strategy = random.choice(["swap", "2-opt", "reversal"])
        
        # Stratejiye göre komşu üret
        max_attempts = 5  # AZALTILDI: Daha hızlı geçiş
        for attempt in range(max_attempts):
            try:
                if strategy == "swap":
                    new_path = self._neighbor_swap(current_path)
                elif strategy == "2-opt":
                    new_path = self._neighbor_2opt(current_path)
                else:  # reversal
                    new_path = self._neighbor_reversal(current_path)
                
                # Geçerliliği ve tabu kontrolü (Tabu kontrolünü sadece %50 ihtimalle uygula)
                if new_path and (not self.use_tabu or random.random() < 0.5 or not self._is_in_tabu(new_path)):
                    cost, metrics = self.calculate_total_cost(new_path)
                    if cost < float('inf'):
                        return new_path, strategy
            except Exception as e:
                if self.verbose:
                    print(f"[DEBUG] Komşu üretme hatası: {e}")
                continue
        
        # Hiçbir geçerli komşu bulunamazsa basit bir değişiklik dene
        try:
            new_path = self._neighbor_2opt(current_path)  # En basit strateji
            return new_path, "2-opt-fallback"
        except Exception:
            return current_path, "failed"  # Son çare: aynı yolu döndür

    def _neighbor_swap(self, path):
        """
        Strateji 1: Node Swap
        Rotadaki iki ara düğümün yerini değiştirir.
        """
        if len(path) < 4:  # En az 4 düğüm gerekli (kaynak, 2 ara, hedef)
            return path
        
        # Kaynak ve hedef hariç ara düğümlerden iki tanesini seç
        inner_nodes = path[1:-1]
        if len(inner_nodes) < 2:
            return path
        
        idx1, idx2 = random.sample(range(len(inner_nodes)), 2)
        
        new_path = path.copy()
        # +1 offset çünkü path[0] kaynak
        new_path[idx1+1], new_path[idx2+1] = new_path[idx2+1], new_path[idx1+1]
        
        return new_path

    def _neighbor_2opt(self, path):
        """
        Strateji 2: 2-opt Optimization
        İki kenarı kesip çaprazlayarak yeni rota oluşturur.
        
        Klasik TSP 2-opt mantığı:
        Örnek: [A, B, C, D, E] -> [A, B, D, C, E] (C-D kenarı ters çevrilir)
        """
        if len(path) < 4:
            return path
        
        # Rastgele iki nokta seç (i < j)
        i = random.randint(1, len(path) - 3)
        j = random.randint(i + 1, len(path) - 2)
        
        # [0:i] + reversed[i:j+1] + [j+1:]
        new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
        
        return new_path

    def _neighbor_reversal(self, path):
        """
        Strateji 3: Segment Reversal
        Alt segment seçip alternatif yol bulur (keşif odaklı).
        """
        if len(path) < 3:
            return path
        
        # İki nokta seç ve arasında alternatif yol ara
        idx1 = random.randint(0, len(path) - 2)
        idx2 = random.randint(idx1 + 1, len(path) - 1)
        
        u, v = path[idx1], path[idx2]
        
        # Mevcut alt yolun kenarlarını geçici olarak kaldır
        temp_G = self.filtered_G.copy()
        for k in range(idx1, idx2):
            n1, n2 = path[k], path[k+1]
            if temp_G.has_edge(n1, n2):
                temp_G.remove_edge(n1, n2)
        
        try:
            # Alternatif bir alt yol bul
            alt_sub_path = nx.shortest_path(temp_G, u, v)
            # Yeni yolu birleştir
            new_path = path[:idx1] + alt_sub_path + path[idx2+1:]
            return new_path
        except nx.NetworkXNoPath:
            # Alternatif bulunamazsa basit 2-opt dene
            return self._neighbor_2opt(path)

    def run(self):
        """
        Profesyonel SA Algoritmasını Çalıştırır
        
        Returns:
        --------
        tuple : (best_path, best_cost, detailed_metrics, history_dict)
        """
        self.start_time = time.time()
        
        # Başlangıç çözümü
        current_path = self.get_initial_solution()
        if not current_path:
            return None, float('inf'), {}, {}

        current_cost, current_metrics = self.calculate_total_cost(current_path)
        
        # En iyi çözüm
        best_path = current_path.copy()
        best_cost = current_cost
        best_metrics = current_metrics.copy() if current_metrics else {}
        
        # Sıcaklık
        T = self.initial_temp
        
        # İzleme değişkenleri
        accepted_moves = 0
        total_moves = 0
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"BENZETİMLİ TAVLAMA BAŞLIYOR")
            print(f"{'='*70}")
            print(f"Başlangıç Sıcaklığı: {T:.2f}")
            print(f"İlk Çözüm Maliyeti: {current_cost:.2f}")
            print(f"Markov Uzunluğu: {self.markov_length}")
            print(f"{'='*70}\n")
        
        # Ana döngü
        while T > self.final_temp:
            self.cooling_step += 1
            epoch_accepts = 0
            
            # Adaptif alpha seçimi (NotebookLM Q1: İki aşamalı soğutma)
            if self.cooling_step <= self.phase_threshold:
                current_alpha = self.alpha_phase1
            else:
                current_alpha = self.alpha_phase2
            
            # İç döngü (Markov Chain - NotebookLM Q2)
            for markov_iter in range(self.markov_length):
                self.iteration_count += 1
                total_moves += 1
                
                # Sıcaklık oranını hesapla (komşuluk stratejisi için)
                temp_ratio = T / self.initial_temp
                
                # Komşu üret
                new_path, strategy = self.generate_neighbor(current_path, temp_ratio)
                new_cost, new_metrics = self.calculate_total_cost(new_path)
                
                # Maliyet farkı
                delta_E = new_cost - current_cost
                
                # Kabul Kriteri (NotebookLM Q4: Metropolis/Boltzmann)
                accept = False
                if delta_E < 0:
                    # Daha iyi çözüm, kabul et
                    accept = True
                else:
                    # Kötü çözüm, olasılıksal kabul
                    probability = math.exp(-delta_E / T)
                    if random.random() < probability:
                        accept = True
                
                if accept:
                    # Yeni çözümü kabul et
                    current_path = new_path
                    current_cost = new_cost
                    current_metrics = new_metrics
                    
                    accepted_moves += 1
                    epoch_accepts += 1
                    
                    # Tabu listesine ekle (sadece kullanılıyorsa)
                    if self.use_tabu:
                        self._add_to_tabu(current_path)
                    
                    # En iyi çözümü güncelle
                    if current_cost < best_cost:
                        best_path = current_path.copy()
                        best_cost = current_cost
                        best_metrics = current_metrics.copy() if current_metrics else {}
                        self.no_improve_counter = 0
                        
                        if self.verbose and self.iteration_count % 100 == 0:
                            print(f"[✓] Yeni En İyi: {best_cost:.2f} (Iter: {self.iteration_count}, T: {T:.2f})")
                    else:
                        self.no_improve_counter += 1
                
                # Strateji kaydı
                self.neighbor_strategy_used.append(strategy)
            
            # Epoch sonu istatistikleri
            acceptance_rate = epoch_accepts / self.markov_length if self.markov_length > 0 else 0
            self.best_history.append(best_cost)
            self.acceptance_history.append(acceptance_rate)
            
            if self.verbose and self.cooling_step % 5 == 0:
                print(f"[Cooling {self.cooling_step:3d}] T={T:7.2f} | Best={best_cost:8.2f} | "
                      f"Accept={acceptance_rate*100:5.1f}% | Alpha={current_alpha:.2f}")
            
            # Restart Mekanizması (Takılıp kalma durumu)
            if self.enable_restart and self.no_improve_counter > self.max_no_improve and self.restart_count < self.max_restarts:
                self.restart_count += 1
                
                if self.verbose:
                    print(f"\n[⚠ RESTART #{self.restart_count}] {self.max_no_improve} iterasyondur iyileşme yok. Yeniden başlatılıyor...\n")
                
                # Sıcaklığı yeniden yükselt (tam başa dönme)
                T = self.initial_temp * 0.7  # %70 sıcaklıkla başla
                self.no_improve_counter = 0
                
                # Mevcut en iyi çözümden devam et (rastgele başlama yerine)
                current_path = best_path.copy()
                current_cost = best_cost
                current_metrics = best_metrics.copy()
            
            # Soğutma (NotebookLM Q1: Adaptif alpha)
            T = T * current_alpha
        
        # Hesaplama süresi
        self.computation_time = time.time() - self.start_time
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"BENZETİMLİ TAVLAMA TAMAMLANDI")
            print(f"{'='*70}")
            print(f"Toplam İterasyon: {self.iteration_count}")
            print(f"Toplam Soğutma Adımı: {self.cooling_step}")
            print(f"Kabul Oranı: {(accepted_moves/total_moves*100):.2f}%")
            print(f"Hesaplama Süresi: {self.computation_time:.2f} saniye")
            print(f"En İyi Maliyet: {best_cost:.2f}")
            print(f"En İyi Yol Uzunluğu: {len(best_path)} düğüm")
            print(f"{'='*70}\n")
        
        # Detaylı sonuç paketi
        detailed_result = {
            "Total Delay (ms)": best_metrics["total_delay"],
            "Total Reliability Cost": best_metrics["reliability_cost"],
            "Total Resource Cost": best_metrics["resource_cost"],
            "Final Reliability (%)": best_metrics["final_reliability"],
            "Weighted Fitness": best_cost,
            "Path Length": len(best_path),
            "Computation Time (s)": self.computation_time,
            "Total Iterations": self.iteration_count,
            "Cooling Steps": self.cooling_step,
            "Acceptance Rate (%)": (accepted_moves / total_moves * 100) if total_moves > 0 else 0
        }
        
        history_data = {
            "best_cost_history": self.best_history,
            "acceptance_rate_history": self.acceptance_history,
            "strategy_usage": self._count_strategy_usage()
        }
        
        return best_path, best_cost, detailed_result, history_data

    def _count_strategy_usage(self):
        """Kullanılan komşuluk stratejilerinin istatistiklerini hesapla."""
        from collections import Counter
        counts = Counter(self.neighbor_strategy_used)
        total = sum(counts.values())
        
        stats = {}
        for strategy, count in counts.items():
            stats[strategy] = {
                "count": count,
                "percentage": (count / total * 100) if total > 0 else 0
            }
        return stats

# --------------------------------------------------------------------------------
# 3. GÖRSELLEŞTİRME VE ANALİZ FONKSİYONLARI
# --------------------------------------------------------------------------------
def plot_convergence(history_data, save_path=None):
    """
    SA algoritmasının yakınsama grafiğini çizer.
    
    Parameters:
    -----------
    history_data : dict
        run() fonksiyonundan dönen history verileri
    save_path : str, optional
        Grafik kaydedilecekse dosya yolu
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Benzetimli Tavlama Performans Analizi', fontsize=16, fontweight='bold')
    
    # 1. En İyi Maliyet Geçmişi
    ax1 = axes[0, 0]
    ax1.plot(history_data['best_cost_history'], linewidth=2, color='#2563eb')
    ax1.set_title('En İyi Çözüm Yakınsaması', fontweight='bold')
    ax1.set_xlabel('Soğutma Adımı')
    ax1.set_ylabel('En İyi Maliyet (Fitness)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Kabul Oranı
    ax2 = axes[0, 1]
    ax2.plot(history_data['acceptance_rate_history'], linewidth=2, color='#10b981')
    ax2.set_title('Kabul Oranı Değişimi', fontweight='bold')
    ax2.set_xlabel('Soğutma Adımı')
    ax2.set_ylabel('Kabul Oranı (%)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Strateji Kullanım Dağılımı
    ax3 = axes[1, 0]
    strategies = history_data['strategy_usage']
    if strategies:
        labels = list(strategies.keys())
        sizes = [strategies[k]['percentage'] for k in labels]
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']
        ax3.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors[:len(labels)], startangle=90)
        ax3.set_title('Komşuluk Stratejisi Dağılımı', fontweight='bold')
    
    # 4. İstatistik Tablosu
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = "📊 ÖZET İSTATİSTİKLER\n\n"
    stats_text += f"Toplam Soğutma Adımı: {len(history_data['best_cost_history'])}\n"
    stats_text += f"Başlangıç Maliyeti: {history_data['best_cost_history'][0]:.2f}\n"
    stats_text += f"Final Maliyeti: {history_data['best_cost_history'][-1]:.2f}\n"
    improvement = ((history_data['best_cost_history'][0] - history_data['best_cost_history'][-1]) / 
                   history_data['best_cost_history'][0] * 100)
    stats_text += f"İyileşme Oranı: %{improvement:.2f}\n\n"
    stats_text += "Strateji Kullanımı:\n"
    for strategy, data in strategies.items():
        stats_text += f"  • {strategy}: {data['count']} kez\n"
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center', 
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[✓] Grafik kaydedildi: {save_path}")
    
    plt.show()


# --------------------------------------------------------------------------------
# 4. ARAYÜZ ENTEGRASYONU İÇİN WRAPPER FONKSİYONU
# --------------------------------------------------------------------------------
def calculate_route_with_sa(graph, source, target, bandwidth_demand, weights,
                            initial_temp=300.0, alpha_phase1=0.85, markov_length=50,
                            verbose=False):
    """
    Arayüzden çağrılabilecek basit wrapper fonksiyon.
    topology.py'deki calculate_path fonksiyonuna benzer formatta döner.
    
    Parameters:
    -----------
    graph : NetworkX DiGraph
        Ağ grafiği
    source : int
        Başlangıç düğümü
    target : int
        Hedef düğüm
    bandwidth_demand : float
        Bant genişliği talebi (Mbps)
    weights : dict
        {'delay': w1, 'reliability': w2, 'resource': w3}
    initial_temp : float, optional
        Başlangıç sıcaklığı
    alpha_phase1 : float, optional
        Soğutma katsayısı
    markov_length : int, optional
        İterasyon sayısı
    verbose : bool, optional
        Detaylı çıktı
        
    Returns:
    --------
    dict or None : topology.py formatında sonuç
        {
            "path": [düğüm listesi],
            "total_delay": float,
            "reliability_log": float,
            "resource_cost": float,
            "final_reliability": float
        }
    """
    try:
        # Eğer graf yönsüz (Graph) ise yönlü (DiGraph) yap
        if isinstance(graph, nx.Graph) and not isinstance(graph, nx.DiGraph):
            digraph = nx.DiGraph()
            # Düğümleri ve özelliklerini kopyala
            for node, data in graph.nodes(data=True):
                digraph.add_node(node, **data)
            # Kenarları çift yönlü ekle
            for u, v, data in graph.edges(data=True):
                digraph.add_edge(u, v, **data)
                digraph.add_edge(v, u, **data)
            graph = digraph
        
        sa_solver = SimulatedAnnealingRouting(
            graph=graph,
            source=source,
            target=target,
            bandwidth_demand=bandwidth_demand,
            weights=weights,
            initial_temp=initial_temp,
            final_temp=1.0,  # Daha yüksek final temp (daha hızlı biter)
            alpha_phase1=alpha_phase1,
            alpha_phase2=0.80,  # Daha hızlı soğutma
            markov_length=markov_length,
            tabu_size=10,  # Daha küçük tabu
            max_no_improve=50,  # Daha az iterasyon
            enable_restart=False,  # Restart'ı kapat (daha hızlı)
            verbose=verbose
        )
        
        best_path, best_cost, detailed_metrics, history = sa_solver.run()
        
        if best_path:
            # topology.py formatına uyarla
            result = {
                "path": best_path,
                "total_delay": detailed_metrics["Total Delay (ms)"],
                "reliability_log": detailed_metrics["Total Reliability Cost"],
                "resource_cost": detailed_metrics["Total Resource Cost"],
                "final_reliability": detailed_metrics["Final Reliability (%)"]
            }
            return result
        else:
            return None
            
    except Exception as e:
        if verbose:
            print(f"[ERROR] SA Hesaplama Hatası: {e}")
        return None


# --------------------------------------------------------------------------------
# 5. TEST VE ÇALIŞTIRMA (Standalone Demo)
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*80)
    print("PROFESYONEL BENZETİMLİ TAVLAMA ALGORİTMASI - TEST MODU")
    print("="*80 + "\n")
    
    # Verileri yükle
    print("[1/5] Veri dosyaları yükleniyor...")
    nodes, edges, demands = load_and_clean_data()
    
    if nodes is not None and edges is not None and demands is not None:
        print(f"[✓] {len(nodes)} düğüm, {len(edges)} kenar, {len(demands)} talep yüklendi.\n")
        
        # Grafiği oluştur
        print("[2/5] Ağ grafiği oluşturuluyor...")
        G_network = build_graph(nodes, edges)
        print(f"[✓] Graf oluşturuldu: {G_network.number_of_nodes()} düğüm, {G_network.number_of_edges()} kenar\n")
        
        # Örnek bir talep seç
        print("[3/5] Test talebi seçiliyor...")
        sample_demand = demands.iloc[0]
        src = int(sample_demand['src'])
        dst = int(sample_demand['dst'])
        bw_req = float(sample_demand['demand_mbps'])
        
        print(f"[✓] Seçilen Talep: Kaynak={src}, Hedef={dst}, Bant Genişliği={bw_req} Mbps\n")
        
        # QoS Ağırlıkları
        user_weights = {'delay': 0.33, 'reliability': 0.33, 'resource': 0.34}
        
        print("[4/5] Benzetimli Tavlama algoritması çalıştırılıyor...")
        print("-" * 80)
        
        # SA Parametreleri (Test için optimize edilmiş)
        sa_solver = SimulatedAnnealingRouting(
            graph=G_network,
            source=src,
            target=dst,
            bandwidth_demand=bw_req,
            weights=user_weights,
            initial_temp=800.0,       # Biraz düşük başlangıç
            final_temp=0.5,           # Biraz yüksek bitiş
            alpha_phase1=0.92,        # Dengeli soğuma
            alpha_phase2=0.88,
            markov_length=120,        # Orta seviye iterasyon
            tabu_size=20,             # Daha küçük tabu
            max_no_improve=150,       # ARTIRILDI: Daha toleranslı
            enable_restart=True,
            verbose=True
        )
        
        best_route, min_cost, details, history = sa_solver.run()
        
        if best_route:
            print("\n" + "="*80)
            print("SONUÇLAR")
            print("="*80)
            print(f"\n✅ EN İYİ ROTA BULUNDU!")
            print(f"\nRota: {best_route[:10]}... → ...{best_route[-5:]}")
            print(f"Yol Uzunluğu: {len(best_route)} düğüm (hop count)")
            print(f"\n📊 METRIKLER:")
            for key, value in details.items():
                if isinstance(value, float):
                    print(f"  • {key}: {value:.4f}")
                else:
                    print(f"  • {key}: {value}")
            
            print("\n[5/5] Performans grafikleri oluşturuluyor...")
            plot_convergence(history, save_path="sa_convergence_analysis.png")
            
        else:
            print("\n❌ UYGUN YOL BULUNAMADI!")
            print("Olası sebepler:")
            print("  - Bant genişliği talebi çok yüksek")
            print("  - Kaynak ve hedef arasında bağlantı yok")
    else:
        print("❌ Veri dosyaları yüklenemedi. Lütfen CSV dosyalarını kontrol edin.")
    
    print("\n" + "="*80)
    print("TEST TAMAMLANDI")
    print("="*80 + "\n")