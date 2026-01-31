"""
===================================================================================
                          NETWORK TOPOLOGY MANAGEMENT MODULE
===================================================================================
Bu modül, ağ topolojisi oluşturma, yönetme ve çoklu QoS optimizasyon algoritmalarını
çalıştırma işlevlerini sağlar.

Temel Özellikler:
------------------
1. Erdős-Rényi rastgele ağ oluşturma
2. Trade-off mekanizması ile link tipi bazlı QoS parametreleri (fiber/microwave/satellite)
3. 4 farklı optimizasyon algoritması desteği:
   - Dijkstra (Baseline)
   - Q-Learning (Pekiştirmeli Öğrenme)
   - PSO (Parçacık Sürü Optimizasyonu)
   - Genetik Algoritma
   - Benzetimli Tavlama (SA)
4. Dinamik ağırlıklı rota hesaplama (delay + reliability + resource)
5. CSV tabanlı ağ yükleme desteği

Kullanım:
---------
    tm = TopologyManager()
    G, pos = tm.create_network()
    result = tm.calculate_path(source=0, target=10, w_delay=0.5, w_rel=0.3, w_res=0.2)

Author: [Proje Ekibi]
Date: Aralık 2024
"""

import networkx as nx  # Ağ grafiği ve algoritmaları için
import random          # Rastgele parametre üretimi
import math            # Logaritmik güvenilirlik hesaplamaları
from Algorithms.pso import QoS_PSO_Solver                # Parçacık Sürü Optimizasyonu
from Algorithms.qlearning_algorithm import QLearningRouter  # Pekiştirmeli Öğrenme
from Algorithms.ga import GenetikAlgorithm               # Genetik Algoritma
from Algorithms.sa_algorithm import calculate_route_with_sa  # Benzetimli Tavlama


class TopologyManager:
    """
    Ağ Topolojisi Yönetim Sınıfı
    
    Bu sınıf, rastgele ağ oluşturma, QoS parametrelerini ayarlama ve çoklu
    optimizasyon algoritmalarıyla rota hesaplama işlevlerini sağlar.
    
    Attributes:
        G (nx.Graph): NetworkX graf nesnesi (ağ topolojisi)
        pos (dict): Düğümlerin görselleştirme konumları {node_id: (x, y)}
        num_nodes (int): Ağdaki toplam düğüm sayısı (varsayılan 250)
    
    Trade-off Mekanizması:
        - Fiber: Yüksek bant genişliği, düşük gecikme → Düşük güvenilirlik
        - Microwave: Orta seviye tüm parametreler → Dengeli
        - Satellite: Düşük bant, yüksek gecikme → Çok yüksek güvenilirlik
    """
    def __init__(self):
        """Topoloji yöneticisini başlat (boş graf)"""
        self.G = None  # NetworkX graf nesnesi
        self.pos = None  # Düğüm pozisyonları
        self.num_nodes = 250  # Varsayılan düğüm sayısı

    def create_network(self, seed=None):
        """
        Erdős-Rényi rastgele graf modeli ile ağ topolojisi oluşturur.
        Trade-off mekanizması ile link tiplerine göre QoS parametreleri atar.
        
        Args:
            seed (int, optional): Rastgele sayı üreteci tohum değeri.
                                  Aynı seed değeri ile aynı ağ topolojisi üretilir.
                                  None ise her çalıştırmada farklı ağ oluşur.
        
        Returns:
            tuple: (G, pos) - NetworkX graf ve düğüm pozisyonları
            
        Trade-off Logic:
            - fiber: bw=800-1000 Mbps, delay=1-5ms, reliability=0.90-0.95
            - microwave: bw=300-600 Mbps, delay=5-10ms, reliability=0.95-0.98
            - satellite: bw=10-100 Mbps, delay=20-50ms, reliability=0.99-0.9999
        
        Seed Kullanımı:
            >>> tm = TopologyManager()
            >>> G1, pos1 = tm.create_network(seed=42)
            >>> G2, pos2 = tm.create_network(seed=42)  # G1 ile aynı ağ
        """
        # Seed ayarla - Tüm rastgele işlemleri etkiler
        if seed is not None:
            random.seed(seed)
        
        # 1. Topoloji Oluşturma (Erdős-Rényi Modeli)
        self.G = nx.erdos_renyi_graph(n=self.num_nodes, p=0.04, seed=seed) 
        
        # 2. Bağlantı Garantisi - Kopuk bileşenleri birleştir
        # Eğer graf bağlantılı değilse (disconnected), ayrı parçaları köprü kenarlarla bağla
        if not nx.is_connected(self.G):
            components = list(nx.connected_components(self.G))
            for i in range(len(components)-1):
                # Her bileşenden bir düğüm seç ve sonraki bileşene bağla
                u = list(components[i])[0]
                v = list(components[i+1])[0]
                self.G.add_edge(u, v)
       
        # 3. Düğüm Konumlarını Belirle (Spring layout - görselleştirme için)
        # Görselleştirme seed'i sabit tutarak konumların tutarlı olmasını sağla
        layout_seed = seed if seed is not None else 42
        self.pos = nx.spring_layout(self.G, seed=layout_seed)

        # 4. Düğüm QoS Özelliklerini Ata
        # Her düğüme işlem gecikmesi ve güvenilirlik atanır
        for node in self.G.nodes():
            self.G.nodes[node]['proc_delay'] = random.uniform(0.5, 2.0)  # ms
            self.G.nodes[node]['reliability'] = random.uniform(0.95, 0.999)  # %95-99.9

        # 5. Bağlantı Özelliklerini Ata (TRADE-OFF MEKANİZMASI)
        # Link tipleri: fiber (hızlı-riskli), microwave (orta), satellite (yavaş-güvenli)
        link_types = ["fiber", "microwave", "satellite"]
        
        # Her kenara rastgele link tipi ata ve QoS parametrelerini belirle
        for u, v in self.G.edges():
            l_type = random.choice(link_types)
            
            if l_type == "fiber":
                # FİBER: Çok Hızlı, Geniş Bant genişliği → Düşük Güvenilirlik (Trade-off!)
                delay = random.uniform(1, 5)        # Düşük gecikme
                bw = random.uniform(800, 1000)      # Yüksek bant genişliği
                rel = random.uniform(0.90, 0.95)    # Düşük güvenilirlik (risk)
                
            elif l_type == "microwave":
                # MİKRODALGA: Tüm parametreler orta seviyede (dengeli)
                delay = random.uniform(5, 10)
                bw = random.uniform(300, 600)
                rel = random.uniform(0.95, 0.98)
                
            else:  # satellite
                # UYDU: Çok Yavaş, Dar Bant → Çok Yüksek Güvenilirlik (Trade-off!)
                delay = random.uniform(20, 50)      # Yüksek gecikme
                bw = random.uniform(10, 100)        # Düşük bant genişliği
                rel = random.uniform(0.99, 0.9999)  # Çok yüksek güvenilirlik
            
            # QoS Parametrelerini Kenara Kaydet
            self.G.edges[u, v]['link_delay'] = delay       # ms
            self.G.edges[u, v]['bandwidth'] = bw           # Mbps
            self.G.edges[u, v]['reliability'] = rel        # 0-1 arası
            self.G.edges[u, v]['type'] = l_type            # Link tipi etiketi

        return self.G, self.pos

   
    def calculate_path(self, source, target, w_delay, w_rel, w_res, algorithm="Dijkstra", demand_value=100, algo_params=None):
        """
        Seçilen algoritmaya göre kaynak-hedef arası optimal rotayı hesaplar.
        
        Çoklu QoS kriterlerini (gecikme, güvenilirlik, kaynak) ağırlıklı toplam
        (Weighted Sum Method) ile tek bir maliyet fonksiyonuna dönüştürür.
        
        Args:
            source (int): Başlangıç düğümü ID'si
            target (int): Hedef düğüm ID'si
            w_delay (float): Gecikme ağırlığı (0-1 arası)
            w_rel (float): Güvenilirlik ağırlığı (0-1 arası)
            w_res (float): Kaynak ağırlığı (0-1 arası)
            algorithm (str): Algoritma adı ("Dijkstra", "Q-Learning", "PSO", "Genetik", "SA")
            demand_value (int): Talep edilen bant genişliği (Mbps)
            algo_params (dict, optional): Algoritma özel parametreleri
                Q-Learning: {'alpha', 'gamma', 'epsilon', 'episodes'}
                PSO: {'swarm_size', 'iterations', 'w', 'c1', 'c2'}
                GA: {'population', 'generations', 'crossover', 'mutation'}
                SA: {'initial_temp', 'cooling_rate', 'iterations'}
        
        Returns:
            dict: Rota bilgileri veya None (yol bulunamazsa)
                {
                    "path": [node_list],
                    "total_delay": float,
                    "final_reliability": float (% cinsinden),
                    "resource_cost": float
                }
        
        Not:
            - Güvenilirlik logaritmik ölçekte hesaplanır: -ln(reliability)
            - Kaynak maliyeti = 1000 / bandwidth (ters orantılı)
            - Dinamik ağırlıklar her algoritma çağrısında güncellenir
        """
        if self.G is None: 
            return None
        
        # Varsayılan parametreleri başlat (algoritma parametresi verilmemişse)
        if algo_params is None:
            algo_params = {}
            
        # ===== AĞIRLIK NORMALİZASYONU =====
        # Kullanıcının verdiği ağırlıkları normalize et (toplamları 1 olsun)
        total_w = w_delay + w_rel + w_res
        if total_w == 0: 
            total_w = 1  # Sıfıra bölünmeyi engelle
        
        wd = w_delay / total_w    # Normalize gecikme ağırlığı
        wr = w_rel / total_w      # Normalize güvenilirlik ağırlığı
        wres = w_res / total_w    # Normalize kaynak ağırlığı
      
        # ===== DİNAMİK KENAR AĞIRLIKLARINI HESAPLA =====
        # Weighted Sum Method: Her kenar için toplam maliyet = wd*delay + wr*rel + wres*resource
        for u, v in self.G.edges():
            data = self.G.edges[u, v]
            
            # 1. Gecikme maliyeti (direkt kullan)
            cost_delay = data['link_delay']
            
            # 2. Güvenilirlik maliyeti (logaritmik ölçek)
            # Yüksek güvenilirlik → düşük maliyet (negatif log kullanıyoruz)
            if data['reliability'] > 0:
                cost_reliability = -math.log(data['reliability'])
            else:
                cost_reliability = 100  # Güvenilirlik 0 ise çok yüksek ceza
            
            # 3. Kaynak maliyeti (bant genişliği ile ters orantılı)
            # Yüksek bant → düşük maliyet
            if data['bandwidth'] > 0:
                cost_resource = 1000.0 / data['bandwidth']
            else:
                cost_resource = 1000.0  # Bant 0 ise yüksek ceza

            # Weighted Sum: Ağırlıklı toplam maliyet
            total_cost = (wd * cost_delay) + (wr * cost_reliability * 100) + (wres * cost_resource)
            
            # Hesaplanan maliyeti kenara ata (shortest_path bu ağırlığı kullanacak)
            self.G.edges[u, v]['dynamic_weight'] = total_cost

        # ===== ALGORİTMA SEÇİMİ VE ÇALIŞTIRMA =====
        try:
            path = []
            
            # ===== 1. Q-LEARNING ALGORİTMASI =====
            # Pekiştirmeli öğrenme tabanlı rota optimizasyonu
            if "Q-Learning" in algorithm:
                # Seed ayarla (checkbox açıksa)
                seed_value = algo_params.get('seed', None)
                if seed_value is not None:
                    random.seed(seed_value)
                    import numpy as np
                    np.random.seed(seed_value)
                
                router = QLearningRouter(
                    alpha=algo_params.get('alpha', 0.1),          # Öğrenme hızı
                    gamma=algo_params.get('gamma', 0.9),          # İndirim faktörü
                    epsilon_start=algo_params.get('epsilon', 0.9) # Keşif oranı
                )
                # Mevcut grafiği Q-Learning modülüne yükle
                router.load_network_from_graph(self.G)
                
                # Rota hesaplama (episode sayısı kadar eğitim)
                result = router.calculate_path(
                    source=source, 
                    target=target,
                    w_delay=w_delay, 
                    w_reliability=w_rel, 
                    w_resource=w_res,
                    bandwidth_demand=demand_value, 
                    num_episodes=algo_params.get('episodes', 500)
                )
                return result
            
            # ===== 2. PSO (Parçacık Sürü Optimizasyonu) =====
            # Sürü zekası tabanlı global optimizasyon
            elif "PSO" in algorithm:
                from Algorithms.pso import QoS_PSO_Solver
                
                # Seed ayarla (checkbox açıksa)
                seed_value = algo_params.get('seed', None)
                if seed_value is not None:
                    random.seed(seed_value)
                    import numpy as np
                    np.random.seed(seed_value)
                
                # PSO solver'ı başlat (self.G'yi doğrudan kullan)
                pso_solver = QoS_PSO_Solver(G=self.G) 
                
                # Talebi çöz (swarm_size, iterations parametreleri PSO içinde)
                path, metrics = pso_solver.solve_demand(
                    source, target, demand_value, (wd, wr, wres)
                )
                
                # Sonuçları standart formata dönüştür
                if path and metrics is not None:
                    return {
                        "path": path,
                        "total_delay": metrics['delay'],
                        "final_reliability": metrics['reliability'] * 100,  # Yüzde cinsine çevir
                        "resource_cost": metrics['resource']
                    }
                return None
            
            # ===== 3. GENETİK ALGORİTMA =====
            # Evrimsel hesaplama tabanlı optimizasyon
            elif "Genetik" in algorithm:
                # Seed ayarla (checkbox açıksa)
                seed_value = algo_params.get('seed', None)
                if seed_value is not None:
                    random.seed(seed_value)
                    import numpy as np
                    np.random.seed(seed_value)
                
                ga_solver = GenetikAlgorithm()
                
                # GA'nın beklediği ağırlık sözlüğünü hazırla
                ga_weights = {
                    "w_delay": wd,
                    "w_reliability": wr,
                    "w_resource": wres
                }
                
                # Genetik algoritmayı çalıştır (populasyon evrimiyle en iyi rotayı bul)
                best_path = ga_solver.genetik_calistir(
                    self.G, source, target, demand_value, ga_weights
                )
                
                # Bulunan rotanın metriklerini hesapla
                res = ga_solver.get_path_metrics(best_path, demand_value)
                
                # Geçerli bir yol bulunduysa döndür
                if res.get("valid"):
                    return {
                        "path": res["path"],
                        "total_delay": res["total_delay"],
                        "final_reliability": res["total_reliability"] * 100,
                        "resource_cost": res["resource_cost"]
                    }
                return None
            
            # ===== 4. BENZETİMLİ TAVLAMA (Simulated Annealing) =====
            # Metalurjideki tavlama sürecinden esinlenmiş lokal arama
            # ===== 4. BENZETİMLİ TAVLAMA (Simulated Annealing) =====
            # Metalurjideki tavlama sürecinden esinlenmiş lokal arama
            elif "SA" in algorithm or "Benzetimli" in algorithm:
                # Seed ayarla (checkbox açıksa)
                seed_value = algo_params.get('seed', None)
                if seed_value is not None:
                    random.seed(seed_value)
                    import numpy as np
                    np.random.seed(seed_value)
                
                sa_weights = {
                    'delay': wd,
                    'reliability': wr,
                    'resource': wres
                }
                print(f"\n[SA] Benzetimli Tavlama başlatılıyor: {source} → {target}")
                print(f"[SA] Seed: {seed_value if seed_value is not None else 'Rastgele'}")
                
                # SA algoritmasını çalıştır (sıcaklık azaltma ile lokal optimum kaçışı)
                result = calculate_route_with_sa(
                    self.G, source, target, demand_value, sa_weights,
                    verbose=True  # İlerleme loglarını göster
                )
                
                if result:
                    print(f"[SA] Başarılı! Yol uzunluğu: {len(result['path'])} düğüm")
                    return result
                print("[SA] Uygun yol bulunamadı!")
                return None
                
            # ===== 5. DİJKSTRA (BASELINE) =====
            # NetworkX shortest_path algoritması (dynamic_weight ile)
            else:  
                path = nx.shortest_path(
                    self.G, 
                    source=source, 
                    target=target, 
                    weight='dynamic_weight'
                )     
            
            # ===== METRİK HESAPLAMA (Dijkstra için) =====
            # Bulunan rotanın toplam gecikme, güvenilirlik ve kaynak maliyetini hesapla
            metrics = {
                "total_delay": 0,         # Toplam gecikme (ms)
                "reliability_log": 0,     # Logaritmik güvenilirlik toplamı
                "resource_cost": 0,       # Toplam kaynak maliyeti
                "path": path,             # Düğüm listesi
                "final_reliability": 0    # Yüzde cinsinden nihai güvenilirlik
            }

            if not path: 
                return None

            # 1. Kenar metriklerini topla (link gecikmeleri, güvenilirlik, bant genişliği)
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                d = self.G.edges[u, v]
                
                metrics["total_delay"] += d['link_delay']
                metrics["reliability_log"] += -math.log(d['reliability'])  # Log ölçeğinde
                metrics["resource_cost"] += (1000.0 / d['bandwidth'])      # Ters orantılı
            
            # 2. Düğüm işlem gecikmelerini ekle (Kaynak ve Hedef hariç ara düğümler)
            for node in path[1:-1]:
                metrics["total_delay"] += self.G.nodes[node]['proc_delay']

            # 3. Logaritmik güvenilirliği yüzdeye çevir
            # e^(-sum(log(rel))) = product(rel) → yüzde olarak
            metrics["final_reliability"] = math.exp(-metrics["reliability_log"]) * 100
            
            return metrics

        except nx.NetworkXNoPath:
            # Kaynak-hedef arası yol yok (disconnected)
            return None
        except Exception as e:
            # Diğer hatalar (bölme, tip hatası vb.)
            print(f"Hata oluştu: {e}")
            return None
    
    def build_from_csv(self, node_file, edge_file, seed=None):
        """
        CSV dosyalarından ağ topolojisi yükler.
        
        Kullanıcının özel ağ yapısını NodeData.csv ve EdgeData.csv'den okur.
        Bu sayede deterministik testler ve özel senaryolar çalıştırılabilir.
        
        Args:
            node_file (str): Düğüm özelliklerini içeren CSV dosya yolu
                Format: node_id; processing_delay_ms; reliability_node
                Örnek: 0;1.2;0.98
            
            edge_file (str): Kenar özelliklerini içeren CSV dosya yolu
                Format: src; dst; capacity_mbps; delay_ms; reliability_link
                Örnek: 0;1;500;3.5;0.95
            
            seed (int, optional): Düğüm konumları için seed değeri.
                Aynı seed ile aynı görsel düzenleme elde edilir.
                Ağ yapısı değişmez, sadece layout değişir.
        
        Returns:
            tuple: (G, pos, success)
                G (nx.Graph): Yüklenen graf
                pos (dict): Düğüm pozisyonları
                success (bool): Başarı durumu
        
        Not:
            - CSV ayırıcı: noktalı virgül (;)
            - Ondalık ayırıcı: virgül veya nokta (otomatik dönüşüm)
            - UTF-8-SIG encoding (BOM karakteri desteği)
            - İlk satır başlık olarak atlanır
            - Seed sadece görsel düzenlemeyi etkiler, ağ yapısını değil!
        """
        try:
            self.G = nx.Graph()  # Yönsüz graf başlat
            
            # ===== 1. DÜĞÜM DOSYASINI OKU =====
            with open(node_file, 'r', encoding='utf-8-sig') as f:
                lines = f.readlines()
                header = lines[0].strip().split(';')  # Başlık satırı
                
                # Her düğüm satırını işle
                for line in lines[1:]:
                    if not line.strip(): 
                        continue  # Boş satırları atla
                    
                    parts = line.strip().split(';')
                    
                    # CSV formatı: node_id; processing_delay_ms; reliability_node
                    node_id = int(parts[0])
                    proc_delay = float(parts[1].replace(',', '.'))  # Virgülü noktaya çevir
                    reliability = float(parts[2].replace(',', '.'))
                    
                    # Düğümü grafiğe ekle
                    self.G.add_node(
                        node_id, 
                        proc_delay=proc_delay, 
                        reliability=reliability
                    )
                    
            # ===== 2. KENAR DOSYASINI OKU =====
            with open(edge_file, 'r', encoding='utf-8-sig') as f:
                lines = f.readlines()
                header = lines[0].strip().split(';')  # Başlık satırı
                
                # Her kenar satırını işle
                for line in lines[1:]:
                    if not line.strip(): 
                        continue  # Boş satırları atla
                    
                    parts = line.strip().split(';')
                    
                    # CSV formatı: src; dst; capacity_mbps; delay_ms; reliability_link
                    u = int(parts[0])
                    v = int(parts[1])
                    bw = float(parts[2].replace(',', '.'))      # Bant genişliği (Mbps)
                    delay = float(parts[3].replace(',', '.'))   # Gecikme (ms)
                    rel = float(parts[4].replace(',', '.'))     # Güvenilirlik (0-1)
                    
                    # Kenarı grafiğe ekle
                    self.G.add_edge(
                        u, v, 
                        bandwidth=bw, 
                        link_delay=delay, 
                        reliability=rel,
                        type='custom'  # CSV'den yüklenen özel tip
                    )
            
            # ===== 3. SON İŞLEMLER =====
            # Bağlantı garantisi kontrolü (opsiyonel)
            if not nx.is_connected(self.G):
                # Eğer graf kopuksa en büyük bileşeni al veya bağla
                # (Şimdilik pas geçildi)
                pass
            
            # Görselleştirme için pozisyonları ayarla (seed kullanarak)
            # Aynı seed ile aynı görsel düzenleme, ağ yapısı aynı kalır
            layout_seed = seed if seed is not None else 42
            self.pos = nx.spring_layout(self.G, seed=layout_seed)
            self.num_nodes = len(self.G.nodes())
            
            return self.G, self.pos, True  # Başarılı
            
        except Exception as e:
            # Dosya bulunamadı, format hatası vb.
            print(f"CSV okuma hatası: {e}")
            return None, None, False       
                       