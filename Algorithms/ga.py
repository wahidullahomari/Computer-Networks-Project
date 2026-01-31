"""
===================================================================================
                        GENETIC ALGORITHM FOR QoS ROUTING
===================================================================================
Bu modül, Genetik Algoritma (GA) kullanarak çoklu QoS kısıtlı rota optimizasyonu
gerçekleştirir. Evrimsel hesaplama prensipleriyle (seçilim, çaprazlama, mutasyon)
optimal ağ rotalarını bulur.

Temel Özellikler:
------------------
1. Bant genişliği kısıtı ile başlangıç popülasyonu oluşturma
2. Ağırlıklı fitness fonksiyonu (gecikme, güvenilirlik, kaynak)
3. Turnuva seçilimi ve elit koruma stratejisi
4. İki noktalı çaprazlama (two-point crossover)
5. Rastgele mutasyon ile çeşitlilik sağlama

Kullanım:
---------
    ga = GenetikAlgorithm()
    best_path = ga.genetik_calistir(G, source=0, target=10, bandwidth=200, weights)
    metrics = ga.get_path_metrics(best_path, bandwidth_demand=200)


"""

import networkx as nx  # Ağ grafiği işlemleri
import random          # Rastgele yol üretimi ve mutasyon

# Modül içi yardımcı fonksiyonlar
try:
    from Algorithms.metrics import calculate_path_attributes, calculate_weighted_cost
    from Algorithms.network_generator import NetworkManager
except ModuleNotFoundError:
    # Algorithms/ klasörü dışından çalıştırılırsa
    from metrics import calculate_path_attributes, calculate_weighted_cost
    from network_generator import NetworkManager


class GenetikAlgorithm:
    """
    Genetik Algoritma Tabanlı QoS Rota Optimizasyonu
    
    Evrimsel hesaplama ilkeleriyle (seçilim, çaprazlama, mutasyon) ağ rotalarını
    optimize eder. Bant genişliği kısıtını sağlayan birey popülasyonundan başlayarak
    en iyi rotayı bulur.
    
    Attributes:
        G (nx.Graph): Ağ topolojisi grafiği
        s (int): Kaynak düğüm ID'si
        d (int): Hedef düğüm ID'si
        bw (float): Minimum bant genişliği kısıtı (Mbps)
        N (int): Popülasyon boyutu (varsayılan 50)
        max_gen (int): Maksimum jenerasyon sayısı (varsayılan 200)
        pc (float): Çaprazlama olasılığı (varsayılan 0.8)
        pm (float): Mutasyon olasılığı (varsayılan 0.08)
        elite (int): Sonraki nesle aktarılacak elit birey sayısı (varsayılan 2)
        weights (dict): QoS ağırlıkları {'w_delay', 'w_reliability', 'w_resource'}
        population (list): Mevcut popülasyondaki rotalar (bireyler)
    
    Evrim Süreci:
        1. Başlangıç popülasyonu oluştur (bant genişliği kısıtını sağlayan)
        2. Fitness değerlerini hesapla (düşük = iyi)
        3. Elite bireyleri koru
        4. Turnuva seçilimi ile ebeveyn seç
        5. Çaprazlama ve mutasyon uygula
        6. Yeni nesil oluştur
        7. max_gen kadar tekrarla
    """

    def __init__(self):
        """Genetik Algoritma parametrelerini başlat"""
        self.G = None        # Ağ yapısı (NetworkX graf)
        self.s = None        # Başlangıç düğümü
        self.d = None        # Hedef düğümü
        self.bw = None       # Bant genişliği kısıtı (Mbps)

        # GA Hiperparametreleri
        self.N = 50          # Popülasyon boyutu
        self.max_gen = 200   # Jenerasyon sayısı
        self.pc = 0.8        # Crossover olasılığı
        self.pm = 0.08       # Mutasyon olasılığı
        self.elite = 2       # Elitizm sayısı
        self.weights = None  # QoS ağırlıkları (dışarıdan gelecek)

        self.population = []  # Mevcut popülasyon (birey listesi)

    def baslangic_popilasyonu(self):
        """
        Bant genişliği kısıtını sağlayan başlangıç popülasyonu oluşturur.
        
        Algoritma:
            1. Ağdan bant genişliği kısıtını sağlamayan kenarları kaldır
            2. Kalan grafta rastgele yollar üret
            3. N adet benzersiz yol bulunana kadar devam et
        
        Raises:
            Exception: Uygun yol bulunamazsa
        """
        print("Kısıtlara göre başlangıç popülasyonu oluşturuluyor...")
        if self.G is None:
            print("Hata: Ağ yapısı tanımlı değil!")
            return
        
        # Ağın kopyasını al (orijinal grafiği bozmamak için)
        G_copy = self.G.copy()

        # Bant genişliği kısıtını sağlamayan kenarları kaldır
        for u, v, data in list(G_copy.edges(data=True)):
            if data.get('bandwidth', 0) < self.bw:
                G_copy.remove_edge(u, v)

        self.G = G_copy  # Filtrelenmiş grafiği kullan

        population = []
        max_attempts = self.N * 20  # Maksimum deneme sayısı
        attempts = 0

        # N adet benzersiz yol üretmeye çalış
        while len(population) < self.N and attempts < max_attempts:
            path = self.rastgele_yol(self.G, self.s, self.d)
            attempts += 1
            if path and path not in population:  # Benzersiz yol ekle
                population.append(path)

        # Hiç yol bulunamadıysa hata fırlat
        if len(population) == 0:
            raise Exception("Başlangıç popülasyonu oluşturulamadı (bant genişliği kısıtı çok yüksek olabilir)")
        
        print(f"Popülasyon oluşturuldu: {len(population)} birey")
        self.population = population

    def rastgele_yol(self, G, s, d, max_len=30):
        """
        Depth-First Search ile rastgele yol üretir.
        
        Args:
            G (nx.Graph): Graf
            s (int): Kaynak düğüm
            d (int): Hedef düğüm
            max_len (int): Maksimum yol uzunluğu (döngüyü engelle)
        
        Returns:
            list: Düğüm listesi [s, ..., d] veya None (yol yoksa)
        """
        stack = [(s, [s])]  # (mevcut_düğüm, yol_listesi)
        
        while stack:
            node, path = stack.pop()

            # Maksimum uzunluğa ulaşıldıysa atla
            if len(path) > max_len:
                continue

            # Hedefe ulaşıldıysa yolu döndür
            if node == d:
                return path

            # Komşuları rastgele sırayla keşfet
            neighbors = list(G.neighbors(node))
            random.shuffle(neighbors)

            for neighbor in neighbors:
                if neighbor not in path:  # Döngü engelleme
                    stack.append((neighbor, path + [neighbor]))
        
        return None  # Yol bulunamadı

    def rastgele_yol2(self, G, s, d):
        stack = [(s, [s])]

        while stack:
            node, path = stack.pop()

            if node == d:
                return path

            neighbors = list(G.neighbors(node))
            random.shuffle(neighbors)

            for neighbor in neighbors:
                if neighbor not in path:  # döngü önler
                    stack.append((neighbor, path + [neighbor]))

        return None

    def mutasyon(self, path):
        # Path çok kısa ise mutasyon anlamsız
        if len(path) < 3:
            return path

        # Baştaki ve sondaki düğüm koruma
        idx = random.randint(1, len(path) - 2)

        sol = path[idx - 1]
        sag = path[idx + 1]

        # Yeni alt yol bul
        yeni_alt_yol = self.rastgele_yol(self.G, sol, sag)

        # Eğer yol bulunamazsa eski path'i döndür
        if yeni_alt_yol is None:
            return path

        # sol ve sag zaten path'te olduğu için tekrarları önle
        # [sol, ..., sag] → sol ve sag hariç
        yeni_alt_yol = yeni_alt_yol[1:-1]

        # Yeni path oluştur
        yeni_path = path[:idx] + yeni_alt_yol + path[idx + 1:]

        return yeni_path

    def caprazlama(self, parent1, parent2):
        # Ortak ARA düğümleri bul (başlangıç ve bitiş hariç)
        ortak_dugumler = list(set(parent1[1:-1]) & set(parent2[1:-1]))

        # Ortak düğüm yoksa çaprazlama yapılamaz en iyi bireyi aktar
        if not ortak_dugumler:
            best = min(parent1, parent2, key=self.fitness).copy()
            return best

        # Rastgele ortak düğüm seç
        ortak = random.choice(ortak_dugumler)

        # Parent'lardaki indeksler
        i1 = parent1.index(ortak)
        i2 = parent2.index(ortak)

        # Child path
        child = parent1[:i1 + 1] + parent2[i2 + 1:]

        return child

    def fitness(self, path):
        g = self.G
        a = calculate_path_attributes(g, path)
        return calculate_weighted_cost(a, self.weights)

    def turnuva_secimi(self, k=3):
        adaylar = random.sample(self.population, k)
        best = min(adaylar, key=self.fitness)
        return best

    def genetik_algoritması(self):

        yeni_popilasyon = []

        elites = sorted(self.population,
                        key=self.fitness)  # elitizm en iyi fitness değerine sahip ilk self.elite adet bireyi yeni popilasyona aktarır
        yeni_popilasyon.extend(elites[:self.elite])

        while (len(yeni_popilasyon) < self.N):  # yeni jenerasyon oluşturuluyor
            p1 = self.turnuva_secimi()
            p2 = self.turnuva_secimi()

            if random.random() < self.pc:
                child = self.caprazlama(p1, p2)
            else:
                child = min(p1, p2, key=self.fitness).copy()  # eğer kopyalamaz isek parrent değişeceği için ga bozulur

            if random.random() < self.pm:
                child = self.mutasyon(child)

            yeni_popilasyon.append(child)

        self.population = yeni_popilasyon

    def genetik_calistir(self, G, s, d, bw, weights):
        print("Genetik algoritması çalıştırıldı")
        self.G = G
        self.s = s
        self.d = d
        self.bw = bw
        self.weights = weights

        self.baslangic_popilasyonu()
        best_path = min(self.population, key=self.fitness)

        for _ in range(self.max_gen):
            self.genetik_algoritması()
            curr = min(self.population, key=self.fitness)
            if self.fitness(curr) < self.fitness(best_path):
                best_path = curr

        return best_path  # en iyi bireyi döndür4

    from typing import List, Dict, Any, Optional

    def get_path_metrics(
            self,
            path: Optional[List[int]],
            demand_bw: float = 0
    ) -> Dict[str, Any]:
        """
        GA tarafından bulunan yol için QoS metriklerini hesaplar.
        """

        if not path or len(path) < 2:
            return {"valid": False, "error": "Geçerli bir yol bulunamadı"}

        g = self.G
        
        if g is None:
            return {"valid": False, "error": "Ağ yapısı tanımlı değil"}

        # Yol öznitelikleri (delay, reliability_cost, resource_cost vs.)
        attrs = calculate_path_attributes(g, path)

        # Toplam güvenilirlik (kenar + düğüm, çarpımsal)
        total_reliability = 1.0
        for i in range(len(path) - 1):
            total_reliability *= g.edges[
                path[i], path[i + 1]
            ].get("reliability", 1.0)

        for n in path:
            total_reliability *= g.nodes[n].get("reliability", 1.0)

        # Darboğaz bant genişliği
        min_bw = min(
            g.edges[path[i], path[i + 1]].get("bandwidth", float("inf"))
            for i in range(len(path) - 1)
        )

        # Bant genişliği kısıtı kontrolü
        if min_bw < demand_bw:
            return {"valid": False, "error": "Bant genişliği kısıtı sağlanmadı"}

        return {
            "valid": True,
            "path": path,
            "total_delay": attrs["total_delay"],
            "total_reliability": total_reliability,
            "reliability_cost": attrs["reliability_cost"],
            "resource_cost": attrs["resource_cost"],
            "total_cost": calculate_weighted_cost(attrs, self.weights),
            "bottleneck_bw": min_bw,
            "hop_count": len(path) - 1,
            "node_count": len(path)
        }


if __name__ == "__main__":
    print("program çalıştırıldı")
    ga = GenetikAlgorithm()
    nm = NetworkManager()

    # 1. TEST: Rastgele Mod
    G_rnd, d_rnd = nm.generate_random(num_nodes=250)
    print(G_rnd)
    weights = {
        "w_delay": 0.5,
        "w_reliability": 0.3,
        "w_resource": 0.2
    }
    best_path = ga.genetik_calistir(G_rnd, 8, 40, 200, weights)
    print("en iyi yol:")
    print(best_path)
