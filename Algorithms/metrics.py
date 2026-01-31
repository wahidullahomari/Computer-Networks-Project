import networkx as nx
import math

def calculate_path_attributes(graph, path):
    """
    Bir yolun ham özelliklerini (toplam gecikme, darboğaz, toplam güvenilirlik) hesaplar.
    Geriye sözlük (dictionary) döndürür.
    """
    total_delay = 0
    reliability_cost = 0
    resource_cost = 0
   
    # 1. Kenar (Link) özelliklerini topla
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge_data = graph.get_edge_data(u, v)
       
        if edge_data:
            # Gecikme (Toplam)
            total_delay += edge_data.get('link_delay', 0)

            # Güvenilirlik (Hocanın istediği gibi -log)
            r_link = edge_data.get('reliability', 1.0)
            if r_link <= 0.000001:
                r_link = 0.000001
            reliability_cost += -math.log(r_link)

            # Bant Genişliği (Kaynak Kullanımı)
            bw = edge_data.get('bandwidth', 1)
            if bw <= 0:
                bw = 1
            resource_cost += (1000 / bw)

    # 2. Düğüm (Node) özelliklerini topla (S ve D HARİÇ)
    for node in path[1:-1]:
        node_data = graph.nodes[node]

        # İşlem süresi (Processing Delay)
        total_delay += node_data.get('proc_delay', 0)
    
    # 3. Düğüm (Node) güvenilirliklerini topla (Tüm yol için)
    for node in path:
        node_data = graph.nodes[node]

        # Eğer düğüm güvenilirliği varsa onu da ekle (-log)
        r_node = node_data.get('reliability', 1.0)
        if r_node <= 0.000001:
            r_node = 0.000001
        reliability_cost += -math.log(r_node)
    
    return {
        'total_delay': total_delay,
        'reliability_cost': reliability_cost,
        'resource_cost': resource_cost
    }


def calculate_weighted_cost(path_attributes, weights):
    """
    Arayüzden gelen 0-1 arasındaki ağırlıklara göre tek bir 'Maliyet' (Cost) puanı hesaplar.
    Algoritmalar (GA, Q-Learning) bu değeri MINIMIZE etmeye çalışacak.
   
    weights: {'w_delay': 0.5, 'w_reliability': 0.3, 'w_resource': 0.2} gibi bir sözlük.
    """
    # 1. Değerleri Al
    delay = path_attributes['total_delay']
    reliability_cost = path_attributes['reliability_cost']
    resource_cost = path_attributes['resource_cost']
   
    # 2. Ağırlıkları Al (Yoksa varsayılan 0 olsun)
    w_d = weights.get('w_delay', 0)        # Gecikme ağırlığı
    w_r = weights.get('w_reliability', 0)  # Güvenilirlik ağırlığı
    w_res = weights.get('w_resource', 0)   # Bant genişliği ağırlığı

    # 3. COST (Maliyet) Hesaplama Mantığı
    # Amaç: Cost'u düşürmek.
    # - Gecikme artarsa -> Cost artmalı (+)
    # - Güvenilirlik azalırsa -> Cost artmalı (-log kullanılır)
    # - Bant genişliği azalırsa -> Cost artmalı (1/BW)
   
    term_delay = w_d * delay
    term_reliability = w_r * reliability_cost
    term_bandwidth = w_res * resource_cost
   
    final_cost = term_delay + term_reliability + term_bandwidth
    return final_cost
