# -*- coding: utf-8 -*-
"""
===================================================================================
QoS-INTELLIGENCE: AKILLI ROTALAMA SİMÜLATÖRÜ
===================================================================================
Proje: BSM307/317 Güz 2025 Dönem Projesi - Bilgisayar Ağları QoS Rotalama

Bu uygulama, ağ topolojilerinde QoS (Quality of Service) gereksinimlerini
karşılayan optimal yolları bulmak için çeşitli AI ve optimizasyon algoritmalarını
kullanır ve performanslarını karşılaştırır.

Desteklenen Algoritmalar:
- Q-Learning (Reinforcement Learning)
- PSO (Particle Swarm Optimization)  
- Genetik Algoritma (Evolutionary Algorithm)
- Benzetimli Tavlama (Simulated Annealing)

Geliştirme: Engin Tekşüt ve Ekibi, Aralık 2025
===================================================================================
"""
import sys
import io
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QComboBox, QSlider, 
                             QGroupBox, QFormLayout, QFrame, QGraphicsDropShadowEffect, 
                             QSizePolicy, QStyle, QLineEdit, QFileDialog, QTextEdit, QSplitter,
                             QStackedWidget, QMenuBar, QAction, QCheckBox, QScrollArea, QGridLayout, QTabWidget,
                             QDialog, QDialogButtonBox, QDoubleSpinBox, QSpinBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QSize
from PyQt5.QtGui import QFont, QColor, QIntValidator, QTextCursor, QIcon, QTextCursor

# --- HARİCİ MODÜL ENTEGRASYONU ---
from topology import TopologyManager  # Ağ topolojisi ve algoritma yöneticisi

# =============================================================================
# YARDIMCI SINIFLAR
# =============================================================================

class EmittingStream(QObject):
    """
    Konsol Çıktı Yönlendirici
    
    Python'un print ve error çıktılarını yakalar ve Qt sinyali olarak
    GUI'ye iletir. Bu sayede konsol mesajları arayüzde görüntülenebilir.
    """
    text_written = pyqtSignal(str)

    def write(self, text):
        if text.strip():  # Boş satırları atla
            self.text_written.emit(str(text))
    
    def flush(self):
        pass

# =============================================================================
# 1. CSS STİL ŞABLONLARI
# =============================================================================

COMMON_CSS = """
QWidget { font-family: 'Segoe UI', sans-serif; }

/* Panel Kartları (Sol Menü) */
QFrame#PanelCard { 
    border-radius: 10px; 
    border-left: 4px solid; 
}
QLabel#PanelTitle { font-size: 13px; font-weight: bold; margin-bottom: 2px; text-transform: uppercase; letter-spacing: 0.5px; }

/* Dashboard Metrik Kartları (Alt Kısım - BÜYÜTÜLDÜ) */
QFrame#MetricCard { border-radius: 12px; }
QLabel#MetricTitle { font-size: 11px; font-weight: bold; letter-spacing: 1px; opacity: 0.8; }
QLabel#MetricValue { font-size: 28px; font-weight: bold; } /* Font büyüdü */

/* Grafik Çerçevesi */
QFrame#GraphContainer { 
    border-radius: 12px; 
    border: 2px solid; 
}

/* Log Konsolu */
QTextEdit#LogConsole {
    background-color: #1a1a2e;
    color: #00ff00;
    border: 1px solid #313244;
    border-radius: 8px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 13px;
    padding: 8px;
}

/* Grafik Başlığı (Sağ Üst Header - Sol ile Eşitlendi) */
QLabel#GraphHeader {
    border-radius: 6px; 
    padding: 12px; 
    font-size: 16px; 
    font-weight: bold;
    margin-bottom: 5px;
}
"""

# --- KARANLIK MOD ---
DARK_THEME = COMMON_CSS + """
QMainWindow, QWidget { background-color: #1e1e2e; color: #cdd6f4; }

/* Kartlar */
QFrame#PanelCard { background-color: #262636; border: 1px solid #313244; border-left-color: #89b4fa; }
QLabel#PanelTitle { color: #89b4fa; }

/* Headerlar (Hem Sol Hem Sağ) */
QLabel#HeaderLabel, QLabel#GraphHeader { 
    background-color: #313244; color: #ffffff; 
    border: 1px solid #45475a;
}
/* Sağ Header Rota Hesaplandığında Renklensin diye özel ID */
QLabel#GraphHeader[active="true"] { color: #fab387; border-color: #fab387; }

/* Form Elemanları */
QComboBox { background-color: #1e1e2e; border: 1px solid #45475a; border-radius: 5px; padding: 5px; color: #cdd6f4; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox::down-arrow { image: none; border-left: 2px solid #45475a; } 
QComboBox QAbstractItemView { background-color: #1e1e2e; color: #cdd6f4; selection-background-color: #45475a; }

QSlider::groove:horizontal { background: #313244; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal { background: #89b4fa; width: 14px; height: 14px; margin: -4px 0; border-radius: 7px; }

/* Grafik Alanı */
QFrame#GraphContainer { background-color: #232330; border-color: #313244; } 

/* Metrik Kartları */
QFrame#MetricCard { background-color: #262636; border: 1px solid #313244; }
QLabel#MetricTitle { color: #a6adc8; }
QLabel#MetricValue { color: #ffffff; }

/* Butonlar */
QPushButton { border-radius: 6px; font-weight: bold; font-size: 13px; }
QPushButton#CalcBtn { background-color: #a6e3a1; color: #1e1e2e; border: none; }
QPushButton#CalcBtn:hover { background-color: #94e2d5; }
QPushButton#ResetBtn { background-color: #fab387; color: #1e1e2e; border: none; }
QPushButton#ResetBtn:hover { background-color: #f9e2af; }
QPushButton#ThemeBtn { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; font-size: 14px; }
"""

# --- AYDINLIK MOD ---
LIGHT_THEME = COMMON_CSS + """
QMainWindow, QWidget { background-color: #f3f4f6; color: #374151; }

/* Kartlar */
QFrame#PanelCard { background-color: #ffffff; border: 1px solid #e5e7eb; border-left-color: #3b82f6; }
QLabel#PanelTitle { color: #3b82f6; }

/* Headerlar */
QLabel#HeaderLabel, QLabel#GraphHeader { 
    background-color: #ffffff; color: #111827; 
    border: 1px solid #d1d5db;
}
QLabel#GraphHeader[active="true"] { color: #ef4444; border-color: #ef4444; }

/* Form Elemanları */
QComboBox { background-color: #ffffff; border: 1px solid #d1d5db; border-radius: 5px; padding: 5px; color: #1f2937; }
QComboBox QAbstractItemView { background-color: white; color: black; selection-background-color: #bfdbfe; }

QSlider::groove:horizontal { background: #e5e7eb; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal { background: #3b82f6; width: 14px; height: 14px; margin: -4px 0; border-radius: 7px; }

/* Grafik Alanı */
QFrame#GraphContainer { background-color: #ffffff; border-color: #e5e7eb; }

/* Log Konsolu */
QTextEdit#LogConsole {
    background-color: #f8f9fa;
    color: #2d3748;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 13px;
    padding: 8px;
}

/* Metrik Kartları */
QFrame#MetricCard { background-color: #ffffff; border: 1px solid #e5e7eb; }
QLabel#MetricTitle { color: #6b7280; }
QLabel#MetricValue { color: #111827; }

/* Butonlar */
QPushButton { border-radius: 6px; font-weight: bold; font-size: 13px; }
QPushButton#CalcBtn { background-color: #10b981; color: white; border: none; }
QPushButton#CalcBtn:hover { background-color: #059669; }
QPushButton#ResetBtn { background-color: #f59e0b; color: white; border: none; }
QPushButton#ResetBtn:hover { background-color: #d97706; }
QPushButton#ThemeBtn { background-color: #ffffff; color: #374151; border: 1px solid #d1d5db; font-size: 14px; }
"""

# =============================================================================
# 2. GRAFİK TUVALİ (NETWORK CANVAS)
# =============================================================================
class NetworkCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        
        # --- DÜZELTME: GRAFİK KENAR BOŞLUKLARINI SIFIRLA ---
        # Bu işlem grafiğin çerçeveye tam oturmasını sağlar
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        super(NetworkCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.ax.axis('off') 
        self.set_theme_colors('dark')
        
        # Otomatik boyutlandırma için size policy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()
    
    def resizeEvent(self, event):
        """Pencere boyutlandırıldığında canvas'ı güncelle"""
        super().resizeEvent(event)
        self.fig.tight_layout(pad=0)
        self.draw_idle()  # draw() yerine draw_idle() thread-safe

    def set_theme_colors(self, mode):
        if mode == 'dark':
            self.bg_color = '#232330' 
            self.node_color = '#cba6f7' 
            self.edge_color = '#45475a' 
            self.glow_color = '#89b4fa' 
            self.text_color = '#ffffff' # Etiket rengi
        else:
            self.bg_color = '#ffffff' 
            self.node_color = '#3b82f6' 
            self.edge_color = '#9ca3af'
            self.glow_color = '#f97316' 
            self.text_color = '#000000'
            
        self.fig.patch.set_facecolor(self.bg_color)
        self.ax.set_facecolor(self.bg_color)
        
    def draw_network(self, G, pos, path_list=None):
        self.ax.clear()
        if not G: return

        if not path_list:
            # --- NORMAL DURUM ---
            nx.draw_networkx_nodes(G, pos, ax=self.ax, node_size=35, node_color=self.node_color, alpha=0.8)
            nx.draw_networkx_edges(G, pos, ax=self.ax, width=0.5, alpha=0.3, edge_color=self.edge_color)
        else:
            # --- ROTA MODU ---
            # Arka planı silikleştir
            nx.draw_networkx_nodes(G, pos, ax=self.ax, node_size=30, node_color='gray', alpha=0.05)
            nx.draw_networkx_edges(G, pos, ax=self.ax, width=0.5, alpha=0.02, edge_color='gray')
            
            # Rota Çizgileri
            path_edges = list(zip(path_list, path_list[1:]))
            
            # Glow Efekti (Dengeli)
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, ax=self.ax, width=6, alpha=0.15, edge_color=self.glow_color)
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, ax=self.ax, width=2, alpha=1.0, edge_color='white') # Çekirdek Beyaz
            
            # Ara Noktalar
            nx.draw_networkx_nodes(G, pos, nodelist=path_list[1:-1], ax=self.ax, node_size=180, node_color='white', edgecolors=self.glow_color, linewidths=2)
            
            # Başlangıç/Bitiş Noktaları (Daha Büyük)
            nx.draw_networkx_nodes(G, pos, nodelist=[path_list[0]], ax=self.ax, node_size=250, node_color='#00ff00', alpha=1.0, edgecolors='white', linewidths=2) 
            nx.draw_networkx_nodes(G, pos, nodelist=[path_list[-1]], ax=self.ax, node_size=250, node_color='#ff0000', alpha=1.0, edgecolors='white', linewidths=2)

            # --- YENİ ÖZELLİK: DÜĞÜM NUMARALARI ---
            # Sadece rota üzerindeki noktaların numaralarını yazalım
            labels = {node: str(node) for node in path_list}
            nx.draw_networkx_labels(G, pos, labels=labels, ax=self.ax, font_size=8, font_color='black', font_weight='bold')

        self.ax.set_aspect('equal')
        self.draw_idle()  # Thread-safe çizim

# =============================================================================
# 3. ANA PENCERE (MAIN WINDOW)
# =============================================================================
# ANA UYGULAMA PENCERESİ
# =============================================================================

class MainWindow(QMainWindow):
    """
    Ana Uygulama Sınıfı
    
    QoS-Intelligence uygulamasının ana penceresi. 4 ekranlı bir yapıya sahiptir:
    1. Ana Ekran: Ağ görselleştirme ve tek rota hesaplama
    2. Karşılaştırma: Birden fazla algoritmayı yan yana test etme
    3. Raporlar: Detaylı analiz ve istatistikler
    4. Konsol: Sistem logları ve mesajları
    
    Özellikler:
    -----------
    - Çoklu algoritma desteği (Q-Learning, PSO, GA, SA)
    - Dinamik parametre ayarlama
    - Karanlık/Aydınlık tema desteği
    - CSV dosyadan ağ yükleme
    - Toplu test (30 senaryo)
    - Performans karşılaştırma ve raporlama
    """
    
    def __init__(self):
        """
        Pencere Başlatıcı
        
        Tüm UI bileşenlerini, değişkenleri ve event handler'ları initialize eder.
        """
        super().__init__()
        self.setWindowTitle("QoS-Intelligence: Akıllı Rotalama Simülatörü")
        self.setGeometry(100, 100, 1600, 1000)  # Pencere boyutu ve konumu
        
        # Tema ayarları
        self.is_dark_mode = True
        
        # Ağ durumu değişkenleri
        self.current_path = None  # Hesaplanan son rota
        self.network_pos = None  # NetworkX düğüm pozisyonları (Qt.pos() ile karışmaması için)
        self.comparison_results = []  # Karşılaştırma ekranı sonuçları
        self.batch_test_results = []  # Toplu test sonuçları (30 senaryo)
        self.current_seed = None  # Ağ topolojisi seed değeri (tekrarlanabilirlik için)
        
        # Algoritma parametreleri - Her algoritma için varsayılan değerler
        self.algo_params = {
            'qlearning': {
                'alpha': 0.1,       # Öğrenme oranı
                'gamma': 0.9,       # İskonto faktörü
                'epsilon': 0.9,     # Keşif oranı
                'episodes': 500     # Eğitim episode sayısı
            },
            'pso': {
                'swarm_size': 30,   # Parçacık sayısı
                'iterations': 25,   # İterasyon sayısı
                'w': 0.7,           # Atalet ağırlığı
                'c1': 1.5,          # Bilişsel katsayı
                'c2': 2.0           # Sosyal katsayı
            },
            'genetic': {
                'population': 50,   # Popülasyon boyutu
                'generations': 200, # Nesil sayısı
                'crossover': 0.8,   # Çaprazlama oranı
                'mutation': 0.08    # Mutasyon oranı
            },
            'sa': {
                'initial_temp': 1000,    # Başlangıç sıcaklığı
                'cooling_rate': 0.95,    # Soğuma oranı
                'iterations': 500        # İterasyon sayısı
            }
        }
        
        # Animasyon için timer
        self.hue = 0  # Renk tonı (HSL)
        self.timer_anim = QTimer()
        self.timer_anim.timeout.connect(self.animate_border)
        self.timer_anim.start(50)  # 50ms aralıkla güncelle
        
        # Stdout/stderr yönlendirme (log_console oluşturulduktan sonra aktif edilecek)
        # self.stdout_stream = EmittingStream()
        # self.stdout_stream.text_written.connect(self.append_log)
        # sys.stdout = self.stdout_stream
        # sys.stderr = self.stdout_stream
        
        self.manager = TopologyManager()
        
        # İlk ağ oluşturma - Varsayılan seed ile (42)
        self.current_seed = 42
        self.G, self.network_pos = self.manager.create_network(seed=self.current_seed)

        self.init_ui()
        self.setup_stdout_redirect()  # UI oluştuktan sonra redirect aktifleştir
        self.apply_theme()
    
    def resizeEvent(self, event):
        """Pencere boyutlandırıldığında widget'ları güncelle"""
        super().resizeEvent(event)
        # Canvas'ı güncelle
        if hasattr(self, 'canvas'):
            self.canvas.updateGeometry()
            self.canvas.draw_idle()
    
    def showEvent(self, event):
        """Pencere gösterildiğinde layout'ları güncelle"""
        super().showEvent(event)
        # İlk gösterimde layout'ları zorla güncelle
        if hasattr(self, 'canvas'):
            QTimer.singleShot(100, lambda: self.canvas.draw_idle())
    
    # ===== SEED MANAGEMENT =====
    
    def on_seed_checkbox_changed(self, state):
        """Seed checkbox durumu değiştiğinde input'u aktif/pasif yap"""
        is_checked = state == 2  # Qt.CheckState.Checked
        self.seed_input.setEnabled(is_checked)
        
        if is_checked:
            seed_value = self.seed_input.value()
            self.lbl_seed_info.setText(f"✅ Seed aktif: {seed_value}")
            self.lbl_seed_info.setStyleSheet("color: #10b981; font-size: 10px; font-style: italic;")
        else:
            self.lbl_seed_info.setText("ℹ️ Seed kapalı (Varsayılan mod)")
            self.lbl_seed_info.setStyleSheet("color: #6b7280; font-size: 10px; font-style: italic;")
    
    def on_seed_changed(self, value):
        """Seed değeri değiştiğinde bilgi etiketini güncelle"""
        self.current_seed = value
        if self.seed_checkbox.isChecked():
            self.lbl_seed_info.setText(f"✅ Seed aktif: {value}")
            self.lbl_seed_info.setStyleSheet("color: #10b981; font-size: 10px; font-style: italic;")
    
    def create_network_with_seed(self):
        """
        Kullanıcının girdiği seed değeri ile ağ oluşturur.
        
        İki mod:
        1. CSV yüklüyse: Aynı CSV'yi yeni seed ile tekrar yükler (layout değişir)
        2. CSV yoksa: Rastgele ağ oluşturur (250 düğüm)
        """
        try:
            seed_value = self.seed_input.value()
            self.current_seed = seed_value
            
            # CSV yüklü mü kontrol et
            if hasattr(self, 'csv_loaded') and self.csv_loaded and hasattr(self, 'csv_node_file'):
                # CSV MOD: Aynı CSV'yi farklı seed ile yükle
                G, pos, success = self.manager.build_from_csv(
                    self.csv_node_file, 
                    self.csv_edge_file, 
                    seed=seed_value
                )
                
                if success and G is not None:
                    self.G = G
                    self.network_pos = pos
                    
                    node_count = len(G.nodes())
                    edge_count = len(G.edges())
                    
                    # Header güncelle
                    if hasattr(self, 'lbl_graph_header'):
                        self.lbl_graph_header.setText(f"CSV (SEED: {seed_value}): {node_count} DÜĞÜM / {edge_count} KENAR")
                    
                    # Bilgi etiketi
                    self.lbl_seed_info.setText(f"✅ Seed: {seed_value} (CSV aktif)")
                    self.lbl_seed_info.setStyleSheet("color: #10b981; font-size: 10px; font-style: italic;")
                    
                    # Konsol log
                    try:
                        self.append_console(f"[CSV+SEED] Aynı ağ, farklı layout (Seed: {seed_value})", "info")
                    except:
                        pass
                else:
                    raise Exception("CSV tekrar yüklenemedi")
            
            else:
                # RASTGELE MOD: Yeni ağ oluştur
                self.G, self.network_pos = self.manager.create_network(seed=seed_value)
                
                node_count = len(self.G.nodes())
                edge_count = len(self.G.edges())
                
                # Header güncelle
                if hasattr(self, 'lbl_graph_header'):
                    self.lbl_graph_header.setText(f"AĞ OLUŞTURULDU (SEED: {seed_value}): {node_count} DÜĞÜM / {edge_count} KENAR")
                
                # Bilgi etiketi
                self.lbl_seed_info.setText(f"✅ Seed: {seed_value} (Aktif)")
                self.lbl_seed_info.setStyleSheet("color: #10b981; font-size: 10px; font-style: italic;")
                
                # Konsol log
                try:
                    self.append_console(f"[SİSTEM] Rastgele ağ oluşturuldu (Seed: {seed_value}, Düğüm: {node_count})", "success")
                except:
                    pass
            
            # Ortak işlemler
            # Combobox'ları güncelle
            if hasattr(self, 'combo_source') and hasattr(self, 'combo_target'):
                self.populate_combos()
            
            # Grafiği çiz
            if hasattr(self, 'canvas'):
                self.canvas.draw_network(self.G, self.network_pos)
            
            # Metrikleri sıfırla
            self.current_path = None
            if hasattr(self, 'card_delay'):
                self.card_delay.findChild(QLabel, "MetricValue").setText("-")
                self.card_rel.findChild(QLabel, "MetricValue").setText("-")
                self.card_res.findChild(QLabel, "MetricValue").setText("-")
                
        except Exception as e:
            # Hata durumunda kullanıcıya bilgi ver
            self.lbl_seed_info.setText(f"❌ Hata: {str(e)[:30]}")
            self.lbl_seed_info.setStyleSheet("color: #ef4444; font-size: 10px; font-style: italic;")
            print(f"Seed ağ oluşturma hatası: {e}")
        
    # ===== FILE OPERATIONS =====
        
    def load_graph_from_file(self):
        """
        CSV dosyalarını yükler ve arayüzü günceller.
        Dosya yolları kaydedilir, böylece seed ile tekrar yüklenebilir.
        """
        
        # 1. NODE Dosyası Seçimi
        fname_node, _ = QFileDialog.getOpenFileName(self, '1. Adım: NodeData (Düğüm) Dosyası', '.', "CSV Files (*.csv);;All Files (*)")
        if not fname_node: return

        # 2. EDGE Dosyası Seçimi
        fname_edge, _ = QFileDialog.getOpenFileName(self, '2. Adım: EdgeData (Kenar) Dosyası', '.', "CSV Files (*.csv);;All Files (*)")
        if not fname_edge: return

        # Dosya yollarını kaydet (seed ile tekrar yüklemek için)
        self.csv_node_file = fname_node
        self.csv_edge_file = fname_edge
        self.csv_loaded = True

        # 3. Yükleme İşlemi (mevcut seed ile)
        current_seed = self.seed_input.value() if hasattr(self, 'seed_input') else None
        G, pos, success = self.manager.build_from_csv(fname_node, fname_edge, seed=current_seed)
        
        if success:
            self.G = G
            self.network_pos = pos
            self.current_path = None
            self.current_seed = current_seed  # Seed'i kaydet
            
            # Comboboxları yeni düğüm sayısına göre güncelle
            self.populate_combos()
            
            # Header Bilgisini Güncelle
            if self.G is not None:
                node_count = len(self.G.nodes())
                edge_count = len(self.G.edges())
                seed_text = f" (SEED: {current_seed})" if current_seed else ""
                self.lbl_graph_header.setText(f"CSV YÜKLENDİ{seed_text}: {node_count} DÜĞÜM / {edge_count} KENAR")
            
            # Seed bilgisini güncelle
            if hasattr(self, 'lbl_seed_info') and current_seed:
                self.lbl_seed_info.setText(f"✅ Seed: {current_seed} (CSV ile aktif)")
                self.lbl_seed_info.setStyleSheet("color: #10b981; font-size: 10px; font-style: italic;")
            
            # Grafiği Çiz
            self.canvas.draw_network(self.G, self.network_pos)
            
            # Konsola log
            try:
                self.append_console(f"[CSV] Dosyalar yüklendi: {node_count} düğüm, {edge_count} kenar (Seed: {current_seed})", "success")
            except:
                pass
            
            # Kartları Sıfırla
            self.card_delay.findChild(QLabel, "MetricValue").setText("-")
            self.card_rel.findChild(QLabel, "MetricValue").setText("-")
            self.card_res.findChild(QLabel, "MetricValue").setText("-")
        else:
            self.lbl_graph_header.setText("HATA: DOSYA OKUNAMADI!")    

    def init_ui(self):
        # Üst Menü Barı
        self.create_menu_bar()
        
        # Merkezi Widget - Stacked Widget ile ekranlar arası geçiş
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # 4 Ana Ekran Oluştur
        self.main_screen = self.create_main_screen()
        self.comparison_screen = self.create_comparison_screen()
        self.reports_screen = self.create_reports_screen()
        self.console_screen = self.create_console_screen()
        
        # Ekranları Stack'e Ekle
        self.stacked_widget.addWidget(self.main_screen)  # Index 0
        self.stacked_widget.addWidget(self.comparison_screen)  # Index 1
        self.stacked_widget.addWidget(self.reports_screen)  # Index 2
        self.stacked_widget.addWidget(self.console_screen)  # Index 3
        
        # Başlangıçta ana ekran
        self.stacked_widget.setCurrentIndex(0)
    
    def create_menu_bar(self):
        """Modern üst menü barı oluştur"""
        menubar = self.menuBar()
        if menubar is None:
            return
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #1a1a2e;
                color: #ffffff;
                border-bottom: 2px solid #3b82f6;
                padding: 2px;
                font-size: 13px;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 4px 15px;
                border-radius: 4px;
            }
            QMenuBar::item:selected {
                background-color: #3b82f6;
            }
            QMenuBar::item:pressed {
                background-color: #2563eb;
            }
        """)
        
        # Ana Ekran
        home_action = QAction("🏠 Ana Ekran", self)
        home_action.triggered.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        menubar.addAction(home_action)
        
        # Karşılaştırma
        compare_action = QAction("🔄 Karşılaştırma", self)
        compare_action.triggered.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        menubar.addAction(compare_action)
        
        # Raporlar
        reports_action = QAction("📈 Raporlar", self)
        reports_action.triggered.connect(self.open_reports)
        menubar.addAction(reports_action)
        
        # Konsol
        console_action = QAction("💻 Konsol", self)
        console_action.triggered.connect(lambda: self.stacked_widget.setCurrentIndex(3))
        menubar.addAction(console_action)
    
    def open_reports(self):
        """Raporlar ekranını aç (karşılaştırma yapılmış mı kontrol et)"""
        if not self.comparison_results:
            self.log_console.insertPlainText("\n⚠️ Önce karşılaştırma yapmalısınız!\n")
            self.log_console.moveCursor(QTextCursor.MoveOperation.End)
            return
        self.update_reports_screen()
        self.stacked_widget.setCurrentIndex(2)
    
    def create_main_screen(self):
        """Mevcut ana ekranı oluştur"""
        main_widget = QWidget()
        central_widget = main_widget
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(25) # Paneller arası boşluk
        central_widget.setLayout(main_layout)
       

        # --- SOL PANEL (KONTROLLER) ---
        left_panel = QWidget()
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(380)
        left_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        left_layout = QVBoxLayout()
        left_layout.setSpacing(12) # Kartlar arası boşluk azaltıldı (20→12)
        left_layout.setContentsMargins(10, 5, 10, 10) # Üst margin azaltıldı
        left_panel.setLayout(left_layout)
        
        # 1. Header (Kontrol Paneli) - Kompakt
        self.header = QLabel("KONTROL PANELİ")
        self.header.setObjectName("HeaderLabel")
        self.header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.header.setStyleSheet("font-size: 16px; font-weight: bold; padding: 3px;")  # Daha kompakt
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setOffset(0, 3)
        self.header.setGraphicsEffect(shadow)
        left_layout.addWidget(self.header)

        # --- DOSYA YÜKLEME KARTI (KOMPAKT) ---
        card_file = self.create_input_card("📁 CSV Dosyaları")
        layout_file = QVBoxLayout()
        layout_file.setContentsMargins(0, 3, 0, 3)
        
        self.btn_load_files = QPushButton("📂 Yükle")
        self.btn_load_files.setFixedHeight(32)
        self.btn_load_files.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_load_files.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6; 
                color: white; 
                border: none; 
                border-radius: 5px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #2563eb; }
        """)
        self.btn_load_files.clicked.connect(self.load_graph_from_file)
        
        layout_file.addWidget(self.btn_load_files)

        card_file.content_layout.addLayout(layout_file)
        left_layout.addWidget(card_file)
        # ---------------------------------------------

        # 2. Rota Seçimi KARTI (KOMPAKT)
        card_route = self.create_input_card("📍 Rota Seçimi")
        layout_route = QFormLayout()
        layout_route.setSpacing(6) # Inputlar arası boşluk daha da azaltıldı
        layout_route.setContentsMargins(0,0,0,0)
        layout_route.setVerticalSpacing(6)
        layout_route.setHorizontalSpacing(5)
        
        # 1. Algoritma Seçim Kutusu
        self.combo_algo = QComboBox()
        self.combo_algo.addItems(["PSO (Meta-Heuristic)", "Q-Learning (AI)", "Genetik Algoritma","Benzetimli Tavlama (SA)"])
        layout_route.addRow("Algoritma:", self.combo_algo)
        
        # Diğer Özellikler Butonu
        self.btn_algo_params = QPushButton("⚙️ Parametreler")
        self.btn_algo_params.setFixedHeight(26)
        self.btn_algo_params.setStyleSheet("""
            QPushButton {
                background-color: #6366f1; 
                color: white; 
                border: none; 
                border-radius: 4px;
                font-weight: bold;
                font-size: 10px;
                padding: 3px;
            }
            QPushButton:hover { background-color: #4f46e5; }
        """)
        self.btn_algo_params.clicked.connect(self.show_algo_params_dialog)
        layout_route.addRow("", self.btn_algo_params)

        # 2. Bant Genişliği Talebi (Demand)
        self.input_demand = QLineEdit()
        self.input_demand.setPlaceholderText("Örn: 100")
        self.input_demand.setText("100") # Varsayılan değer
        self.input_demand.setValidator(QIntValidator(1, 1000))
        layout_route.addRow("Talep (Mbps):", self.input_demand)
        
        # --- YENİLİK: Editable (Yazılabilir) Combobox ve Validator ---
        self.combo_source = QComboBox()
        self.combo_source.setEditable(True) # Yazılabilir
        self.combo_source.setValidator(QIntValidator(0, 249)) # Sadece 0-249 arası sayı
        
        self.combo_target = QComboBox()
        self.combo_target.setEditable(True)
        self.combo_target.setValidator(QIntValidator(0, 249))
        
        self.populate_combos()
        layout_route.addRow("Kaynak (S):", self.combo_source)
        layout_route.addRow("Hedef (D):", self.combo_target)
        
        card_route.content_layout.addLayout(layout_route)
        left_layout.addWidget(card_route)

        # 2.5 SEED AYARI (ALGORİTMA İÇİN)
        card_seed = self.create_input_card("🎲 Seed (Tekrarlanabilirlik)")
        layout_seed = QVBoxLayout()
        layout_seed.setSpacing(5)
        layout_seed.setContentsMargins(0, 3, 0, 3)
        
        # Seed checkbox (Aktif/Pasif)
        self.seed_checkbox = QCheckBox("Seed Kullan")
        self.seed_checkbox.setChecked(False)  # Varsayılan kapalı
        self.seed_checkbox.setToolTip("✅ İşaretli: Algoritma belirtilen seed kullanır (aynı seed → aynı yol)\n❌ İşaretsiz: Her çalıştırmada farklı yol bulunur")
        self.seed_checkbox.stateChanged.connect(self.on_seed_checkbox_changed)
        self.seed_checkbox.setStyleSheet("""
            QCheckBox {
                font-weight: bold;
                font-size: 10px;
                color: #3b82f6;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)
        
        # Seed input (0-9999 arası) - Başlangıçta pasif
        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 9999)
        self.seed_input.setValue(42)  # Varsayılan seed
        self.seed_input.setPrefix("Seed: ")
        self.seed_input.setToolTip("Algoritmanın rastgele kararlarını kontrol eder.\nAynı seed → Aynı yol bulunur.")
        self.seed_input.setEnabled(False)  # Başlangıçta pasif
        self.seed_input.valueChanged.connect(self.on_seed_changed)
        
        layout_seed.addWidget(self.seed_checkbox)
        layout_seed.addWidget(self.seed_input)
        
        # Seed bilgi etiketi
        self.lbl_seed_info = QLabel("ℹ️ Seed kapalı (Varsayılan mod)")
        self.lbl_seed_info.setStyleSheet("color: #6b7280; font-size: 10px; font-style: italic;")
        self.lbl_seed_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_seed.addWidget(self.lbl_seed_info)
        
        card_seed.content_layout.addLayout(layout_seed)
        left_layout.addWidget(card_seed)

        # 3. QoS Ağırlıkları (KOMPAKT)
        card_qos = self.create_input_card("⚖️ QoS Ağırlıkları")
        layout_qos = QVBoxLayout()
        layout_qos.setSpacing(5) 
        layout_qos.setContentsMargins(0,0,0,0)
        
        self.lbl_delay = QLabel("Gecikme (Hız): %33")
        self.slider_delay = QSlider(Qt.Orientation.Horizontal)
        self.slider_delay.setRange(0, 100)
        self.slider_delay.setValue(33)
        self.slider_delay.valueChanged.connect(self.update_ui_labels)
        
        self.lbl_rel = QLabel("Güvenilirlik (Sağlamlık): %33")
        self.slider_rel = QSlider(Qt.Orientation.Horizontal)
        self.slider_rel.setRange(0, 100)
        self.slider_rel.setValue(33)
        self.slider_rel.valueChanged.connect(self.update_ui_labels)

        self.lbl_res = QLabel("Kaynak (Maliyet): %34")
        self.slider_res = QSlider(Qt.Orientation.Horizontal)
        self.slider_res.setRange(0, 100)
        self.slider_res.setValue(34)
        self.slider_res.valueChanged.connect(self.update_ui_labels)

        layout_qos.addWidget(self.lbl_delay)
        layout_qos.addWidget(self.slider_delay)
        layout_qos.addWidget(self.lbl_rel)
        layout_qos.addWidget(self.slider_rel)
        layout_qos.addWidget(self.lbl_res)
        layout_qos.addWidget(self.slider_res)
        card_qos.content_layout.addLayout(layout_qos)
        left_layout.addWidget(card_qos)

        # 4. Butonlar (Daha Büyük)
        self.btn_calc = QPushButton("▶ ROTA HESAPLA")
        self.btn_calc.setObjectName("CalcBtn")
        self.btn_calc.setFixedHeight(50)
        self.btn_calc.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_calc.clicked.connect(self.on_calculate_click)
        left_layout.addWidget(self.btn_calc)

        self.btn_reset = QPushButton("↺ SİSTEMİ SIFIRLA")
        self.btn_reset.setObjectName("ResetBtn")
        self.btn_reset.setFixedHeight(45)
        self.btn_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reset.clicked.connect(self.reset_application)
        left_layout.addWidget(self.btn_reset)
        
        # 5. Mod Butonu (BÜYÜTÜLDÜ)
        self.btn_theme = QPushButton("Mod: Karanlık 🌙")
        self.btn_theme.setObjectName("ThemeBtn")
        self.btn_theme.setFixedHeight(45) # Yükseklik artırıldı
        self.btn_theme.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_theme.clicked.connect(self.toggle_theme)
        left_layout.addWidget(self.btn_theme)
        
        # Sol alttaki boşluğu doldurması için stretch
        left_layout.addStretch()

        lbl_sign = QLabel("Interface Designed by Yusuf MEYDAN")
        lbl_sign.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_sign.setStyleSheet("color: gray; font-size: 10px;")
        left_layout.addWidget(lbl_sign)
        
        main_layout.addWidget(left_panel)

        # --- SAĞ PANEL ---
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15) # Dikey boşluk
        right_panel.setLayout(right_layout)

        # Grafik Başlığı (OPTİMİZE EDİLDİ)
        self.lbl_graph_header = QLabel("AĞ TOPOLOJİSİ - BEKLEMEDE")
        self.lbl_graph_header.setObjectName("GraphHeader")
        self.lbl_graph_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_graph_header.setStyleSheet("font-size: 18px; font-weight: bold;")  # Font optimize edildi
        # Buna da aynı gölgeyi verelim
        shadow_g = QGraphicsDropShadowEffect()
        shadow_g.setBlurRadius(20)
        shadow_g.setOffset(0, 5)
        self.lbl_graph_header.setGraphicsEffect(shadow_g)
        right_layout.addWidget(self.lbl_graph_header)

        # Grafik Çerçevesi
        graph_container = QFrame()
        graph_container.setObjectName("GraphContainer")
        graph_layout = QVBoxLayout()
        # --- DÜZELTME: GRAFİK ÇERÇEVESİ İÇ BOŞLUĞU SIFIRLANDI ---
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.setSpacing(0)
        graph_container.setLayout(graph_layout)
        
        self.canvas = NetworkCanvas(self, width=5, height=4, dpi=100)
        graph_layout.addWidget(self.canvas)
        
        right_layout.addWidget(graph_container, stretch=1) # Grafik için maksimum yer
        
        # Log console değişkenini oluştur (görünmez, sadece Konsol ekranı için)
        self.log_console = QTextEdit()
        self.log_console.setObjectName("LogConsole")
        self.log_console.setReadOnly(True)
        doc = self.log_console.document()
        if doc is not None:
            doc.setMaximumBlockCount(1000)  # Maksimum 1000 satır - performans

        # Dashboard Alanı (OPTİMİZE EDİLDİ)
        dashboard_widget = QWidget()
        dashboard_widget.setMinimumHeight(100)
        dashboard_widget.setMaximumHeight(130)  # Dinamik yükseklik
        self.dash_layout = QHBoxLayout()
        self.dash_layout.setContentsMargins(0, 0, 0, 0)
        self.dash_layout.setSpacing(15)  # Spacing azaltıldı
        dashboard_widget.setLayout(self.dash_layout)
        
        self.card_delay = self.create_metric_card("⏱️ GECİKME", "-", "#ff6b6b")
        self.card_rel = self.create_metric_card("🛡️ GÜVENİLİRLİK", "-", "#1dd1a1")
        self.card_res = self.create_metric_card("💰 MALİYET", "-", "#feca57")
        
        self.dash_layout.addWidget(self.card_delay)
        self.dash_layout.addWidget(self.card_rel)
        self.dash_layout.addWidget(self.card_res)
        
        right_layout.addWidget(dashboard_widget, stretch=1)

        main_layout.addWidget(right_panel)
        
        self.canvas.draw_network(self.G, self.network_pos, path_list=self.current_path)
        
        return main_widget

    # --- YARDIMCI FONKSİYONLAR ---
    def append_log(self, text):
        """Konsola log mesajı ekler"""
        self.log_console.moveCursor(QTextCursor.MoveOperation.End)
        self.log_console.insertPlainText(text)
        self.log_console.moveCursor(QTextCursor.MoveOperation.End)
    
    def animate_border(self):
        self.hue = (self.hue + 5) % 360
        color = QColor.fromHsl(self.hue, 200, 150)
        rgb = color.name()
        
        # Sadece border-color'ı değiştir
        base_style = self.header.styleSheet() # Mevcut stili alamayız, tekrar tanımlamalıyız.
        
        # Temaya göre baz stil
        bg = "#313244" if self.is_dark_mode else "#ffffff"
        fg = "#ffffff" if self.is_dark_mode else "#111827"
        
        new_style = f"""
            QLabel#HeaderLabel {{
                background-color: {bg}; color: {fg};
                border-radius: 6px; padding: 12px; font-size: 16px; font-weight: bold; 
                border: 3px solid {rgb};
            }}
        """
        self.header.setStyleSheet(new_style)

    def create_input_card(self, title):
        card = QFrame()
        card.setObjectName("PanelCard")
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 12, 10, 12) # Daha sıkı boşluklar
        lbl_title = QLabel(title)
        lbl_title.setObjectName("PanelTitle")
        layout.addWidget(lbl_title)
        card.setLayout(layout)
        card.content_layout = layout  # Layout'u sakla
        return card

    def create_metric_card(self, title, value, accent_color):
        """Metrik kartı oluştur (Optimize edilmiş layout)"""
        card = QFrame()
        card.setObjectName("MetricCard")
        card.setMinimumHeight(80)  # Minimum yükseklik
        card.setMaximumHeight(110)  # Maksimum yükseklik
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setOffset(0, 3)
        shadow.setColor(QColor(0, 0, 0, 40))
        card.setGraphicsEffect(shadow)
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(5)  # Elementler arası boşluk
        layout.setContentsMargins(10, 10, 10, 10)
        
        lbl_title = QLabel(title)
        lbl_title.setObjectName("MetricTitle")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("font-size: 12px; font-weight: bold;")  # Font optimize
        
        lbl_value = QLabel(value)
        lbl_value.setObjectName("MetricValue")
        lbl_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        line = QFrame()
        line.setFixedWidth(50)
        line.setFixedHeight(3)  # Çizgi inceldi
        line.setStyleSheet(f"background-color: {accent_color}; border: none; border-radius: 2px;")
        
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_value)
        layout.addWidget(line, 0, Qt.AlignmentFlag.AlignCenter)
        card.setLayout(layout)
        return card

    def highlight_cards(self):
        if self.is_dark_mode:
            flash_style = "QFrame#MetricCard { background-color: #45475a; border: 2px solid #b4befe; }"
        else:
            flash_style = "QFrame#MetricCard { background-color: #d1d5db; border: 2px solid #6b7280; }"
        self.card_delay.setStyleSheet(flash_style)
        self.card_rel.setStyleSheet(flash_style)
        self.card_res.setStyleSheet(flash_style)
        QTimer.singleShot(250, lambda: self.apply_theme_to_cards())

    def apply_theme_to_cards(self):
        self.card_delay.setStyleSheet("")
        self.card_rel.setStyleSheet("")
        self.card_res.setStyleSheet("")

    def populate_combos(self):
        nodes = [str(i) for i in range(250)]
        self.combo_source.clear()
        self.combo_target.clear()
        self.combo_source.addItems(nodes)
        self.combo_target.addItems(nodes)
        self.combo_target.setCurrentIndex(249)

    def update_ui_labels(self):
        self.lbl_delay.setText(f"Gecikme (Hız): %{self.slider_delay.value()}")
        self.lbl_rel.setText(f"Güvenilirlik (Sağlamlık): %{self.slider_rel.value()}")
        self.lbl_res.setText(f"Kaynak (Maliyet): %{self.slider_res.value()}")

    def toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        self.apply_theme()
        
    def setup_stdout_redirect(self):
        """UI oluştuktan sonra stdout redirect'i aktifleştir"""
        self.stdout_stream = EmittingStream()
        self.stdout_stream.text_written.connect(self.append_log)
        sys.stdout = self.stdout_stream
        sys.stderr = self.stdout_stream
    
    def apply_theme(self):
        if self.is_dark_mode:
            self.setStyleSheet(DARK_THEME)
            self.canvas.set_theme_colors('dark')
            self.btn_theme.setText("Mod: Karanlık 🌙")
        else:
            self.setStyleSheet(LIGHT_THEME)
            self.canvas.set_theme_colors('light')
            self.btn_theme.setText("Mod: Aydınlık ☀️")
        
        # Sağ Header'ın stilini güncelle (aktiflik durumunu koru)
        is_active = "true" if self.current_path else "false"
        self.lbl_graph_header.setProperty("active", is_active)
        style = self.lbl_graph_header.style()
        if style:
            style.unpolish(self.lbl_graph_header)
            style.polish(self.lbl_graph_header)
        
        self.canvas.draw_network(self.G, self.network_pos, path_list=self.current_path)
        self.canvas.updateGeometry()  # Layout'u güncelle

    def reset_application(self):
        self.slider_delay.setValue(33)
        self.slider_rel.setValue(33)
        self.slider_res.setValue(34)
        self.current_path = None 
        self.card_delay.findChild(QLabel, "MetricValue").setText("-")
        self.card_rel.findChild(QLabel, "MetricValue").setText("-")
        self.card_res.findChild(QLabel, "MetricValue").setText("-")
        
        self.lbl_graph_header.setText("AĞ TOPOLOJİSİ - SIFIRLANDI")
        self.lbl_graph_header.setProperty("active", "false")
        style = self.lbl_graph_header.style()
        if style:
            style.unpolish(self.lbl_graph_header)
            style.polish(self.lbl_graph_header)
        
        self.G, self.network_pos = self.manager.create_network()
        self.canvas.draw_network(self.G, self.network_pos)
        
        # Konsolu temizle
        self.log_console.clear()

    def on_calculate_click(self):
        """Rota Hesapla butonuna basıldığında:
        1. CSV yüklü değilse hata göster
        2. Seed checkbox işaretliyse: CSV'yi seed ile yükle
        3. Seed checkbox işaretsizse: CSV'yi varsayılan seed ile yükle
        4. Rotayı hesapla
        """
        # 1. CSV yüklü mü kontrol et
        if not hasattr(self, 'csv_loaded') or not self.csv_loaded:
            self.lbl_graph_header.setText("⚠️ HATA: ÖNCE CSV DOSYALARİNİ YÜKLEYİN!")
            try:
                self.append_console("[HATA] CSV dosyaları yüklenmeden rota hesaplanamaz!", "error")
            except:
                pass
            return
        
        # 2. CSV'yi yükle (Görsel düzenleme sabit, seed sadece algoritmaları etkiler)
        try:
            # Sabit görsel düzenleme (layout) - Ağ yapısı değişmez
            G, pos, success = self.manager.build_from_csv(
                self.csv_node_file, 
                self.csv_edge_file, 
                seed=42  # Görsel için sabit seed
            )
            if success:
                self.G = G
                self.network_pos = pos
                self.canvas.draw_network(self.G, self.network_pos)
                try:
                    self.append_console(f"[CSV] Ağ yüklendi - Rota hesaplamaya hazır", "success")
                except:
                    pass
        except Exception as e:
            self.lbl_graph_header.setText(f"⚠️ HATA: CSV YÜKLENEMEDİ ({str(e)[:20]})")
            return
        
        # 3. Kaynak/Hedef ve parametreleri al
        try:
            s = int(self.combo_source.currentText())
            t = int(self.combo_target.currentText())
            selected_algo = self.combo_algo.currentText()
            demand_text = self.input_demand.text()
            demand = int(float(demand_text)) if demand_text else 100
            
        except ValueError:
             self.lbl_graph_header.setText("HATA: GEÇERSİZ GİRİŞ!")
             return

        w_d = self.slider_delay.value() / 100.0
        w_r = self.slider_rel.value() / 100.0
        w_res = self.slider_res.value() / 100.0
        
        # Seçili algoritmanın parametrelerini al
        algo_index = self.combo_algo.currentIndex()
        algo_map = {0: 'pso', 1: 'qlearning', 2: 'genetic', 3: 'sa'}
        algo_key = algo_map.get(algo_index, 'pso')
        algo_params = self.algo_params[algo_key]
        
        # Seed parametresini ekle (checkbox işaretliyse)
        if self.seed_checkbox.isChecked():
            algo_params['seed'] = self.seed_input.value()
        else:
            algo_params['seed'] = None  # Rastgele davranış

        # 4. Rotayı hesapla (seed dahil)
        result = self.manager.calculate_path(s,t,w_d,w_r,w_res,algorithm=selected_algo,demand_value=demand,algo_params=algo_params)
        if result:
            self.current_path = result["path"]
            self.card_delay.findChild(QLabel, "MetricValue").setText(f"{result['total_delay']:.2f} ms")
            self.card_rel.findChild(QLabel, "MetricValue").setText(f"%{result['final_reliability']:.4f}")
            self.card_res.findChild(QLabel, "MetricValue").setText(f"{result['resource_cost']:.2f}")
            self.highlight_cards()
            
            # Seed bilgisini header'a ekle
            seed_info = f" [SEED: {self.seed_input.value()}]" if self.seed_checkbox.isChecked() else ""
            self.lbl_graph_header.setText(f"ROTA HESAPLANDI{seed_info}: DÜĞÜM {s} ➝ DÜĞÜM {t}")
            self.lbl_graph_header.setProperty("active", "true")
            style = self.lbl_graph_header.style()
            if style:
                style.unpolish(self.lbl_graph_header)
                style.polish(self.lbl_graph_header)
            
            self.canvas.draw_network(self.G, self.network_pos, path_list=self.current_path)
        else:
            self.lbl_graph_header.setText("HATA: UYGUN YOL BULUNAMADI!")
            self.canvas.draw_idle()
    
    def create_comparison_screen(self):
        """Karşılaştırma ekranı - Çoklu algoritma karşılaştırması"""
        screen = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        screen.setLayout(layout)
        
        # Başlık
        title = QLabel("🔄 Algoritma Karşılaştırması")
        title.setStyleSheet("font-size: 32px; font-weight: bold; color: #3b82f6; margin-bottom: 10px;")
        title.setWordWrap(True)
        layout.addWidget(title)
        
        subtitle = QLabel("Meta-sezgisel algoritmaların karmaşık ağ topolojisi üzerindeki yol bulma performansını analiz edin.")
        subtitle.setStyleSheet("font-size: 16px; color: #6b7280; margin-bottom: 20px;")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        
        # İçerik Alanı (Sol: Ayarlar, Sağ: Sonuçlar)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        
        # --- SOL PANEL: AYARLAR (Scroll Area içinde) ---
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(500)
        left_scroll.setMaximumWidth(600)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        left_container = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setSpacing(20)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_container.setLayout(left_layout)
        
        # 1. Performans Hedefleri Kartı
        goals_card = QFrame()
        goals_card.setStyleSheet("""
            QFrame {
                background: #262636; 
                border-radius: 12px; 
                padding: 20px; 
                border: 1px solid #313244;
                margin: 0px;
            }
        """)
        goals_layout = QVBoxLayout()
        goals_layout.setSpacing(12)
        goals_layout.setContentsMargins(15, 15, 15, 15)
        goals_card.setLayout(goals_layout)
        
        goals_title = QLabel("🎯 1. Performans Hedefleri")
        goals_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #89b4fa; margin-bottom: 10px;")
        goals_layout.addWidget(goals_title)
        
        goals_desc = QLabel("En iyi yol tanımını etkileyecek faktörlerin ağırlıklarını belirleyin. Toplam ağırık %100 olmak zorunda değildir; değerler normalize edilecektir.")
        goals_desc.setWordWrap(True)
        goals_desc.setStyleSheet("font-size: 13px; color: #a6adc8; margin-bottom: 10px;")
        goals_layout.addWidget(goals_desc)
        
        # Ağırlık Önizleme Butonları
        weight_preview_layout = QHBoxLayout()
        
        btn_speed = QPushButton("Hız")
        btn_speed.setStyleSheet("background: #44475a; color: white; border: none; padding: 12px; border-radius: 6px; font-size: 14px; font-weight: bold;")
        btn_speed.clicked.connect(lambda: self.set_weight_preset(80, 10, 10))
        weight_preview_layout.addWidget(btn_speed)
        
        btn_reliable = QPushButton("Güvenilir")
        btn_reliable.setStyleSheet("background: #44475a; color: white; border: none; padding: 12px; border-radius: 6px; font-size: 14px; font-weight: bold;")
        btn_reliable.clicked.connect(lambda: self.set_weight_preset(10, 80, 10))
        weight_preview_layout.addWidget(btn_reliable)
        
        btn_balanced = QPushButton("Dengeli")
        btn_balanced.setStyleSheet("background: #44475a; color: white; border: none; padding: 12px; border-radius: 6px; font-size: 14px; font-weight: bold;")
        btn_balanced.clicked.connect(lambda: self.set_weight_preset(33, 33, 34))
        weight_preview_layout.addWidget(btn_balanced)
        
        goals_layout.addLayout(weight_preview_layout)
        
        # Slider'lar
        self.comp_delay_label = QLabel("En Az Gecikme (Latency): 33%")
        self.comp_delay_label.setStyleSheet("font-size: 14px; color: #cdd6f4; margin-top: 10px; font-weight: bold;")
        goals_layout.addWidget(self.comp_delay_label)
        
        self.comp_delay_slider = QSlider(Qt.Orientation.Horizontal)
        self.comp_delay_slider.setRange(0, 100)
        self.comp_delay_slider.setValue(33)
        self.comp_delay_slider.setMinimumHeight(30)
        self.comp_delay_slider.valueChanged.connect(self.update_comparison_labels)
        goals_layout.addWidget(self.comp_delay_slider)
        
        self.comp_rel_label = QLabel("En Yüksek Güvenilirlik (Reliability): 33%")
        self.comp_rel_label.setStyleSheet("font-size: 14px; color: #cdd6f4; font-weight: bold;")
        goals_layout.addWidget(self.comp_rel_label)
        
        self.comp_rel_slider = QSlider(Qt.Orientation.Horizontal)
        self.comp_rel_slider.setRange(0, 100)
        self.comp_rel_slider.setValue(33)
        self.comp_rel_slider.valueChanged.connect(self.update_comparison_labels)
        goals_layout.addWidget(self.comp_rel_slider)
        
        self.comp_res_label = QLabel("En Az Kaynak Kullanımı (Cost): 34%")
        self.comp_res_label.setStyleSheet("font-size: 11px; color: #cdd6f4;")
        goals_layout.addWidget(self.comp_res_label)
        
        self.comp_res_slider = QSlider(Qt.Orientation.Horizontal)
        self.comp_res_slider.setRange(0, 100)
        self.comp_res_slider.setValue(34)
        self.comp_res_slider.valueChanged.connect(self.update_comparison_labels)
        goals_layout.addWidget(self.comp_res_slider)
        
        left_layout.addWidget(goals_card)
        
        # 2. Algoritma Seçimi Kartı
        algo_card = QFrame()
        algo_card.setStyleSheet("""
            QFrame {
                background: #262636; 
                border-radius: 12px; 
                padding: 20px; 
                border: 1px solid #313244;
                margin: 0px;
            }
        """)
        algo_layout = QVBoxLayout()
        algo_layout.setSpacing(10)
        algo_layout.setContentsMargins(15, 15, 15, 15)
        algo_card.setLayout(algo_layout)
        
        algo_title = QLabel("🔬 2. Algoritma Seçimi")
        algo_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #89b4fa; margin-bottom: 10px;")
        algo_layout.addWidget(algo_title)
        
        algo_desc = QLabel("Karşılaştırmak istediğiniz az iki algoritma seçin.")
        algo_desc.setStyleSheet("font-size: 13px; color: #a6adc8; margin-bottom: 10px;")
        algo_layout.addWidget(algo_desc)
        
        self.algo_checkboxes = {}
        algorithms = [
            ("Genetik Algoritma", "GA", "Evrimsel süreçleri taklit ederek global optimum arar."),
            ("Q-Learning (RL)", "QL", "Ajân tabanlı, canlı mekanizmalar ile öğrenerek yol bulur."),
            ("PSO (Meta-Heuristic)", "PSO", "Sürü davranışı benzer, çok sayıda şimdiki zaman tabanlı.", ),
            ("Benzetimli Tavlama (SA)", "SA", "Katıların soğuma sürecini taklit ederek yol bulur.")
        ]
        
        for name, code, desc in algorithms:
            algo_row_layout = QHBoxLayout()
            
            cb = QCheckBox(name)
            cb.setStyleSheet("font-size: 14px; color: #cdd6f4; padding: 8px; font-weight: bold;")
            cb.setChecked(True)  # Varsayılan olarak hepsi seçili
            self.algo_checkboxes[code] = cb
            algo_row_layout.addWidget(cb)
            
            # Diğer Özellikler butonu
            btn_params = QPushButton("⚙️")
            btn_params.setFixedSize(30, 30)
            btn_params.setStyleSheet("""
                QPushButton {
                    background-color: #6366f1; 
                    color: white; 
                    border: none; 
                    border-radius: 5px;
                    font-weight: bold;
                    font-size: 12px;
                }
                QPushButton:hover { background-color: #4f46e5; }
            """)
            btn_params.setToolTip("Diğer Özellikler")
            btn_params.clicked.connect(lambda checked, c=code: self.show_comparison_algo_params(c))
            algo_row_layout.addWidget(btn_params)
            algo_row_layout.addStretch()
            
            algo_layout.addLayout(algo_row_layout)
            
            desc_label = QLabel(f"    → {desc}")
            desc_label.setStyleSheet("font-size: 12px; color: #6b7280; margin-left: 25px; margin-bottom: 8px;")
            desc_label.setWordWrap(True)
            algo_layout.addWidget(desc_label)
        
        left_layout.addWidget(algo_card)
        
        # 3. Rota Parametreleri Kartı
        route_card = QFrame()
        route_card.setStyleSheet("""
            QFrame {
                background: #262636; 
                border-radius: 12px; 
                padding: 20px; 
                border: 1px solid #313244;
                margin: 0px;
            }
        """)
        route_layout = QFormLayout()
        route_layout.setSpacing(15)
        route_layout.setContentsMargins(15, 15, 15, 15)
        route_card.setLayout(route_layout)
        
        route_title = QLabel("📍 3. Rota Parametreleri")
        route_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #89b4fa; margin-bottom: 10px;")
        route_layout.addRow(route_title)
        
        # Font boyutunu artır için label wrapper
        source_label = QLabel("Kaynak (S):")
        source_label.setStyleSheet("font-size: 14px; color: #cdd6f4; font-weight: bold;")
        self.comp_source = QComboBox()
        self.comp_source.setEditable(True)
        self.comp_source.addItems([str(i) for i in range(250)])
        self.comp_source.setMinimumHeight(40)
        self.comp_source.setStyleSheet("background: #1e1e2e; border: 1px solid #45475a; padding: 8px; border-radius: 4px; color: #cdd6f4; font-size: 14px;")
        route_layout.addRow(source_label, self.comp_source)
        
        target_label = QLabel("Hedef (D):")
        target_label.setStyleSheet("font-size: 14px; color: #cdd6f4; font-weight: bold;")
        self.comp_target = QComboBox()
        self.comp_target.setEditable(True)
        self.comp_target.addItems([str(i) for i in range(250)])
        self.comp_target.setCurrentIndex(249)
        self.comp_target.setMinimumHeight(40)
        self.comp_target.setStyleSheet("background: #1e1e2e; border: 1px solid #45475a; padding: 8px; border-radius: 4px; color: #cdd6f4; font-size: 14px;")
        route_layout.addRow(target_label, self.comp_target)
        
        demand_label = QLabel("Talep (Mbps):")
        demand_label.setStyleSheet("font-size: 14px; color: #cdd6f4; font-weight: bold;")
        self.comp_demand = QLineEdit("100")
        self.comp_demand.setValidator(QIntValidator(1, 1000))
        self.comp_demand.setMinimumHeight(40)
        self.comp_demand.setStyleSheet("background: #1e1e2e; border: 1px solid #45475a; padding: 8px; border-radius: 4px; color: #cdd6f4; font-size: 14px;")
        route_layout.addRow(demand_label, self.comp_demand)
        
        left_layout.addWidget(route_card)
        
        # Analizi Başlat Butonu
        self.btn_run_comparison = QPushButton("▶ Analizi Başlat")
        self.btn_run_comparison.setObjectName("CompareBtn")
        self.btn_run_comparison.setMinimumHeight(60)
        self.btn_run_comparison.setStyleSheet("""
            QPushButton#CompareBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #2563eb);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton#CompareBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563eb, stop:1 #1d4ed8);
            }
        """)
        self.btn_run_comparison.clicked.connect(self.run_comparison)
        left_layout.addWidget(self.btn_run_comparison)
        
        # Toplu Test Butonu
        self.btn_batch_test = QPushButton("📊 Toplu Test Başlat (DemandData)")
        self.btn_batch_test.setObjectName("BatchBtn")
        self.btn_batch_test.setMinimumHeight(60)
        self.btn_batch_test.setStyleSheet("""
            QPushButton#BatchBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10b981, stop:1 #059669);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton#BatchBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #059669, stop:1 #047857);
            }
        """)
        self.btn_batch_test.clicked.connect(self.run_batch_test)
        left_layout.addWidget(self.btn_batch_test)
        
        left_layout.addStretch()
        
        # Scroll area'yı container'a bağla
        left_scroll.setWidget(left_container)
        content_layout.addWidget(left_scroll)
        
        # --- SAĞ PANEL: SONUÇLAR ---
        right_panel = QFrame()
        right_panel.setStyleSheet("background: #232330; border-radius: 10px; border: 1px solid #313244;")
        right_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        
        # Sonuç Başlığı
        result_header = QLabel("📊 Karşılaştırma Sonuçları")
        result_header.setStyleSheet("font-size: 22px; font-weight: bold; color: #ffffff; padding: 20px;")
        right_layout.addWidget(result_header)
        
        # Sonuç Tablosu
        self.comparison_results_widget = QWidget()
        self.comparison_results_layout = QVBoxLayout()
        self.comparison_results_widget.setLayout(self.comparison_results_layout)
        
        scroll = QScrollArea()
        scroll.setWidget(self.comparison_results_widget)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background: transparent;")
        right_layout.addWidget(scroll)
        
        content_layout.addWidget(right_panel, stretch=1)
        layout.addLayout(content_layout)
        
        return screen
    
    def create_console_screen(self):
        """Tam ekran konsol görünümü"""
        screen = QWidget()
        screen.setStyleSheet("background: #1e1e2e;")
        layout = QVBoxLayout()
        layout.setSpacing(3)  # Minimal boşluk
        layout.setContentsMargins(10, 2, 10, 10)  # Üst margin 2px - çok minimal
        screen.setLayout(layout)
        
        # Başlık - Kompakt
        title = QLabel("💻 Sistem Konsolu")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #10b981; margin: 0px; padding: 2px;")
        layout.addWidget(title)
        
        # Butonlar - Kompakt
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(6)
        buttons_layout.setContentsMargins(0, 0, 0, 0)  # Margin sıfır
        
        btn_clear = QPushButton("🗑️ Temizle")
        btn_clear.setFixedHeight(28)  # Sabit küçük yükseklik
        btn_clear.setStyleSheet("background: #ef4444; color: white; border: none; padding: 4px 12px; border-radius: 4px; font-size: 12px; font-weight: bold;")
        btn_clear.clicked.connect(lambda: self.log_console.clear())
        buttons_layout.addWidget(btn_clear)
        
        btn_copy = QPushButton("📋 Kopyala")
        btn_copy.setFixedHeight(28)
        btn_copy.setStyleSheet("background: #3b82f6; color: white; border: none; padding: 4px 12px; border-radius: 4px; font-size: 12px; font-weight: bold;")
        btn_copy.clicked.connect(lambda: self.copy_console_to_clipboard())
        buttons_layout.addWidget(btn_copy)
        
        btn_save = QPushButton("💾 Kaydet")
        btn_save.setFixedHeight(28)
        btn_save.setStyleSheet("background: #10b981; color: white; border: none; padding: 4px 12px; border-radius: 4px; font-size: 12px; font-weight: bold;")
        btn_save.clicked.connect(self.save_console_log)
        buttons_layout.addWidget(btn_save)
        
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)
        
        # Konsol yazı boyutunu büyüt ve tam ekran yap
        # setSizePolicy ile widget'i expand yaparak tüm alanı kaplasin
        self.log_console.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        doc = self.log_console.document()
        if doc is not None:
            doc.setMaximumBlockCount(5000)  # Konsol ekranında daha fazla satır
        
        # Konsol alanını maksimum yap - stretch factor ile tüm kalan alanı doldur
        layout.addWidget(self.log_console, 1)
        
        return screen
    
    def copy_console_to_clipboard(self):
        """Konsol içeriğini panoya kopyala"""
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self.log_console.toPlainText())
    
    def save_console_log(self):
        """Konsol logunu dosyaya kaydet"""
        filename, _ = QFileDialog.getSaveFileName(self, "Log Dosyası Kaydet", "", "Text Files (*.txt);;All Files (*)")
        if not filename:
            return
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.log_console.toPlainText())
            self.log_console.insertPlainText(f"\n✅ Log dosyası kaydedildi: {filename}\n")
            self.log_console.moveCursor(QTextCursor.MoveOperation.End)
        except Exception as e:
            self.log_console.insertPlainText(f"\n❌ Kaydetme hatası: {str(e)}\n")
            self.log_console.moveCursor(QTextCursor.MoveOperation.End)
    
    def create_reports_screen(self):
        """Raporlar ekranı - Grafiksel analiz ve sonuçlar"""
        screen = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        screen.setLayout(layout)
        
        # Başlık
        title = QLabel("📈 Simülasyon Raporları")
        title.setStyleSheet("font-size: 32px; font-weight: bold; color: #3b82f6; margin-bottom: 10px;")
        title.setWordWrap(True)
        layout.addWidget(title)
        
        subtitle = QLabel("Geçmiş simülasyonların performans analizlerini inceleyin, algoritmaları karşılaştırın ve sonuçları dışa aktarın.")
        subtitle.setStyleSheet("font-size: 16px; color: #6b7280; margin-bottom: 20px;")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        
        # Üst Butonlar
        top_buttons = QHBoxLayout()
        
        btn_filter = QPushButton("🔍 Filtrele")
        btn_filter.setMinimumHeight(45)
        btn_filter.setStyleSheet("background: #313244; color: white; border: none; padding: 12px 25px; border-radius: 8px; font-size: 15px; font-weight: bold;")
        top_buttons.addWidget(btn_filter)
        
        btn_pdf = QPushButton("📄 PDF İndir")
        btn_pdf.setMinimumHeight(45)
        btn_pdf.setStyleSheet("background: #313244; color: white; border: none; padding: 12px 25px; border-radius: 8px; font-size: 15px; font-weight: bold;")
        btn_pdf.clicked.connect(self.export_pdf_report)
        top_buttons.addWidget(btn_pdf)
        
        btn_csv = QPushButton("📊 CSV Dışa Aktar")
        btn_csv.setMinimumHeight(45)
        btn_csv.setStyleSheet("background: #3b82f6; color: white; border: none; padding: 12px 25px; border-radius: 8px; font-size: 15px; font-weight: bold;")
        btn_csv.clicked.connect(self.export_csv_report)
        top_buttons.addWidget(btn_csv)
        
        btn_refresh = QPushButton("🔄 Raporları Yenile")
        btn_refresh.setMinimumHeight(45)
        btn_refresh.setStyleSheet("background: #10b981; color: white; border: none; padding: 12px 25px; border-radius: 8px; font-size: 15px; font-weight: bold;")
        btn_refresh.clicked.connect(self.update_reports_screen)
        top_buttons.addWidget(btn_refresh)
        
        top_buttons.addStretch()
        layout.addLayout(top_buttons)
        
        # Rapor İçeriği (Tab Widget)
        self.report_tabs = QTabWidget()
        self.report_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #313244;
                border-radius: 8px;
                background: #232330;
            }
            QTabBar::tab {
                background: #262636;
                color: #cdd6f4;
                padding: 15px 30px;
                margin-right: 5px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #3b82f6;
                color: white;
            }
        """)
        
        # Tab 1: Özet Metr Kartları
        summary_tab = self.create_summary_tab()
        self.report_tabs.addTab(summary_tab, "📊 Özet")
        
        # Tab 2: Detaylı Karşılaştırma
        comparison_tab = self.create_comparison_tab()
        self.report_tabs.addTab(comparison_tab, "📈 Karşılaştırma")
        
        # Tab 3: Performans Grafiği
        performance_tab = self.create_performance_tab()
        self.report_tabs.addTab(performance_tab, "⚡ Performans")
        
        # Tab 4: Toplu Test Sonuçları
        batch_tab = self.create_batch_test_tab()
        self.report_tabs.addTab(batch_tab, "📦 Toplu Test")
        
        layout.addWidget(self.report_tabs)
        
        return screen
    
    def create_summary_tab(self):
        """Özet metrik kartları"""
        widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)
        widget.setLayout(main_layout)
        
        # SOL PANEL: Metrik Kartları (Dikey)
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(500)
        left_panel.setMinimumWidth(450)
        
        self.report_summary_widgets = {}
        
        # 4 Ana Metrik Kartı - Dikey olarak
        metrics = [
            ("fastest", "⚡ En Hızlı", "Algoritma", "-", "#10b981"),
            ("reliable", "🛡️ En Güvenilir", "Algoritma", "-", "#3b82f6"),
            ("efficient", "💰 En Verimli", "Algoritma", "-", "#f59e0b"),
            ("balanced", "⚖️ En Dengeli", "Algoritma", "-", "#8b5cf6")
        ]
        
        for key, title, label, value, color in metrics:
            card = self.create_report_metric_card(title, label, value, color)
            self.report_summary_widgets[key] = card
            left_layout.addWidget(card)
        
        left_layout.addStretch()
        main_layout.addWidget(left_panel)
        
        # SAĞ PANEL: Detaylı Tablo
        table_frame = QFrame()
        table_frame.setStyleSheet("background: #262636; border-radius: 12px; padding: 20px; border: 1px solid #313244;")
        table_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        table_layout = QVBoxLayout()
        table_layout.setContentsMargins(10, 10, 10, 10)
        table_frame.setLayout(table_layout)
        
        table_title = QLabel("📋 Detaylı Performans Karşılaştırması")
        table_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff; margin-bottom: 12px;")
        table_layout.addWidget(table_title)
        
        # Scroll area for table
        table_scroll = QScrollArea()
        table_scroll.setWidgetResizable(True)
        table_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        self.report_detail_table = QLabel("Henüz karşılaştırma yapılmadı.")
        self.report_detail_table.setStyleSheet("font-size: 14px; color: #a6adc8; padding: 15px;")
        self.report_detail_table.setWordWrap(True)
        from PyQt5.QtCore import Qt
        self.report_detail_table.setAlignment(Qt.AlignmentFlag(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft))
        
        table_scroll.setWidget(self.report_detail_table)
        table_layout.addWidget(table_scroll)
        
        main_layout.addWidget(table_frame, 1)
        
        return widget
    
    def create_comparison_tab(self):
        """Grafik karşılaştırma tab'ı"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Matplotlib grafik için canvas
        from matplotlib.figure import Figure
        self.comparison_figure = Figure(figsize=(8, 5), facecolor='#1e1e2e')
        self.comparison_canvas = FigureCanvas(self.comparison_figure)
        self.comparison_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.comparison_canvas.updateGeometry()
        layout.addWidget(self.comparison_canvas)
        
        return widget
    
    def create_performance_tab(self):
        """Performans çizgi grafiği tab'ı"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Matplotlib grafik için canvas
        from matplotlib.figure import Figure
        self.performance_figure = Figure(figsize=(8, 5), facecolor='#1e1e2e')
        self.performance_canvas = FigureCanvas(self.performance_figure)
        self.performance_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.performance_canvas.updateGeometry()
        layout.addWidget(self.performance_canvas)
        
        return widget
    
    def create_batch_test_tab(self):
        """Toplu test sonuçları tab'ı"""
        widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)
        widget.setLayout(main_layout)
        
        # SOL PANEL: Özet Kartları (Dikey)
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(500)
        left_panel.setMinimumWidth(450)
        
        # Başlık
        title = QLabel("📦 Toplu Test")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #10b981; margin-bottom: 10px;")
        left_layout.addWidget(title)
        
        desc = QLabel("DemandData.csv'den 30 senaryo")
        desc.setStyleSheet("font-size: 13px; color: #6b7280; margin-bottom: 10px;")
        left_layout.addWidget(desc)
        
        # 4 Özet Kartı - Dikey olarak
        self.batch_total_card = self.create_batch_stat_card("📊 Toplam Test", "0", "#3b82f6")
        self.batch_winner_card = self.create_batch_stat_card("🏆 En Başarılı", "-", "#10b981")
        self.batch_avg_time_card = self.create_batch_stat_card("⏱️ Ort. Süre", "0s", "#f59e0b")
        self.batch_success_card = self.create_batch_stat_card("✅ Başarı Oranı", "0%", "#8b5cf6")
        
        left_layout.addWidget(self.batch_total_card)
        left_layout.addWidget(self.batch_winner_card)
        left_layout.addWidget(self.batch_avg_time_card)
        left_layout.addWidget(self.batch_success_card)
        
        left_layout.addStretch()
        main_layout.addWidget(left_panel)
        
        # SAĞ PANEL: Detaylı Sonuçlar
        details_frame = QFrame()
        details_frame.setStyleSheet("background: #262636; border-radius: 12px; padding: 20px; border: 1px solid #313244;")
        details_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        details_layout = QVBoxLayout()
        details_layout.setContentsMargins(10, 10, 10, 10)
        details_frame.setLayout(details_layout)
        
        details_title = QLabel("📋 Senaryo Bazlı Detay")
        details_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff; margin-bottom: 12px;")
        details_layout.addWidget(details_title)
        
        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        self.batch_detail_table = QLabel("Henüz toplu test yapılmadı.\n\n'Karşılaştır' ekranından '📊 Toplu Test Başlat' butonuna tıklayın.")
        self.batch_detail_table.setStyleSheet("font-size: 14px; color: #a6adc8; padding: 15px;")
        self.batch_detail_table.setWordWrap(True)
        self.batch_detail_table.setAlignment(Qt.AlignmentFlag(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft))
        
        scroll.setWidget(self.batch_detail_table)
        details_layout.addWidget(scroll)
        
        main_layout.addWidget(details_frame, 1)
        
        return widget
    
    def create_batch_stat_card(self, title, value, color):
        """Toplu test istatistik kartı"""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background: #262636;
                border-radius: 10px;
                border-top: 4px solid {color};
                padding: 20px;
            }}
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(8)
        card.setLayout(layout)
        
        title_label = QLabel(title)
        title_label.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {color};")
        layout.addWidget(title_label)
        
        value_label = QLabel(value)
        value_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #ffffff;")
        layout.addWidget(value_label)
        
        card.value_label = value_label
        return card
    
    def create_report_metric_card(self, title, label, value, color):
        """Rapor için metrik kartı oluştur"""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background: #262636;
                border-radius: 12px;
                border-left: 5px solid {color};
                padding: 20px;
            }}
        """)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        card.setMinimumHeight(150)
        
        layout = QVBoxLayout()
        layout.setSpacing(5)
        card.setLayout(layout)
        
        title_label = QLabel(title)
        title_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {color};")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)
        
        desc_label = QLabel(label)
        desc_label.setStyleSheet("font-size: 13px; color: #6b7280; margin-top: 5px;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        value_label = QLabel(value)
        value_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #ffffff; margin-top: 10px;")
        value_label.setWordWrap(True)
        layout.addWidget(value_label)
        
        # Value label'ı daha sonra güncellemek için sakla
        card.value_label = value_label
        
        return card
    
    def set_weight_preset(self, delay, rel, res):
        """Ağırlık preset'lerini ayarla"""
        self.comp_delay_slider.setValue(delay)
        self.comp_rel_slider.setValue(rel)
        self.comp_res_slider.setValue(res)
    
    def update_comparison_labels(self):
        """Karşılaştırma slider etiketlerini güncelle"""
        self.comp_delay_label.setText(f"En Az Gecikme (Latency): {self.comp_delay_slider.value()}%")
        self.comp_rel_label.setText(f"En Yüksek Güvenilirlik (Reliability): {self.comp_rel_slider.value()}%")
        self.comp_res_label.setText(f"En Az Kaynak Kullanımı (Cost): {self.comp_res_slider.value()}%")
    
    def run_comparison(self):
        """Karşılaştırmayı çalıştır"""
        # Seçili algoritmaları al
        selected_algos = []
        algo_map = {
            'GA': 'Genetik Algoritma',
            'QL': 'Q-Learning (AI)',
            'PSO': 'PSO (Meta-Heuristic)',
            'SA': 'Benzetimli Tavlama (SA)'
        }
        
        for code, checkbox in self.algo_checkboxes.items():
            if checkbox.isChecked():
                selected_algos.append((code, algo_map[code]))
        
        if len(selected_algos) < 2:
            self.log_console.insertPlainText("\n⚠️ En az 2 algoritma seçmelisiniz!\n")
            self.log_console.moveCursor(QTextCursor.MoveOperation.End)
            return
        
        # Parametreleri al
        try:
            source = int(self.comp_source.currentText())
            target = int(self.comp_target.currentText())
            demand = int(self.comp_demand.text())
            w_delay = self.comp_delay_slider.value() / 100.0
            w_rel = self.comp_rel_slider.value() / 100.0
            w_res = self.comp_res_slider.value() / 100.0
        except ValueError:
            self.log_console.insertPlainText("\n⚠️ Geçersiz parametre değerleri!\n")
            self.log_console.moveCursor(QTextCursor.MoveOperation.End)
            return
        
        self.log_console.append(f"\n{'='*70}\n")
        self.log_console.append(f"🔄 KARŞILAŞTIRMA BAŞLATILIYOR\n")
        self.log_console.append(f"{'='*70}\n")
        self.log_console.append(f"Kaynak: {source} → Hedef: {target}\n")
        self.log_console.append(f"Talep: {demand} Mbps\n")
        self.log_console.append(f"Ağırlıklar: Gecikme={w_delay:.2f}, Güvenilirlik={w_rel:.2f}, Maliyet={w_res:.2f}\n")
        self.log_console.append(f"Seçili Algoritmalar: {', '.join([name for _, name in selected_algos])}\n")
        self.log_console.append(f"{'='*70}\n\n")
        
        self.comparison_results = []
        
        # Butonudevre dışı bırak
        self.btn_run_comparison.setEnabled(False)
        self.btn_run_comparison.setText("⏳ Analiz Ediliyor...")
        
        # Her algoritma için hesaplama yap
        for code, name in selected_algos:
            self.log_console.append(f"\n📊 {name} çalıştırılıyor...\n")
            
            # Algoritma parametrelerini al
            algo_map = {'Genetik Algoritma': 'genetic', 'Q-Learning (AI)': 'qlearning',
                       'PSO (Meta-Heuristic)': 'pso', 'Benzetimli Tavlama (SA)': 'sa'}
            algo_key = algo_map.get(code, 'pso')
            algo_params = self.algo_params[algo_key]
            
            start_time = time.time()
            try:
                result = self.manager.calculate_path(
                    source, target, w_delay, w_rel, w_res,
                    algorithm=code, demand_value=demand, algo_params=algo_params
                )
            except Exception as e:
                self.log_console.append(f"❌ {name} - Hata: {str(e)}\n")
                result = None
            elapsed = time.time() - start_time
            
            if result:
                result['algorithm'] = name
                result['algorithm_code'] = code
                result['computation_time'] = elapsed
                result['timestamp'] = datetime.now().strftime("%H:%M:%S")
                self.comparison_results.append(result)
                
                self.log_console.append(f"✅ {name} tamamlandı ({elapsed:.2f}s)\n")
                self.log_console.append(f"   Yol: {len(result['path'])} düğüm\n")
                self.log_console.append(f"   Gecikme: {result['total_delay']:.2f} ms\n")
                self.log_console.append(f"   Güvenilirlik: %{result['final_reliability']:.4f}\n")
                self.log_console.append(f"   Maliyet: {result['resource_cost']:.2f}\n")
            else:
                self.log_console.append(f"❌ {name} - Uygun yol bulunamadı\n")
        
        # Sonuçları göster
        self.display_comparison_results()
        
        # Butonu tekrar aktif et
        self.btn_run_comparison.setEnabled(True)
        self.btn_run_comparison.setText("▶ Analizi Başlat")
        
        self.log_console.append(f"\n{'='*70}\n")
        self.log_console.append(f"✅ KARŞILAŞTIRMA TAMAMLANDI\n")
        self.log_console.append(f"{'='*70}\n\n")
    
    def run_batch_test(self):
        """DemandData.csv'den toplu test çalıştır"""
        import csv
        import os
        from PyQt5.QtWidgets import QProgressDialog
        from PyQt5.QtCore import Qt
        
        # Seçili algoritmaları al
        selected_algos = []
        algo_map = {
            'GA': 'Genetik Algoritma',
            'QL': 'Q-Learning (AI)',
            'PSO': 'PSO (Meta-Heuristic)',
            'SA': 'Benzetimli Tavlama (SA)'
        }
        
        for code, checkbox in self.algo_checkboxes.items():
            if checkbox.isChecked():
                selected_algos.append((code, algo_map[code]))
        
        if len(selected_algos) < 2:
            self.log_console.append("\n⚠️ Toplu test için en az 2 algoritma seçmelisiniz!\n")
            return
        
        # DemandData.csv dosyasını bul
        demand_file = os.path.join('Algorithms', 'DemandData.csv')
        if not os.path.exists(demand_file):
            self.log_console.append(f"\n❌ HATA: {demand_file} dosyası bulunamadı!\n")
            return
        
        # CSV'den talepleri oku
        demands = []
        try:
            # Try UTF-8 with BOM first (utf-8-sig removes BOM automatically)
            encodings = ['utf-8-sig', 'utf-8', 'latin1', 'cp1252']
            file_content = None
            
            for enc in encodings:
                try:
                    with open(demand_file, 'r', encoding=enc) as f:
                        file_content = f.read()
                    self.log_console.append(f"\n✅ CSV dosyası {enc} encoding ile okundu\n")
                    break
                except:
                    continue
            
            if not file_content:
                self.log_console.append(f"\n❌ CSV dosyası okunamadı!\n")
                return
            
            # Parse CSV
            import io
            reader = csv.DictReader(io.StringIO(file_content), delimiter=';')
            for row in reader:
                # Strip whitespace and BOM from keys and values
                row = {k.strip().lstrip('\ufeff'): v.strip() for k, v in row.items()}
                
                # Debug: İlk satırı logla
                if len(demands) == 0:
                    self.log_console.append(f"İlk satır sütunları: {list(row.keys())}\n")
                    self.log_console.append(f"İlk satır değerleri: {list(row.values())}\n")
                
                if 'src' in row and 'dst' in row and 'demand_mbps' in row:
                    demands.append({
                        'src': int(row['src']),
                        'dst': int(row['dst']),
                        'demand': int(row['demand_mbps'])
                    })
            
            if not demands:
                self.log_console.append(f"\n⚠️ CSV'de hiç geçerli veri bulunamadı!\n")
                return
            
            self.log_console.append(f"✅ {len(demands)} senaryo yüklendi\n")
                
        except Exception as e:
            self.log_console.append(f"\n❌ CSV okuma hatası: {str(e)}\n")
            self.log_console.append(f"Dosya yolu: {demand_file}\n")
            import traceback
            self.log_console.append(f"Detay: {traceback.format_exc()}\n")
            return
        
        self.log_console.append(f"\n{'='*70}\n")
        self.log_console.append(f"📊 TOPLU TEST BAŞLATILIYOR\n")
        self.log_console.append(f"{'='*70}\n")
        self.log_console.append(f"Toplam Senaryo: {len(demands)}\n")
        self.log_console.append(f"Seçili Algoritmalar: {', '.join([name for _, name in selected_algos])}\n")
        self.log_console.append(f"{'='*70}\n\n")
        
        # Progress dialog - Non-modal (engelleyici değil)
        progress = QProgressDialog("Toplu test çalıştırılıyor...", "İptal", 0, len(demands), self)
        progress.setWindowTitle("Toplu Test İlerlemesi")
        progress.setWindowModality(Qt.WindowModality.NonModal)  # Diğer pencereleri bloklamaz
        progress.setMinimumDuration(0)
        progress.setMinimumWidth(500)
        progress.setValue(0)
        progress.show()
        
        # Ağırlıkları al
        w_delay = self.comp_delay_slider.value() / 100.0
        w_rel = self.comp_rel_slider.value() / 100.0
        w_res = self.comp_res_slider.value() / 100.0
        
        # Butonları devre dışı bırak
        self.btn_run_comparison.setEnabled(False)
        self.btn_batch_test.setEnabled(False)
        self.btn_batch_test.setText("⏳ Toplu Test Çalışıyor...")
        
        # Batch sonuçlarını sıfırla
        self.batch_test_results = []
        
        # Her senaryo için test
        for idx, demand_row in enumerate(demands):
            if progress.wasCanceled():
                self.log_console.append("\n⚠️ Toplu test kullanıcı tarafından iptal edildi.\n")
                break
            
            progress.setValue(idx)
            progress.setLabelText(f"Senaryo {idx+1}/{len(demands)}: {demand_row['src']}→{demand_row['dst']} ({demand_row['demand']} Mbps)")
            QApplication.processEvents()
            
            source = demand_row['src']
            target = demand_row['dst']
            demand = demand_row['demand']
            
            # Log'a da yazdır
            if (idx + 1) % 5 == 0 or idx == 0:  # Her 5 senaryoda bir
                self.log_console.append(f"⏳ İlerleme: {idx+1}/{len(demands)} senaryo tamamlandı...\n")
            
            scenario_results = []
            
            for code, name in selected_algos:
                # Algoritma parametrelerini al
                algo_map = {'Genetik Algoritma': 'genetic', 'Q-Learning (AI)': 'qlearning',
                           'PSO (Meta-Heuristic)': 'pso', 'Benzetimli Tavlama (SA)': 'sa'}
                algo_key = algo_map.get(code, 'pso')
                algo_params = self.algo_params[algo_key]
                
                start_time = time.time()
                try:
                    result = self.manager.calculate_path(
                        source, target, w_delay, w_rel, w_res,
                        algorithm=code, demand_value=demand, algo_params=algo_params
                    )
                except Exception as e:
                    result = None
                
                elapsed = time.time() - start_time
                
                if result:
                    result['algorithm'] = name
                    result['algorithm_code'] = code
                    result['computation_time'] = elapsed
                    result['scenario'] = f"{source}→{target}"
                    result['demand'] = demand
                    scenario_results.append(result)
            
            if scenario_results:
                self.batch_test_results.append({
                    'scenario': f"{source}→{target}",
                    'source': source,
                    'target': target,
                    'demand': demand,
                    'results': scenario_results
                })
        
        progress.setValue(len(demands))
        
        # Butonları tekrar aktif et
        self.btn_run_comparison.setEnabled(True)
        self.btn_batch_test.setEnabled(True)
        self.btn_batch_test.setText("📊 Toplu Test Başlat (DemandData)")
        
        self.log_console.append(f"\n{'='*70}\n")
        self.log_console.append(f"✅ TOPLU TEST TAMAMLANDI\n")
        self.log_console.append(f"Toplam Senaryo: {len(self.batch_test_results)}\n")
        self.log_console.append(f"Toplam Hesaplama: {len(self.batch_test_results) * len(selected_algos)}\n")
        self.log_console.append(f"{'='*70}\n\n")
        
        # Raporları güncelle
        self.update_batch_reports()
        
        # Raporlar ekranına ve Toplu Test tab'ına geç
        self.stacked_widget.setCurrentIndex(2)  # Raporlar ekranı
        self.report_tabs.setCurrentIndex(3)  # Toplu Test tab'ı
    
    def display_comparison_results(self):
        """Karşılaştırma sonuçlarını göster"""
        # Önceki sonuçları temizle
        for i in reversed(range(self.comparison_results_layout.count())): 
            item = self.comparison_results_layout.itemAt(i)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
        if not self.comparison_results:
            no_result = QLabel("Henüz sonuç yok.")
            no_result.setStyleSheet("color: #6b7280; font-size: 12px; padding: 20px;")
            self.comparison_results_layout.addWidget(no_result)
            return
        
        # Her algoritma için sonuç kartı
        for result in self.comparison_results:
            card = self.create_result_card(result)
            self.comparison_results_layout.addWidget(card)
        
        self.comparison_results_layout.addStretch()
    
    def create_result_card(self, result):
        """Tek bir algoritma sonucu için kart oluştur"""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background: #262636;
                border-radius: 12px;
                padding: 20px;
                border: 1px solid #313244;
                margin: 8px;
            }
        """)
        
        layout = QVBoxLayout()
        card.setLayout(layout)
        
        # Başlık
        header_layout = QHBoxLayout()
        
        algo_icon = {"GA": "🧬", "QL": "🤖", "PSO": "🐝", "SA": "🔥"}
        icon = algo_icon.get(result['algorithm_code'], "📊")
        
        title = QLabel(f"{icon} {result['algorithm']}")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #3b82f6;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        time_label = QLabel(f"⏱️ {result['computation_time']:.2f}s")
        time_label.setStyleSheet("font-size: 14px; color: #10b981; font-weight: bold;")
        header_layout.addWidget(time_label)
        
        layout.addLayout(header_layout)
        
        # Metrikler
        metrics_layout = QHBoxLayout()
        
        delay_widget = self.create_mini_metric("Gecikme", f"{result['total_delay']:.2f} ms", "#ff6b6b")
        reliability_widget = self.create_mini_metric("Güvenilirlik", f"%{result['final_reliability']:.4f}", "#1dd1a1")
        cost_widget = self.create_mini_metric("Maliyet", f"{result['resource_cost']:.2f}", "#feca57")
        
        metrics_layout.addWidget(delay_widget)
        metrics_layout.addWidget(reliability_widget)
        metrics_layout.addWidget(cost_widget)
        
        layout.addLayout(metrics_layout)
        
        # Yol Bilgisi
        path_info = QLabel(f"🛤️ Yol Uzunluğu: {len(result['path'])} düğüm | {result['timestamp']}")
        path_info.setStyleSheet("font-size: 13px; color: #6b7280; margin-top: 8px;")
        layout.addWidget(path_info)
        
        return card
    
    def create_mini_metric(self, title, value, color):
        """Küçük metrik widget'ı"""
        widget = QFrame()
        widget.setStyleSheet(f"background: #1e1e2e; border-radius: 8px; padding: 15px; border-left: 4px solid {color};")
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 11px; color: #6b7280; font-weight: bold;")
        layout.addWidget(title_label)
        
        value_label = QLabel(value)
        value_label.setStyleSheet("font-size: 16px; color: #ffffff; font-weight: bold; margin-top: 5px;")
        layout.addWidget(value_label)
        
        return widget
    
    def update_reports_screen(self):
        """Raporlar ekranını güncelle"""
        if not self.comparison_results:
            return
        
        # En iyi sonuçları bul
        best_delay = min(self.comparison_results, key=lambda x: x['total_delay'])
        best_reliability = max(self.comparison_results, key=lambda x: x['final_reliability'])
        best_cost = min(self.comparison_results, key=lambda x: x['resource_cost'])
        
        # Dengeli skoru hesapla (normalize edilmiş)
        for r in self.comparison_results:
            delays = [x['total_delay'] for x in self.comparison_results]
            rels = [x['final_reliability'] for x in self.comparison_results]
            costs = [x['resource_cost'] for x in self.comparison_results]
            
            norm_delay = 1 - ((r['total_delay'] - min(delays)) / (max(delays) - min(delays) + 0.001))
            norm_rel = (r['final_reliability'] - min(rels)) / (max(rels) - min(rels) + 0.001)
            norm_cost = 1 - ((r['resource_cost'] - min(costs)) / (max(costs) - min(costs) + 0.001))
            
            r['balanced_score'] = (norm_delay + norm_rel + norm_cost) / 3
        
        best_balanced = max(self.comparison_results, key=lambda x: x['balanced_score'])
        
        # Özet kartlarını güncelle
        self.report_summary_widgets['fastest'].value_label.setText(best_delay['algorithm'])
        self.report_summary_widgets['reliable'].value_label.setText(best_reliability['algorithm'])
        self.report_summary_widgets['efficient'].value_label.setText(best_cost['algorithm'])
        self.report_summary_widgets['balanced'].value_label.setText(best_balanced['algorithm'])
        
        # İstatistikler hesapla
        delays = [x['total_delay'] for x in self.comparison_results]
        rels = [x['final_reliability'] for x in self.comparison_results]
        costs = [x['resource_cost'] for x in self.comparison_results]
        times = [x['computation_time'] for x in self.comparison_results]
        
        avg_delay = sum(delays) / len(delays)
        avg_rel = sum(rels) / len(rels)
        avg_cost = sum(costs) / len(costs)
        avg_time = sum(times) / len(times)
        
        # Özet Bilgileri
        summary_html = "<div style='background: #1e1e2e; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>"
        summary_html += f"<p style='margin: 5px 0; font-size: 14px;'><b style='color: #3b82f6;'>📊 Toplam Test:</b> {len(self.comparison_results)} algoritma</p>"
        summary_html += f"<p style='margin: 5px 0; font-size: 14px;'><b style='color: #10b981;'>⚡ Ort. Gecikme:</b> {avg_delay:.2f} ms</p>"
        summary_html += f"<p style='margin: 5px 0; font-size: 14px;'><b style='color: #f59e0b;'>🛡️ Ort. Güvenilirlik:</b> %{avg_rel:.4f}</p>"
        summary_html += f"<p style='margin: 5px 0; font-size: 14px;'><b style='color: #ef4444;'>💰 Ort. Maliyet:</b> {avg_cost:.2f}</p>"
        summary_html += f"<p style='margin: 5px 0; font-size: 14px;'><b style='color: #8b5cf6;'>⏱️ Ort. Hesaplama Süresi:</b> {avg_time:.3f}s</p>"
        summary_html += "</div>"
        
        # Detaylı tablo
        table_html = "<table style='width:100%; border-collapse: collapse; font-size: 13px;'>"
        table_html += "<tr style='background: #1e1e2e; font-weight: bold;'>"
        table_html += "<td style='padding: 10px; border: 1px solid #313244;'>🏆 Sıra</td>"
        table_html += "<td style='padding: 10px; border: 1px solid #313244;'>Algoritma</td>"
        table_html += "<td style='padding: 10px; border: 1px solid #313244;'>Gecikme (ms)</td>"
        table_html += "<td style='padding: 10px; border: 1px solid #313244;'>Güvenilirlik (%)</td>"
        table_html += "<td style='padding: 10px; border: 1px solid #313244;'>Maliyet</td>"
        table_html += "<td style='padding: 10px; border: 1px solid #313244;'>Süre (s)</td>"
        table_html += "<td style='padding: 10px; border: 1px solid #313244;'>Dengeli Skor</td>"
        table_html += "<td style='padding: 10px; border: 1px solid #313244;'>Performans</td>"
        table_html += "</tr>"
        
        sorted_results = sorted(self.comparison_results, key=lambda x: x['balanced_score'], reverse=True)
        for rank, r in enumerate(sorted_results, 1):
            # Yüzdesel fark hesapla (en iyiye göre)
            delay_diff = ((r['total_delay'] - best_delay['total_delay']) / best_delay['total_delay']) * 100
            rel_diff = ((r['final_reliability'] - best_reliability['final_reliability']) / best_reliability['final_reliability']) * 100
            cost_diff = ((r['resource_cost'] - best_cost['resource_cost']) / best_cost['resource_cost']) * 100
            
            # Performans yüzdesi
            perf_percent = (r['balanced_score'] / sorted_results[0]['balanced_score']) * 100
            
            # Sıra ikonu
            rank_icon = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            
            # Performans bar
            bar_width = int(perf_percent)
            perf_bar = f"<div style='background: linear-gradient(90deg, #3b82f6 {bar_width}%, #313244 {bar_width}%); width: 100px; height: 20px; border-radius: 4px; display: inline-block;'></div> {perf_percent:.1f}%"
            
            table_html += f"<tr style='background: {'#262636' if rank == 1 else '#232330'};'>"
            table_html += f"<td style='padding: 8px; border: 1px solid #313244; text-align: center;'>{rank_icon}</td>"
            table_html += f"<td style='padding: 8px; border: 1px solid #313244; font-weight: bold;'>{r['algorithm']}</td>"
            table_html += f"<td style='padding: 8px; border: 1px solid #313244;'>{r['total_delay']:.2f} <span style='color: {'#10b981' if delay_diff <= 0 else '#ef4444'}; font-size: 11px;'>({delay_diff:+.1f}%)</span></td>"
            table_html += f"<td style='padding: 8px; border: 1px solid #313244;'>{r['final_reliability']:.4f} <span style='color: {'#10b981' if rel_diff >= 0 else '#ef4444'}; font-size: 11px;'>({rel_diff:+.1f}%)</span></td>"
            table_html += f"<td style='padding: 8px; border: 1px solid #313244;'>{r['resource_cost']:.2f} <span style='color: {'#10b981' if cost_diff <= 0 else '#ef4444'}; font-size: 11px;'>({cost_diff:+.1f}%)</span></td>"
            table_html += f"<td style='padding: 8px; border: 1px solid #313244;'>{r['computation_time']:.3f}</td>"
            table_html += f"<td style='padding: 8px; border: 1px solid #313244; text-align: center; font-weight: bold;'>{r['balanced_score']:.3f}</td>"
            table_html += f"<td style='padding: 8px; border: 1px solid #313244;'>{perf_bar}</td>"
            table_html += f"</tr>"
        
        table_html += "</table>"
        
        final_html = summary_html + table_html
        self.report_detail_table.setText(final_html)
        self.report_detail_table.setTextFormat(Qt.TextFormat.RichText)
        
        # Grafikleri güncelle
        self.update_comparison_chart()
        self.update_performance_chart()
    
    def update_comparison_chart(self):
        """Karşılaştırma grafiğini güncelle"""
        self.comparison_figure.clear()
        
        if not self.comparison_results:
            return
        
        axes = self.comparison_figure.subplots(1, 3)
        
        algorithms = [r['algorithm'] for r in self.comparison_results]
        delays = [r['total_delay'] for r in self.comparison_results]
        reliabilities = [r['final_reliability'] for r in self.comparison_results]
        costs = [r['resource_cost'] for r in self.comparison_results]
        
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']
        
        # Gecikme
        axes[0].bar(algorithms, delays, color=colors[:len(algorithms)])
        axes[0].set_title('Gecikme (ms)', color='white', fontsize=12, fontweight='bold')
        axes[0].set_facecolor('#232330')
        axes[0].tick_params(colors='white', labelsize=9, rotation=15)
        axes[0].spines['bottom'].set_color('white')
        axes[0].spines['left'].set_color('white')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        
        # Güvenilirlik
        axes[1].bar(algorithms, reliabilities, color=colors[:len(algorithms)])
        axes[1].set_title('Güvenilirlik (%)', color='white', fontsize=12, fontweight='bold')
        axes[1].set_facecolor('#232330')
        axes[1].tick_params(colors='white', labelsize=9, rotation=15)
        axes[1].spines['bottom'].set_color('white')
        axes[1].spines['left'].set_color('white')
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        
        # Maliyet
        axes[2].bar(algorithms, costs, color=colors[:len(algorithms)])
        axes[2].set_title('Kaynak Maliyeti', color='white', fontsize=12, fontweight='bold')
        axes[2].set_facecolor('#232330')
        axes[2].tick_params(colors='white', labelsize=9, rotation=15)
        axes[2].spines['bottom'].set_color('white')
        axes[2].spines['left'].set_color('white')
        axes[2].spines['top'].set_visible(False)
        axes[2].spines['right'].set_visible(False)
        
        self.comparison_figure.tight_layout()
        self.comparison_canvas.draw_idle()  # Thread-safe çizim
    
    def update_performance_chart(self):
        """Performans radar grafiğini güncelle"""
        self.performance_figure.clear()
        
        if not self.comparison_results:
            return
        
        import numpy as np
        
        ax = self.performance_figure.add_subplot(111, projection='polar')
        
        categories = ['Gecikme', 'Güvenilirlik', 'Maliyet']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']
        
        for idx, result in enumerate(self.comparison_results):
            # Normalize et (0-1 arası)
            delays = [x['total_delay'] for x in self.comparison_results]
            rels = [x['final_reliability'] for x in self.comparison_results]
            costs = [x['resource_cost'] for x in self.comparison_results]
            
            norm_delay = 1 - ((result['total_delay'] - min(delays)) / (max(delays) - min(delays) + 0.001))
            norm_rel = (result['final_reliability'] - min(rels)) / (max(rels) - min(rels) + 0.001)
            norm_cost = 1 - ((result['resource_cost'] - min(costs)) / (max(costs) - min(costs) + 0.001))
            
            values = [norm_delay, norm_rel, norm_cost]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=result['algorithm'], color=colors[idx % len(colors)])
            ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
        
        ax.set_theta_offset(np.pi / 2)  # type: ignore
        ax.set_theta_direction(-1)  # type: ignore
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, color='white', size=10)
        ax.set_ylim(0, 1)
        ax.set_facecolor('#232330')
        ax.spines['polar'].set_color('white')
        ax.tick_params(colors='white')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor='#262636', edgecolor='white', labelcolor='white')
        ax.grid(color='#45475a', linestyle='--', linewidth=0.5)
        
        self.performance_figure.tight_layout()
        self.performance_canvas.draw_idle()  # Thread-safe çizim
    
    def update_batch_reports(self):
        """Toplu test raporlarını güncelle"""
        if not self.batch_test_results:
            return
        
        # İstatistikleri hesapla
        total_scenarios = len(self.batch_test_results)
        algo_wins = {}  # Her algoritmanın kaç kez kazandığı
        algo_times = {}  # Her algoritmanın toplam süresi
        
        for scenario in self.batch_test_results:
            results = scenario['results']
            if not results:
                continue
            
            # En iyi sonucu bul (balanced score'a göre)
            for r in results:
                delays = [x['total_delay'] for x in results]
                rels = [x['final_reliability'] for x in results]
                costs = [x['resource_cost'] for x in results]
                
                norm_delay = 1 - ((r['total_delay'] - min(delays)) / (max(delays) - min(delays) + 0.001))
                norm_rel = (r['final_reliability'] - min(rels)) / (max(rels) - min(rels) + 0.001)
                norm_cost = 1 - ((r['resource_cost'] - min(costs)) / (max(costs) - min(costs) + 0.001))
                
                r['balanced_score'] = (norm_delay + norm_rel + norm_cost) / 3
            
            best = max(results, key=lambda x: x['balanced_score'])
            
            # Kazananı kaydet
            algo_name = best['algorithm']
            algo_wins[algo_name] = algo_wins.get(algo_name, 0) + 1
            
            # Süreleri topla
            for r in results:
                algo_times[r['algorithm']] = algo_times.get(r['algorithm'], 0) + r['computation_time']
        
        # En başarılı algoritma
        winner = max(algo_wins.items(), key=lambda x: x[1]) if algo_wins else ("-", 0)
        
        # Ortalama süre
        total_time = sum(algo_times.values())
        avg_time = total_time / (len(algo_times) * total_scenarios) if algo_times else 0
        
        # Başarı oranı (en az bir algoritmanın başarılı olduğu senaryolar)
        success_rate = (len([s for s in self.batch_test_results if s['results']]) / total_scenarios) * 100
        
        # Kartları güncelle
        self.batch_total_card.value_label.setText(f"{total_scenarios}")
        self.batch_winner_card.value_label.setText(f"{winner[0]}\n({winner[1]} / {total_scenarios})")
        self.batch_avg_time_card.value_label.setText(f"{avg_time:.3f}s")
        self.batch_success_card.value_label.setText(f"{success_rate:.1f}%")
        
        # Detaylı tablo
        table_html = "<div style='background: #1e1e2e; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>"
        table_html += f"<p style='margin: 5px 0; font-size: 14px;'><b style='color: #3b82f6;'>📊 Algoritma Başarı Tablosu:</b></p>"
        for algo, wins in sorted(algo_wins.items(), key=lambda x: x[1], reverse=True):
            win_rate = (wins / total_scenarios) * 100
            bar_width = int(win_rate)
            bar = f"<div style='background: linear-gradient(90deg, #10b981 {bar_width}%, #313244 {bar_width}%); width: 200px; height: 20px; border-radius: 4px; display: inline-block;'></div>"
            table_html += f"<p style='margin: 8px 0; font-size: 13px;'><b>{algo}:</b> {wins} kazanım ({win_rate:.1f}%) {bar}</p>"
        table_html += "</div>"
        
        # Senaryo detayları
        table_html += "<table style='width:100%; border-collapse: collapse; font-size: 12px;'>"
        table_html += "<tr style='background: #1e1e2e; font-weight: bold;'>"
        table_html += "<td style='padding: 8px; border: 1px solid #313244;'>Senaryo</td>"
        table_html += "<td style='padding: 8px; border: 1px solid #313244;'>Talep</td>"
        table_html += "<td style='padding: 8px; border: 1px solid #313244;'>Kazanan</td>"
        table_html += "<td style='padding: 8px; border: 1px solid #313244;'>Gecikme</td>"
        table_html += "<td style='padding: 8px; border: 1px solid #313244;'>Güvenilirlik</td>"
        table_html += "<td style='padding: 8px; border: 1px solid #313244;'>Skor</td>"
        table_html += "</tr>"
        
        for idx, scenario in enumerate(self.batch_test_results, 1):
            if not scenario['results']:
                continue
            
            best = max(scenario['results'], key=lambda x: x['balanced_score'])
            
            table_html += f"<tr style='background: {'#262636' if idx % 2 == 0 else '#232330'};'>"
            table_html += f"<td style='padding: 6px; border: 1px solid #313244;'>{scenario['scenario']}</td>"
            table_html += f"<td style='padding: 6px; border: 1px solid #313244;'>{scenario['demand']} Mbps</td>"
            table_html += f"<td style='padding: 6px; border: 1px solid #313244; font-weight: bold; color: #10b981;'>{best['algorithm']}</td>"
            table_html += f"<td style='padding: 6px; border: 1px solid #313244;'>{best['total_delay']:.2f} ms</td>"
            table_html += f"<td style='padding: 6px; border: 1px solid #313244;'>{best['final_reliability']:.4f}%</td>"
            table_html += f"<td style='padding: 6px; border: 1px solid #313244;'>{best['balanced_score']:.3f}</td>"
            table_html += f"</tr>"
        
        table_html += "</table>"
        
        self.batch_detail_table.setText(table_html)
        self.batch_detail_table.setTextFormat(Qt.TextFormat.RichText)
    
    def export_csv_report(self):
        """CSV olarak rapor dışa aktar"""
        if not self.comparison_results:
            self.log_console.append("\n⚠️ Dışa aktarmak için önce karşılaştırma yapmalısınız!\n")
            return
        
        filename, _ = QFileDialog.getSaveFileName(self, "CSV Olarak Kaydet", "", "CSV Files (*.csv)")
        if not filename:
            return
        
        try:
            import csv
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Algoritma', 'Gecikme (ms)', 'Güvenilirlik (%)', 'Maliyet', 
                               'Hesaplama Süresi (s)', 'Yol Uzunluğu', 'Zaman Damgası'])
                
                for r in self.comparison_results:
                    writer.writerow([
                        r['algorithm'],
                        f"{r['total_delay']:.2f}",
                        f"{r['final_reliability']:.4f}",
                        f"{r['resource_cost']:.2f}",
                        f"{r['computation_time']:.2f}",
                        len(r['path']),
                        r['timestamp']
                    ])
            
            self.log_console.append(f"\n✅ Rapor başarıyla kaydedildi: {filename}\n")
        except Exception as e:
            self.log_console.append(f"\n❌ CSV kaydedilemedi: {str(e)}\n")
    
    def export_pdf_report(self):
        """PDF olarak rapor dışa aktar (placeholder)"""
        self.log_console.append("\n📄 PDF dışa aktarma özelliği yakında eklenecek!\n")
        self.log_console.append("Şu an için CSV formatını kullanabilirsiniz.\n")
    
    def show_algo_params_dialog(self):
        """Seçili algoritmanın parametrelerini düzenlemek için dialog göster"""
        algo_index = self.combo_algo.currentIndex()
        algo_map = {0: 'pso', 1: 'qlearning', 2: 'genetic', 3: 'sa'}
        algo_key = algo_map.get(algo_index, 'pso')
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"⚙️ {self.combo_algo.currentText()} - Parametreler")
        dialog.setMinimumWidth(400)
        dialog.setStyleSheet("background: #2d3748; color: white;")
        
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        form_layout = QFormLayout()
        form_layout.setSpacing(10)
        
        # Parametreleri algoritma türüne göre ekle
        param_widgets = {}
        
        if algo_key == 'qlearning':
            # Q-Learning parametreleri
            alpha_spin = QDoubleSpinBox()
            alpha_spin.setRange(0.01, 1.0)
            alpha_spin.setSingleStep(0.05)
            alpha_spin.setValue(self.algo_params['qlearning']['alpha'])
            alpha_spin.setDecimals(2)
            form_layout.addRow("Alpha (Öğrenme Oranı):", alpha_spin)
            param_widgets['alpha'] = alpha_spin
            
            gamma_spin = QDoubleSpinBox()
            gamma_spin.setRange(0.0, 1.0)
            gamma_spin.setSingleStep(0.05)
            gamma_spin.setValue(self.algo_params['qlearning']['gamma'])
            gamma_spin.setDecimals(2)
            form_layout.addRow("Gamma (İskonto Faktörü):", gamma_spin)
            param_widgets['gamma'] = gamma_spin
            
            epsilon_spin = QDoubleSpinBox()
            epsilon_spin.setRange(0.0, 1.0)
            epsilon_spin.setSingleStep(0.05)
            epsilon_spin.setValue(self.algo_params['qlearning']['epsilon'])
            epsilon_spin.setDecimals(2)
            form_layout.addRow("Epsilon (Keşif Oranı):", epsilon_spin)
            param_widgets['epsilon'] = epsilon_spin
            
            episodes_spin = QSpinBox()
            episodes_spin.setRange(100, 5000)
            episodes_spin.setSingleStep(100)
            episodes_spin.setValue(self.algo_params['qlearning']['episodes'])
            form_layout.addRow("Episode Sayısı:", episodes_spin)
            param_widgets['episodes'] = episodes_spin
            
        elif algo_key == 'pso':
            # PSO parametreleri
            swarm_spin = QSpinBox()
            swarm_spin.setRange(10, 100)
            swarm_spin.setSingleStep(5)
            swarm_spin.setValue(self.algo_params['pso']['swarm_size'])
            form_layout.addRow("Parçacık Sayısı:", swarm_spin)
            param_widgets['swarm_size'] = swarm_spin
            
            iter_spin = QSpinBox()
            iter_spin.setRange(10, 200)
            iter_spin.setSingleStep(5)
            iter_spin.setValue(self.algo_params['pso']['iterations'])
            form_layout.addRow("İterasyon Sayısı:", iter_spin)
            param_widgets['iterations'] = iter_spin
            
            w_spin = QDoubleSpinBox()
            w_spin.setRange(0.1, 1.0)
            w_spin.setSingleStep(0.05)
            w_spin.setValue(self.algo_params['pso']['w'])
            w_spin.setDecimals(2)
            form_layout.addRow("Atalet Ağırlığı (w):", w_spin)
            param_widgets['w'] = w_spin
            
            c1_spin = QDoubleSpinBox()
            c1_spin.setRange(0.5, 3.0)
            c1_spin.setSingleStep(0.1)
            c1_spin.setValue(self.algo_params['pso']['c1'])
            c1_spin.setDecimals(1)
            form_layout.addRow("Bilişsel Katsayı (c1):", c1_spin)
            param_widgets['c1'] = c1_spin
            
            c2_spin = QDoubleSpinBox()
            c2_spin.setRange(0.5, 3.0)
            c2_spin.setSingleStep(0.1)
            c2_spin.setValue(self.algo_params['pso']['c2'])
            c2_spin.setDecimals(1)
            form_layout.addRow("Sosyal Katsayı (c2):", c2_spin)
            param_widgets['c2'] = c2_spin
            
        elif algo_key == 'genetic':
            # Genetik Algoritma parametreleri
            pop_spin = QSpinBox()
            pop_spin.setRange(20, 200)
            pop_spin.setSingleStep(10)
            pop_spin.setValue(self.algo_params['genetic']['population'])
            form_layout.addRow("Popülasyon Boyutu:", pop_spin)
            param_widgets['population'] = pop_spin
            
            gen_spin = QSpinBox()
            gen_spin.setRange(50, 1000)
            gen_spin.setSingleStep(50)
            gen_spin.setValue(self.algo_params['genetic']['generations'])
            form_layout.addRow("Nesil Sayısı:", gen_spin)
            param_widgets['generations'] = gen_spin
            
            cross_spin = QDoubleSpinBox()
            cross_spin.setRange(0.5, 1.0)
            cross_spin.setSingleStep(0.05)
            cross_spin.setValue(self.algo_params['genetic']['crossover'])
            cross_spin.setDecimals(2)
            form_layout.addRow("Crossover Oranı:", cross_spin)
            param_widgets['crossover'] = cross_spin
            
            mut_spin = QDoubleSpinBox()
            mut_spin.setRange(0.01, 0.5)
            mut_spin.setSingleStep(0.01)
            mut_spin.setValue(self.algo_params['genetic']['mutation'])
            mut_spin.setDecimals(2)
            form_layout.addRow("Mutasyon Oranı:", mut_spin)
            param_widgets['mutation'] = mut_spin
            
        elif algo_key == 'sa':
            # Simulated Annealing parametreleri
            temp_spin = QSpinBox()
            temp_spin.setRange(100, 5000)
            temp_spin.setSingleStep(100)
            temp_spin.setValue(self.algo_params['sa']['initial_temp'])
            form_layout.addRow("Başlangıç Sıcaklığı:", temp_spin)
            param_widgets['initial_temp'] = temp_spin
            
            cool_spin = QDoubleSpinBox()
            cool_spin.setRange(0.80, 0.99)
            cool_spin.setSingleStep(0.01)
            cool_spin.setValue(self.algo_params['sa']['cooling_rate'])
            cool_spin.setDecimals(2)
            form_layout.addRow("Soğuma Oranı:", cool_spin)
            param_widgets['cooling_rate'] = cool_spin
            
            iter_spin = QSpinBox()
            iter_spin.setRange(100, 2000)
            iter_spin.setSingleStep(100)
            iter_spin.setValue(self.algo_params['sa']['iterations'])
            form_layout.addRow("İterasyon Sayısı:", iter_spin)
            param_widgets['iterations'] = iter_spin
        
        layout.addLayout(form_layout)
        
        # Butonlar
        from PyQt5.QtWidgets import QDialogButtonBox as QDB
        btn_box = QDB(QDB.StandardButton(QDB.StandardButton.Ok | QDB.StandardButton.Cancel))
        btn_box.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover { background-color: #2563eb; }
        """)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.DialogCode.Accepted:
            # Parametreleri kaydet
            for key, widget in param_widgets.items():
                self.algo_params[algo_key][key] = widget.value()
            self.log_console.insertPlainText(f"\n✅ {self.combo_algo.currentText()} parametreleri güncellendi\n")
            self.log_console.moveCursor(QTextCursor.MoveOperation.End)
    
    def show_comparison_algo_params(self, algo_code):
        """Karşılaştırma ekranında seçili algoritmanın parametrelerini düzenle"""
        from PyQt5.QtWidgets import QDialogButtonBox as QDB
        
        algo_map = {'GA': ('genetic', 'Genetik Algoritma'),
                    'QL': ('qlearning', 'Q-Learning'),
                    'PSO': ('pso', 'PSO'),
                    'SA': ('sa', 'Benzetimli Tavlama')}
        
        algo_key, algo_name = algo_map.get(algo_code, ('pso', 'PSO'))
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"⚙️ {algo_name} - Parametreler")
        dialog.setMinimumWidth(400)
        dialog.setStyleSheet("background: #2d3748; color: white;")
        
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        form_layout = QFormLayout()
        form_layout.setSpacing(10)
        
        # Parametreleri algoritma türüne göre ekle (ana ekranla aynı kod)
        param_widgets = {}
        
        if algo_key == 'qlearning':
            alpha_spin = QDoubleSpinBox()
            alpha_spin.setRange(0.01, 1.0)
            alpha_spin.setSingleStep(0.05)
            alpha_spin.setValue(self.algo_params['qlearning']['alpha'])
            alpha_spin.setDecimals(2)
            form_layout.addRow("Alpha (Öğrenme Oranı):", alpha_spin)
            param_widgets['alpha'] = alpha_spin
            
            gamma_spin = QDoubleSpinBox()
            gamma_spin.setRange(0.0, 1.0)
            gamma_spin.setSingleStep(0.05)
            gamma_spin.setValue(self.algo_params['qlearning']['gamma'])
            gamma_spin.setDecimals(2)
            form_layout.addRow("Gamma (İskonto Faktörü):", gamma_spin)
            param_widgets['gamma'] = gamma_spin
            
            epsilon_spin = QDoubleSpinBox()
            epsilon_spin.setRange(0.0, 1.0)
            epsilon_spin.setSingleStep(0.05)
            epsilon_spin.setValue(self.algo_params['qlearning']['epsilon'])
            epsilon_spin.setDecimals(2)
            form_layout.addRow("Epsilon (Keşif Oranı):", epsilon_spin)
            param_widgets['epsilon'] = epsilon_spin
            
            episodes_spin = QSpinBox()
            episodes_spin.setRange(100, 5000)
            episodes_spin.setSingleStep(100)
            episodes_spin.setValue(self.algo_params['qlearning']['episodes'])
            form_layout.addRow("Episode Sayısı:", episodes_spin)
            param_widgets['episodes'] = episodes_spin
            
        elif algo_key == 'pso':
            swarm_spin = QSpinBox()
            swarm_spin.setRange(10, 100)
            swarm_spin.setSingleStep(5)
            swarm_spin.setValue(self.algo_params['pso']['swarm_size'])
            form_layout.addRow("Parçacık Sayısı:", swarm_spin)
            param_widgets['swarm_size'] = swarm_spin
            
            iter_spin = QSpinBox()
            iter_spin.setRange(10, 200)
            iter_spin.setSingleStep(5)
            iter_spin.setValue(self.algo_params['pso']['iterations'])
            form_layout.addRow("İterasyon Sayısı:", iter_spin)
            param_widgets['iterations'] = iter_spin
            
            w_spin = QDoubleSpinBox()
            w_spin.setRange(0.1, 1.0)
            w_spin.setSingleStep(0.05)
            w_spin.setValue(self.algo_params['pso']['w'])
            w_spin.setDecimals(2)
            form_layout.addRow("Atalet Ağırlığı (w):", w_spin)
            param_widgets['w'] = w_spin
            
            c1_spin = QDoubleSpinBox()
            c1_spin.setRange(0.5, 3.0)
            c1_spin.setSingleStep(0.1)
            c1_spin.setValue(self.algo_params['pso']['c1'])
            c1_spin.setDecimals(1)
            form_layout.addRow("Bilişsel Katsayı (c1):", c1_spin)
            param_widgets['c1'] = c1_spin
            
            c2_spin = QDoubleSpinBox()
            c2_spin.setRange(0.5, 3.0)
            c2_spin.setSingleStep(0.1)
            c2_spin.setValue(self.algo_params['pso']['c2'])
            c2_spin.setDecimals(1)
            form_layout.addRow("Sosyal Katsayı (c2):", c2_spin)
            param_widgets['c2'] = c2_spin
            
        elif algo_key == 'genetic':
            pop_spin = QSpinBox()
            pop_spin.setRange(20, 200)
            pop_spin.setSingleStep(10)
            pop_spin.setValue(self.algo_params['genetic']['population'])
            form_layout.addRow("Popülasyon Boyutu:", pop_spin)
            param_widgets['population'] = pop_spin
            
            gen_spin = QSpinBox()
            gen_spin.setRange(50, 1000)
            gen_spin.setSingleStep(50)
            gen_spin.setValue(self.algo_params['genetic']['generations'])
            form_layout.addRow("Nesil Sayısı:", gen_spin)
            param_widgets['generations'] = gen_spin
            
            cross_spin = QDoubleSpinBox()
            cross_spin.setRange(0.5, 1.0)
            cross_spin.setSingleStep(0.05)
            cross_spin.setValue(self.algo_params['genetic']['crossover'])
            cross_spin.setDecimals(2)
            form_layout.addRow("Crossover Oranı:", cross_spin)
            param_widgets['crossover'] = cross_spin
            
            mut_spin = QDoubleSpinBox()
            mut_spin.setRange(0.01, 0.5)
            mut_spin.setSingleStep(0.01)
            mut_spin.setValue(self.algo_params['genetic']['mutation'])
            mut_spin.setDecimals(2)
            form_layout.addRow("Mutasyon Oranı:", mut_spin)
            param_widgets['mutation'] = mut_spin
            
        elif algo_key == 'sa':
            temp_spin = QSpinBox()
            temp_spin.setRange(100, 5000)
            temp_spin.setSingleStep(100)
            temp_spin.setValue(self.algo_params['sa']['initial_temp'])
            form_layout.addRow("Başlangıç Sıcaklığı:", temp_spin)
            param_widgets['initial_temp'] = temp_spin
            
            cool_spin = QDoubleSpinBox()
            cool_spin.setRange(0.80, 0.99)
            cool_spin.setSingleStep(0.01)
            cool_spin.setValue(self.algo_params['sa']['cooling_rate'])
            cool_spin.setDecimals(2)
            form_layout.addRow("Soğuma Oranı:", cool_spin)
            param_widgets['cooling_rate'] = cool_spin
            
            iter_spin = QSpinBox()
            iter_spin.setRange(100, 2000)
            iter_spin.setSingleStep(100)
            iter_spin.setValue(self.algo_params['sa']['iterations'])
            form_layout.addRow("İterasyon Sayısı:", iter_spin)
            param_widgets['iterations'] = iter_spin
        
        layout.addLayout(form_layout)
        
        # Butonlar
        btn_box = QDB(QDB.StandardButton(QDB.StandardButton.Ok | QDB.StandardButton.Cancel))
        btn_box.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover { background-color: #2563eb; }
        """)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.DialogCode.Accepted:
            # Parametreleri kaydet
            for key, widget in param_widgets.items():
                self.algo_params[algo_key][key] = widget.value()
            self.log_console.insertPlainText(f"\n✅ {algo_name} parametreleri güncellendi\n")
            self.log_console.moveCursor(QTextCursor.MoveOperation.End)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())