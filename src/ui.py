import sys
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QFrame, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QLabel, QPushButton, QSlider, QGroupBox, QTextEdit, QDialog, QMenu
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QAction
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class NetworkPainterWidget(QWidget):
    def __init__(self, agent, parent=None):
        super().__init__(parent)
        self.agent = agent
        self.setMinimumSize(600, 450)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        try:
            model = self.agent.model
            w1 = model.network[0].weight.data.cpu().numpy()  # (16, 3)
            w2 = model.network[2].weight.data.cpu().numpy()  # (12, 16)
            w3 = model.network[4].weight.data.cpu().numpy()  # (2, 12)
        except Exception:
            return

        layer_x = [60, 220, 380, 540]
        layers_nodes = [3, 16, 12, 2]
        node_positions = []
        
        for layer_idx, nodes_count in enumerate(layers_nodes):
            x = layer_x[layer_idx]
            y_positions = []
            spacing = (self.height() - 40) / (nodes_count + 1)
            for i in range(nodes_count):
                y = 20 + (i + 1) * spacing
                y_positions.append((x, y))
            node_positions.append(y_positions)

        for i in range(3):
            for j in range(16):
                weight = w1[j, i]
                self.draw_connection(painter, node_positions[0][i], node_positions[1][j], weight)

        for i in range(16):
            for j in range(12):
                weight = w2[j, i]
                self.draw_connection(painter, node_positions[1][i], node_positions[2][j], weight)

        for i in range(12):
            for j in range(2):
                weight = w3[j, i]
                self.draw_connection(painter, node_positions[2][i], node_positions[3][j], weight)

        neuron_radius = 8
        input_names = ["Gün", "Min\nRisk", "Max\nRisk"]
        output_names = ["Ekmek", "Peynir"]

        for layer_idx, nodes in enumerate(node_positions):
            for node_idx, (x, y) in enumerate(nodes):
                if layer_idx == 0:
                    color = QColor(100, 180, 244)
                elif layer_idx == 3:
                    color = QColor(255, 193, 7)
                else:
                    color = QColor(180, 180, 180)
                
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(QColor(255, 255, 255), 1.5))
                painter.drawEllipse(x - neuron_radius, y - neuron_radius, neuron_radius * 2, neuron_radius * 2)

                painter.setPen(QPen(QColor(240, 240, 240)))
                if layer_idx == 0:
                    painter.drawText(x - 55, y + 4, input_names[node_idx])
                elif layer_idx == 3:
                    painter.drawText(x + 12, y + 4, output_names[node_idx])

    def draw_connection(self, painter, pt1, pt2, weight):
        abs_w = abs(weight)
        if abs_w < 0.05:
            return
        width = max(0.5, min(4.0, abs_w * 1.5))
        alpha = max(20, min(220, int(abs_w * 120)))
        if weight > 0:
            color = QColor(76, 175, 80, alpha)
        else:
            color = QColor(229, 115, 115, alpha)
        painter.setPen(QPen(color, width))
        painter.drawLine(pt1[0], pt1[1], pt2[0], pt2[1])


class NetworkViewerDialog(QDialog):
    def __init__(self, agent, house_id, parent=None):
        super().__init__(parent)
        self.agent = agent
        self.setWindowTitle(f"Ev {house_id + 1} - Canlı Yapay Sinir Ağı (DQN)")
        self.resize(650, 480)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        info_panel = QFrame()
        info_panel.setStyleSheet("background-color: #2D2D2D; color: white;")
        info_layout = QHBoxLayout(info_panel)
        info_lbl = QLabel(
            f"<b>Ev {house_id + 1} Yapay Beyni</b><br/>"
            "Yeşil bağlar pozitif, Kırmızı bağlar negatif ağırlığı simgeler. "
            "Bağlantı kalınlığı sinirsel etki gücünü gösterir.", self
        )
        info_lbl.setStyleSheet("font-size: 10px; margin: 5px;")
        info_layout.addWidget(info_lbl)
        layout.addWidget(info_panel)

        self.painter_widget = NetworkPainterWidget(agent, self)
        layout.addWidget(self.painter_widget)

    def refresh(self):
        self.painter_widget.update()


class HouseWidget(QFrame):
    clicked = Signal(int)                   # Sol tık: Nöron penceresini açar
    personality_changed = Signal(int, str)  # Sağ tık: Kişilik değiştirme talebi fırlatır

    def __init__(self, house_id, parent=None):
        super().__init__(parent)
        self.house_id = house_id
        self.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.setLineWidth(1)
        self.init_ui()

    def mousePressEvent(self, event):
        """Sadece sol tıklandığında nöron izleme penceresini açar"""
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.house_id)

    def contextMenuEvent(self, event):
        """Sağ tıklandığında kişilik değiştirme menüsü açar"""
        menu = QMenu(self)
        
        action_cesur = menu.addAction("Cesur Yap (CESUR)")
        action_korkak = menu.addAction("Korkak Yap (KORKAK)")
        action_dengeli = menu.addAction("Dengeli Yap (DENGELİ)")
        
        # Menüyü tıklanan noktada göster
        action = menu.exec(self.mapToGlobal(event.pos()))
        
        if action == action_cesur:
            self.personality_changed.emit(self.house_id, "CESUR")
        elif action == action_korkak:
            self.personality_changed.emit(self.house_id, "KORKAK")
        elif action == action_dengeli:
            self.personality_changed.emit(self.house_id, "DENGELI")

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)
        
        self.personality_lbl = QLabel("DENGELİ", self)
        self.personality_lbl.setAlignment(Qt.AlignCenter)
        self.personality_lbl.setStyleSheet("""
            background-color: #757575; 
            color: white; 
            font-size: 8px; 
            font-weight: bold; 
            border-radius: 3px; 
            padding: 1px;
        """)
        
        self.title_label = QLabel(f"EV {self.house_id + 1}", self)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #333333;")
        
        self.status_label = QLabel("YASIYOR", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 10px;")
        
        self.score_label = QLabel("Skor: 0 | Gün: 1", self)
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setStyleSheet("font-size: 10px;")
        
        self.q_label = QLabel("Q (Ekmek/Peynir)\n0.0 / 0.0", self)
        self.q_label.setAlignment(Qt.AlignCenter)
        self.q_label.setStyleSheet("font-size: 9px; color: #555555;")
        
        layout.addWidget(self.personality_lbl)
        layout.addWidget(self.title_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.score_label)
        layout.addWidget(self.q_label)
        
        self.update_appearance("alive")

    def update_appearance(self, status, reason=None):
        if status == "alive":
            self.setStyleSheet("""
                HouseWidget {
                    background-color: #E8F5E9; 
                    border: 2px solid #81C784; 
                    border-radius: 6px;
                }
            """)
            self.status_label.setText("YASIYOR (Aktif)")
            self.status_label.setStyleSheet("color: #2E7D32; font-weight: bold; font-size: 10px;")
        elif status == "dead":
            if reason == "cheese":
                self.setStyleSheet("""
                    HouseWidget {
                        background-color: #E3F2FD; 
                        border: 2px solid #64B5F6; 
                        border-radius: 6px;
                    }
                """)
                self.status_label.setText("ÖLDÜ (Peynir / Kapan)")
                self.status_label.setStyleSheet("color: #1565C0; font-weight: bold; font-size: 10px;")
            else:
                self.setStyleSheet("""
                    HouseWidget {
                        background-color: #FFEBEE; 
                        border: 2px solid #E57373; 
                        border-radius: 6px;
                    }
                """)
                self.status_label.setText("ÖLDÜ (Yakalama Riski)")
                self.status_label.setStyleSheet("color: #C62828; font-weight: bold; font-size: 10px;")

    def update_data(self, data):
        if data["is_dead"]:
            self.update_appearance("dead", data["death_reason"])
        else:
            self.update_appearance("alive")
            
        self.score_label.setText(f"Skor: {data['score']} | Gün: {data['day']}")
        self.q_label.setText(f"Q (Ekmek / Peynir)\n{data['q_breadcrumbs']:.1f} / {data['q_cheese']:.1f}")
        
        p = data["personality"]
        self.personality_lbl.setText(p)
        if p == "CESUR":
            self.personality_lbl.setStyleSheet("background-color: #FF9800; color: white; font-size: 8px; font-weight: bold; border-radius: 3px;")
        elif p == "KORKAK":
            self.personality_lbl.setStyleSheet("background-color: #00BCD4; color: white; font-size: 8px; font-weight: bold; border-radius: 3px;")
        else:
            self.personality_lbl.setStyleSheet("background-color: #757575; color: white; font-size: 8px; font-weight: bold; border-radius: 3px;")


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        fig.patch.set_facecolor('#F5F5F5')
        self.axes.set_facecolor('#FFFFFF')
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)


class MainWindow(QMainWindow):
    def __init__(self, manager):
        super().__init__()
        self.manager = manager
        self.setWindowTitle("DQN Fare Simülasyonu Kontrol Paneli")
        self.resize(1350, 780)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.single_step)
        
        self.active_viewer = None
        self.active_viewer_house_id = -1
        
        self.init_ui()
        self.update_ui()
        
        if len(self.manager.history_avg_scores) > 0:
            self.plot_scores()
            self.log_message(f"Simülasyon kaldığı yerden yüklendi. Mevcut Episod: {self.manager.episode}")
        else:
            self.log_message("Simülasyon başlatıldı. Yeni beyinler hazırlandı.")

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # ----------------- SOL TARAF: 5x5 GRID VE GENEL BİLGİLER -----------------
        left_layout = QVBoxLayout()
        
        self.stats_group = QGroupBox("Genel Simülasyon Durumu")
        stats_layout = QHBoxLayout(self.stats_group)
        self.episode_lbl = QLabel("Episod: 1", self)
        self.avg_score_lbl = QLabel("Ortalama Skor: 0.0", self)
        self.survival_lbl = QLabel("Hayatta Kalan: 25 / 25", self)
        self.epsilon_lbl = QLabel("Ort. Keşif Hızı (Epsilon): 1.00", self)
        
        for lbl in [self.episode_lbl, self.avg_score_lbl, self.survival_lbl, self.epsilon_lbl]:
            lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #333333;")
            stats_layout.addWidget(lbl)
            
        left_layout.addWidget(self.stats_group)
        
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(8)
        self.house_widgets = []
        for i in range(25):
            hw = HouseWidget(house_id=i)
            hw.clicked.connect(self.open_network_viewer)
            hw.personality_changed.connect(self.on_house_personality_changed)  # Sağ tık sinyalini bağlama
            self.house_widgets.append(hw)
            row = i // 5
            col = i % 5
            self.grid_layout.addWidget(hw, row, col)
            
        left_layout.addLayout(self.grid_layout)
        main_layout.addLayout(left_layout, stretch=3)
        
        # ----------------- SAĞ TARAF: KONTROLLER, GRAFİK VE GÜNLÜK -----------------
        right_layout = QVBoxLayout()
        
        control_group = QGroupBox("Simülasyon Kontrolleri")
        control_layout = QVBoxLayout(control_group)
        
        self.btn_auto = QPushButton("Otomatik Oynat (Başlat)", self)
        self.btn_auto.clicked.connect(self.toggle_autoplay)
        self.btn_auto.setStyleSheet("height: 35px; font-weight: bold; background-color: #2E7D32; color: white;")
        
        self.btn_step = QPushButton("Sonraki Gün (Manuel)", self)
        self.btn_step.clicked.connect(self.single_step)
        self.btn_step.setStyleSheet("height: 30px; font-weight: bold;")
        
        speed_label = QLabel("Simülasyon Adım Hızı (MS):", self)
        speed_label.setStyleSheet("font-size: 10px; font-weight: bold;")
        self.slider_speed = QSlider(Qt.Horizontal, self)
        self.slider_speed.setRange(10, 1000)
        self.slider_speed.setValue(100)
        self.slider_speed.valueChanged.connect(self.update_timer_speed)
        
        self.btn_new_run = QPushButton("Yeni Episod Başlat", self)
        self.btn_new_run.clicked.connect(self.new_run)
        self.btn_new_run.setStyleSheet("height: 30px;")
        
        self.btn_save = QPushButton("Beyinleri & Gelişimi Kaydet", self)
        self.btn_save.clicked.connect(self.save_brains)
        self.btn_save.setStyleSheet("height: 30px; background-color: #1565C0; color: white;")
        
        self.btn_reset = QPushButton("Yapay Zekaları Sıfırla", self)
        self.btn_reset.clicked.connect(self.reset_all)
        self.btn_reset.setStyleSheet("height: 30px; background-color: #C62828; color: white;")
        
        control_layout.addWidget(self.btn_auto)
        control_layout.addWidget(self.btn_step)
        control_layout.addWidget(speed_label)
        control_layout.addWidget(self.slider_speed)
        control_layout.addWidget(self.btn_new_run)
        control_layout.addWidget(self.btn_save)
        control_layout.addWidget(self.btn_reset)
        
        right_layout.addWidget(control_group)
        
        log_group = QGroupBox("Sistem Günlüğü")
        log_layout = QVBoxLayout(log_group)
        self.txt_log = QTextEdit(self)
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet("background-color: #1E1E1E; color: #00FF00; font-family: Consolas; font-size: 9px;")
        log_layout.addWidget(self.txt_log)
        right_layout.addWidget(log_group, stretch=1)
        
        chart_group = QGroupBox("Gelişim Grafiği (Ortalama Skor)")
        chart_layout = QVBoxLayout(chart_group)
        self.canvas = MplCanvas(self, width=5, height=3, dpi=100)
        chart_layout.addWidget(self.canvas)
        right_layout.addWidget(chart_group, stretch=2)
        
        main_layout.addLayout(right_layout, stretch=1)

    def log_message(self, text):
        self.txt_log.append(text)
        self.txt_log.ensureCursorVisible()

    def on_house_personality_changed(self, house_id, new_personality):
        """Sağ tık menüsü üzerinden kişilik değiştirildiğinde çağrılır."""
        self.manager.personalities[house_id] = new_personality
        self.update_ui()
        self.log_message(f"[Kişilik Değişimi] Ev {house_id + 1} artık: {new_personality}")
        # Metadata kaydını tazeleyelim
        self.manager.save_metadata()

    def open_network_viewer(self, house_id):
        if self.active_viewer and self.active_viewer_house_id == house_id:
            self.active_viewer.raise_()
            self.active_viewer.activateWindow()
            return
            
        if self.active_viewer:
            self.active_viewer.close()

        agent = self.manager.agents[house_id]
        self.active_viewer = NetworkViewerDialog(agent, house_id, self)
        self.active_viewer_house_id = house_id
        
        self.active_viewer.finished.connect(self.clear_viewer_reference)
        self.active_viewer.show()

    def clear_viewer_reference(self):
        self.active_viewer = None
        self.active_viewer_house_id = -1

    def single_step(self):
        all_done = self.manager.step_all()
        self.update_ui()
        
        if self.active_viewer:
            self.active_viewer.refresh()
            
        if all_done:
            statuses = self.manager.get_all_statuses()
            avg_score = sum(s["score"] for s in statuses) / len(statuses)
            self.manager.history_avg_scores.append(avg_score)
            
            self.manager.save_all_agents()
            self.plot_scores()
            self.update_ui()
            
            self.log_message(f"[Episod {self.manager.episode} Bitti] Ortalama Skor: {avg_score:.2f} (Kayıt Edildi)")
            
            if self.manager.episode % 10 == 0:
                self.log_message(">> [P2P İletişim] Akranlar arası bilgi paylaşım evresi tetiklendi!")
                leader_idx, best_score = self.manager.share_knowledge()
                self.log_message(f">> Lider Fare: Ev {leader_idx + 1} (Skor: {best_score}). Tecrübeler harmanlandı.")
            
            if self.timer.isActive():
                self.manager.episode += 1
                self.manager.reset_simulation_run()
                self.update_ui()
                self.log_message(f"[Episod {self.manager.episode} Başlatıldı] Otomatik ilerliyor...")
            else:
                self.timer.stop()
                self.btn_auto.setText("Otomatik Oynat (Başlat)")
                self.btn_auto.setStyleSheet("height: 35px; font-weight: bold; background-color: #2E7D32; color: white;")

    def toggle_autoplay(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_auto.setText("Otomatik Oynat (Başlat)")
            self.btn_auto.setStyleSheet("height: 35px; font-weight: bold; background-color: #2E7D32; color: white;")
            self.log_message("Otomatik oynatma durduruldu.")
        else:
            self.timer.start(self.slider_speed.value())
            self.btn_auto.setText("Otomatik Oynat (Durdur)")
            self.btn_auto.setStyleSheet("height: 35px; font-weight: bold; background-color: #D84315; color: white;")
            self.log_message(f"Otomatik oynatma başlatıldı (Hız: {self.slider_speed.value()} MS).")

    def update_timer_speed(self):
        if self.timer.isActive():
            self.timer.start(self.slider_speed.value())

    def new_run(self):
        self.timer.stop()
        self.btn_auto.setText("Otomatik Oynat (Başlat)")
        self.btn_auto.setStyleSheet("height: 35px; font-weight: bold; background-color: #2E7D32; color: white;")
        
        self.manager.episode += 1
        self.manager.reset_simulation_run()
        self.update_ui()
        self.log_message(f"[Episod {self.manager.episode} Manuel Başlatıldı]")

    def save_brains(self):
        self.manager.save_all_agents()
        self.log_message("Tüm beyinler ve grafik gelişimi (simulation_meta.json) başarıyla kaydedildi.")

    def reset_all(self):
        self.timer.stop()
        self.btn_auto.setText("Otomatik Oynat (Başlat)")
        self.btn_auto.setStyleSheet("height: 35px; font-weight: bold; background-color: #2E7D32; color: white;")
        
        if self.active_viewer:
            self.active_viewer.close()
            
        self.manager.reset_brains()
        self.canvas.axes.clear()
        self.canvas.draw()
        self.update_ui()
        self.log_message("Tüm yapay zekalar ve diske kayıtlı geçmiş sıfırlandı.")

    def update_ui(self):
        statuses = self.manager.get_all_statuses()
        
        for i, status in enumerate(statuses):
            self.house_widgets[i].update_data(status)
            
        self.episode_lbl.setText(f"Episod: {self.manager.episode}")
        
        alive_count = sum(1 for s in statuses if not s["is_dead"])
        self.survival_lbl.setText(f"Hayatta Kalan: {alive_count} / 25")
        
        avg_score = sum(s["score"] for s in statuses) / len(statuses)
        self.avg_score_lbl.setText(f"Ortalama Skor: {avg_score:.2f}")
        
        avg_eps = sum(s["epsilon"] for s in statuses) / len(statuses)
        self.epsilon_lbl.setText(f"Ort. Epsilon: {avg_eps:.2f}")

    def plot_scores(self):
        self.canvas.axes.clear()
        self.canvas.axes.set_title("Ortalama Skor Gelişimi (Episod Bazlı)", fontsize=10, fontweight='bold')
        self.canvas.axes.set_xlabel("Episod", fontsize=8)
        self.canvas.axes.set_ylabel("Skor", fontsize=8)
        
        self.canvas.axes.plot(
            range(1, len(self.manager.history_avg_scores) + 1), 
            self.manager.history_avg_scores, 
            marker='o', color='#1565C0', linewidth=2
        )
        self.canvas.axes.grid(True, linestyle='--', alpha=0.5)
        self.canvas.draw()