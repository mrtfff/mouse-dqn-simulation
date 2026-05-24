# main.py
import sys
from PySide6.QtWidgets import QApplication
from src.simulation import SimulationManager
from src.ui import MainWindow

def main():
    app = QApplication(sys.argv)
    
    # Simülasyon yöneticisini oluştur
    manager = SimulationManager()
    
    # Görsel arayüzü başlat ve yöneticisi ile ilişkilendir
    window = MainWindow(manager)
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
