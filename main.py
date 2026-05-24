import sys
from PySide6.QtWidgets import QApplication
from src.simulation import SimulationManager
from src.ui import MainWindow

def main():
    # PySide6 application initialization
    app = QApplication(sys.argv)
    
    # Initialize the simulation manager with 25 houses
    manager = SimulationManager(num_houses=25)
    
    # Create the main window and pass the simulation manager
    window = MainWindow(manager)
    window.show()
    
    # Start the Qt application event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
