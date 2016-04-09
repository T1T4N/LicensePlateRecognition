from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QDesktopWidget, QGridLayout,
                             QGroupBox, QListWidget, QListWidgetItem, QAbstractItemView,
                             QCheckBox, QVBoxLayout, QFileDialog)
from detector import ThresholdBlurDetector, MorphologyTransformDetector, CannyDetector


class MainWidget(QWidget):
    def __init__(self):
        super(MainWidget, self).__init__()
        self.selected_files = []
        self.initUI()

    def initUI(self):
        grid_layout = QGridLayout()
        grid_layout.setObjectName("grid_layout")
        grid_layout.setContentsMargins(10, 10, 10, 10)

        grp_input = QGroupBox()
        grp_input.setObjectName("grp_input")
        grp_input.setTitle("Input images")

        self.lst_images = QListWidget(grp_input)
        self.lst_images.setObjectName("lst_images")
        self.lst_images.setSelectionMode(QAbstractItemView.NoSelection)

        self.btn_load = QPushButton()
        self.btn_load.setObjectName("btn_load")
        self.btn_load.setText("1. Load images")
        self.btn_load.clicked.connect(self.load_files)

        input_layout = QGridLayout()
        input_layout.setContentsMargins(0, 0, 0, 0)
        # Set the stretch
        input_layout.setColumnStretch(0, 0)  # row, stretch
        input_layout.setColumnStretch(3, 0)
        input_layout.setRowStretch(0, 0)
        input_layout.setRowStretch(3, 0)
        # Add widgets
        input_layout.addWidget(self.lst_images, 2, 2)
        input_layout.addWidget(self.btn_load, 3, 2)
        grp_input.setLayout(input_layout)

        grid_layout.addWidget(grp_input, 0, 0, 3, 1)

        grp_detectors = QGroupBox()
        grp_detectors.setObjectName("grp_detectors")
        grp_detectors_layout = QVBoxLayout()

        self.chk_canny = QCheckBox(grp_detectors)
        self.chk_canny.setObjectName("chk_canny")
        self.chk_canny.setText("CannyDetector")
        grp_detectors_layout.addWidget(self.chk_canny)

        self.chk_morphology = QCheckBox(grp_detectors)
        self.chk_morphology.setObjectName("chk_morphology")
        self.chk_morphology.setText("MorphologyTransformDetector")
        grp_detectors_layout.addWidget(self.chk_morphology)

        self.chk_thresh = QCheckBox(grp_detectors)
        self.chk_thresh.setObjectName("chk_thresh")
        self.chk_thresh.setText("ThresholdBlurDetector")
        self.chk_thresh.setChecked(True)
        grp_detectors_layout.addWidget(self.chk_thresh)

        grp_detectors.setLayout(grp_detectors_layout)
        grid_layout.addWidget(grp_detectors, 3, 0, 1, 1)

        self.btn_process = QPushButton()
        self.btn_process.setObjectName("btn_process")
        self.btn_process.setText("2. Process")
        self.btn_process.setEnabled(False)
        self.btn_process.clicked.connect(self.process_files)

        # addWidget(*Widget, row, column, rowspan, colspan)
        grid_layout.addWidget(self.btn_process, 4, 0, 1, 1)

        self.setLayout(grid_layout)
        self.setToolTip('This is a <b>QWidget</b> widget')
        self.setGeometry(300, 300, 400, 400)
        self.setWindowTitle('License Plate Number Recognition')
        self.center()
        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def load_files(self):
        dialog_result = QFileDialog.getOpenFileNames(self, 'Open file', '.',
                                                     "JPEG (*.jpg *.jpeg);;All files (*.*);;PNG (*.png)")
        if dialog_result[0]:
            self.selected_files = dialog_result[0]
            self.btn_process.setEnabled(True)
            for file_path in self.selected_files:
                item = QListWidgetItem(file_path)
                self.lst_images.addItem(item)

    def process_files(self):

        pass
