import sys
import math
import cv2
import cvzone

from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

from ultralytics import YOLO


class ImageProcessingDemo(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Анализ МРТ головного мозга')

        # Создаем кнопку
        self.button = QPushButton('Выбрать изображение', self)
        self.button.clicked.connect(self.process_image)

        # Создаем виджет для отображения изображения
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Создаем вертикальный layout и добавляем в него кнопку и виджет
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.image_label)

        self.model = YOLO('tumor_detector.pt')
        self.model_classes = self.model.model.names

    def process_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Выберите изображение', '',
                                                   'Images (*.png *.jpg *.jpeg *.bmp *.gif)')
        if file_name:
            image = cv2.imread(file_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = self.model(image, stream=True)

            for r in results:
                boxes = r.boxes

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    cvzone.cornerRect(image, (x1, y1, w, h))

                    conf = math.ceil((box.conf[0] * 100)) / 100

                    cls = box.cls[0]
                    name = self.model_classes[int(cls)]

                    cvzone.putTextRect(image, f'{name} 'f'{conf}', (max(0, x1), max(35, y1)), scale=1.5)

            height, width, channel = image.shape
            bytesPerLine = 3 * width
            q_image = QImage(image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)

            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))

            print(self.model_classes)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageProcessingDemo()
    window.show()
    sys.exit(app.exec())


# import sys
# import cv2
# from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog
# from PyQt6.QtGui import QPixmap, QImage
# from PyQt6.QtCore import Qt
#
#
# class ImageProcessingDemo(QWidget):
#     def __init__(self):
#         super().__init__()
#
#         self.setWindowTitle('Image Processing Demo')
#
#         # Создаем кнопку
#         self.button = QPushButton('Выбрать изображение', self)
#         self.button.clicked.connect(self.process_image)
#
#         # Создаем виджет для отображения изображения
#         self.image_label = QLabel(self)
#         self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
#
#         # Создаем вертикальный layout и добавляем в него кнопку и виджет
#         layout = QVBoxLayout(self)
#         layout.addWidget(self.button)
#         layout.addWidget(self.image_label)
#
#     def process_image(self):
#         file_name, _ = QFileDialog.getOpenFileName(self, 'Выберите изображение', '',
#                                                    'Images (*.png *.jpg *.jpeg *.bmp *.gif)')
#         if file_name:
#             image = cv2.imread(file_name)
#
#             # Ваш код предварительной обработки изображения с использованием OpenCV (cv2)
#             # Например, можно применить фильтр или изменить размер изображения
#
#             # Преобразование изображения из BGR в RGB, т.к. OpenCV использует BGR, а QPixmap ожидает RGB
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#             # Преобразование изображения в формат, подходящий для Qt
#             height, width, channel = image.shape
#             bytesPerLine = 3 * width
#             q_image = QImage(image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
#
#             # Преобразование QImage в QPixmap и отображение в QLabel
#             pixmap = QPixmap.fromImage(q_image)
#             self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
#
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = ImageProcessingDemo()
#     window.show()
#     sys.exit(app.exec())
