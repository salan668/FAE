from pdfdocument.document import PDFDocument
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import Image
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph

import copy
from matplotlib.image import imread


class MyPdfDocument(PDFDocument):
    def __init__(self, *args, **kwargs):
        super(MyPdfDocument, self).__init__(*args, **kwargs)

    def generate_style(self, font_name=None, font_size=None):
        super(MyPdfDocument, self).generate_style(font_name, font_size)

        _styles = getSampleStyleSheet()

        self.style.normal = copy.deepcopy(_styles['Normal'])
        self.style.normal.alignment = 4
        self.style.normal.fontName = '%s' % self.style.fontName
        self.style.normal.fontSize = self.style.fontSize
        self.style.normal.firstLineIndent = 0.4 * cm
        self.style.normal.spaceBefore = self.style.fontSize * 1.5
        # normal.textColor = '#0e2b58'

        self.style.end_connection = copy.deepcopy(_styles['Normal'])
        self.style.end_connection.alignment = 0
        self.style.end_connection.fontName = '%s-Bold' % self.style.fontName
        self.style.end_connection.fontSize = self.style.fontSize
        self.style.end_connection.spaceBefore = self.style.fontSize * 3

        self.style.table_header = copy.deepcopy(self.style.normal)
        self.style.table_header.alignment = TA_CENTER

    def table_header(self, text, style=None):
        self.story.append(Paragraph(text, style or self.style.table_header))

    def image(self, image_path, width=8 * cm, style=None):
        img = imread(image_path)
        x, y = img.shape[:2]
        image = Image(image_path, width=width, height=width * x / y)

        self.story.append(image)

    def end_connect(self, text, style=None):
        self.story.append(Paragraph(text, style or self.style.end_connection))
