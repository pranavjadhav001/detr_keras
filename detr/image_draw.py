import colorsys
from PIL import Image, ImageFont, ImageDraw
import numpy as np

class BboxDraw:
    def __init__(self,font_path,class_list):
        self.class_list = class_list
        hsv_tuples = [(x / len(self.class_list), 1., 1.)
                              for x in range(len(class_list))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        self.font_path = font_path
    def draw(self, image_path, boxes, labels):
        img = Image.open(image_path)
        font = ImageFont.truetype(font=self.font_path,
                            size=np.floor(3e-2*img.size[1]  + 0.5).astype('int32'))
        thickness = (img.size[0] + img.size[1]) // 300
        for box,j in zip(boxes,labels):
            label = self.class_list[j]
            draw = ImageDraw.Draw(img)
            label_size = draw.textsize(label, font)

            left, top, right, bottom = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(img.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(img.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[j])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[j])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        return np.array(img)