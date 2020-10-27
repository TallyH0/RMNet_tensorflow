import tensorflow as tf
import numpy as np 
import cv2
import random
import math

class Data_loader():
    def __init__(self, txt_path, h, w, in_memory=False):
        self.paths = {}
        labels = []
        self.img_w, self.img_h = w, h
        self.in_memory = in_memory
        self.size = 0
        
        with open(txt_path) as f:
            for line in f:
                self.size += 1
                parse = line.strip().split('\t')
                if not int(parse[1]) in self.paths:
                    self.paths[int(parse[1])] = []

                if in_memory:
                    img = cv2.resize(cv2.imread(parse[0]), (w, h))
                    self.paths[int(parse[1])].append(img)
                else:
                    self.paths[int(parse[1])].append(parse[0])
                labels.append(int(parse[1]))
        
        self.identity = np.arange(np.amax(labels) + 1)

    def get_status(self):
        return self.size, len(self.identity)

    def batch(self, batch_size, param_erasing):
        batch_label = np.random.choice(self.identity, batch_size)
        prob, sl, sh, r1, r2 = param_erasing['prob'], param_erasing['sl'], param_erasing['sh'], param_erasing['r1'], param_erasing['r2']
        batch_image = []
        for label in batch_label:
            class_index = random.randint(0, len(self.paths[label]) - 1)
            if self.in_memory:
                img_src = self.paths[label][class_index]
                img_erasing = self.random_erasing(img_src, prob, sl, sh, r1, r2)
                batch_image.append(img_erasing)
            else:
                img = cv2.imread(self.paths[label][class_index])
                if img is None:
                    print(self.paths[label][class_index])
                img = cv2.resize(img, (self.img_w, self.img_h))
                img_erasing = self.random_erasing(img, prob, sl, sh, r1, r2)
                batch_image.append(img_erasing)

        return batch_image, batch_label

    def random_erasing(self, img_src, prob, sl, sh, r1, r2):
        img = img_src.copy()
        h, w, c = img.shape
        s = h * w
        p = random.random()
        if p > prob:
            return img
        else:
            s_e = random.uniform(sl, sh) * s
            r_e = random.uniform(r1, r2)
            h_e = min(round(math.sqrt(s_e * r_e)), h - 1)
            w_e = min(round(math.sqrt(s_e / r_e)), w - 1)

            x_0 = int(random.uniform(0, w))
            y_0 = int(random.uniform(0, h))
            x_1 = x_0 + w_e
            y_1 = y_0 + h_e

            if x_1 > w:
                x_1 = w -1
                x_0 = x_1 - w_e

            if y_1 > h:
                y_1 = h - 1
                y_0 = y_1 - h_e

            noise = np.random.uniform(0, 255, size=(h_e, w_e, 3))
            img[y_0 : y_1, x_0 : x_1, :] = noise

            return img

        

if __name__ == '__main__':
    txt_path = 'train_list_sample.txt'
    # data_loader = Data_loader(txt_path, 128, 64)
    # ds, ds_size, num_class = data_loader.dataset_from_txt(txt_path, 32, 1000, 300)

    loader = Data_loader(txt_path, 128, 64, True)
    n_data, n_class = loader.get_status()