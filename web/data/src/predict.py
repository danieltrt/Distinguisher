import os
import sys
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import torch.utils.data

import pandas as pd
import cv2
from scipy import ndimage
from operator import itemgetter
import copy
import math
import statistics

### OCR
import easyocr

reader = easyocr.Reader(['en'], gpu=False)
reader_pt = easyocr.Reader(['pt'], gpu=False)

### RANSAC
from sklearn import linear_model, datasets
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(2048, 512),
                         nn.ReLU(),
                         nn.Dropout(0.2),
                         nn.Linear(512, 10),
                         nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.002)
model.to(device)

test_transforms = transforms.Compose([transforms.Resize((224, 244)),
                                      transforms.ToTensor(),
                                      ])

load_model = torch.load("/home/vmadmin/Distinguisher/web/data/src/models/model_classifier.pth",
                        map_location=torch.device('cpu'))
model.load_state_dict(load_model['model_state_dict'])
model.eval()


def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.to(device)
    output = model(image_tensor)
    index = output.data.cpu().numpy().argmax()
    return index


import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


from .engine import evaluate
from .utils import *
from .transforms import *


def get_transform():
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(ToTensor())
    return Compose(transforms)


def add_padding(pad, label_cropped):
    padding = np.ones(((int(label_cropped.shape[0]) + pad * 2), (int(label_cropped.shape[1]) + pad * 2), 3),
                      dtype=np.uint8) * 255
    padding[pad:int(label_cropped.shape[0]) + pad, pad:int(label_cropped.shape[1]) + pad] = label_cropped
    return padding


# ocr + pixel_data_ratio
def compute_bar_heights3(chart_type, filename, save_csv_path, scientific_notation, pixel_data_ratio, all_bars, maxi,
                         labels_ocr, axis_ocr):
    if pixel_data_ratio != 0:

        if chart_type >= 1:  # GROUPED / STACKED BAR CHART
            groups_bar_height = []
            for group in all_bars:
                bar_height = []
                for bar in group:
                    bar_height.append(bar[3] - bar[1])  # ymax - ymin
                groups_bar_height.append(bar_height)

            with open(save_csv_path, "w") as csv_file:
                if axis_ocr:
                    csv_file.write("\"" + axis_ocr[1] + "\"")
                else:
                    csv_file.write("\"X\"")
                for i in range(len(groups_bar_height[0])):
                    csv_file.write("," + "\"" + "color" + str(i) + "\"")
                csv_file.write("\n")

                for i in range(len(groups_bar_height)):
                    group = groups_bar_height[i]
                    if not labels_ocr:
                        csv_file.write("group{},".format(i))
                    else:
                        csv_file.write("\"" + labels_ocr[i].replace("\"", "&aspa") + "\"" + ",")
                    for j in range(len(group)):
                        bar = group[j]
                        if j == len(group) - 1:
                            csv_file.write("{}".format(bar * pixel_data_ratio))
                        else:
                            csv_file.write("{},".format(bar * pixel_data_ratio))
                    csv_file.write("\n")


        else:  # SIMPLE BAR CHART
            bar_height = []
            for bar in all_bars:
                bar_height.append(bar[3] - bar[1])  # ymax - ymin

            with open(save_csv_path, "w", newline='') as csv_file:
                if axis_ocr:
                    csv_file.write("\"" + axis_ocr[1] + "\"" + "," + "\"" + axis_ocr[0] + "\"" + "\n")
                else:
                    csv_file.write("col0,col1\n")
                for i in range(len(bar_height)):
                    if not labels_ocr:
                        csv_file.write("bar{},".format(i))
                    else:
                        csv_file.write("\"" + labels_ocr[i].replace("\"", "&aspa") + "\"" + ",")
                    if i == len(bar_height) - 1:
                        csv_file.write("{}".format(bar_height[i] * pixel_data_ratio))
                    else:
                        csv_file.write("{}\n".format(bar_height[i] * pixel_data_ratio))


def color_diferent(color1, color2):
    dif = []
    for c in range(3):
        dif.append(abs(color1[c] - color2[c]))
    for d in dif:
        if (d > 25):
            return True
    return False


def DFS(G, v, seen=None, path=None):
    if seen is None: seen = []
    if path is None: path = [v]

    seen.append(v)

    paths = []
    for t in G[v]:
        if t not in seen:
            t_path = path + [t]
            paths.append(tuple(t_path))
            paths.extend(DFS(G, t, seen[:], t_path))
    return paths


## TO VISUALIZE

def extract_chart_values(model, img, show, filename, calculate_heights=True, chart_type=0):
    # chart_type 0 - simple bar chart
    # chart_type 1 - grouped bar chart
    # chart_type 2 - stacked bar chart

    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

    img = img.permute(1, 2, 0)  # C,H,W_H,W,C
    img = (img * 255).byte().data.cpu()  # * 255
    img = np.array(img)  # tensor to ndarray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB -- para desenhar por cima as bounding boxes
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    original_img = copy.deepcopy(img)

    # VARIAVEIS PARA CALCULAR BARRAS
    bar_height = []
    MAX = sys.maxsize
    n_labels_barras = 0  # numero de labels do eixo x detetadas
    detected_bars = []  # barras detetadas com certeza >= 0.7
    other_detected_bars = []  # barras que foram detetadas mas a prediction tem um score < 0.7
    detected_bar_labels = []
    text_title = []  # titulo
    text_label = []  # labels dos eixos
    detected_numbers = []  # texto yy e notacao cientifica
    legenda_cor = []  # legendas cor das barras
    mean_width_bar = 0

    for i in range(prediction[0]['boxes'].cpu().shape[0]):
        xmin = round(prediction[0]['boxes'][i][0].item())
        ymin = round(prediction[0]['boxes'][i][1].item())
        xmax = round(prediction[0]['boxes'][i][2].item())
        ymax = round(prediction[0]['boxes'][i][3].item())

        label = prediction[0]['labels'][i].item()  # bar, text_bar, text_num, ...
        score = prediction[0]['scores'][i].item()

        if label == 1 and score >= 0.7:
            if (chart_type != 2):  # not stacked
                if (mean_width_bar == 0):
                    detected_bars.append([xmin, ymin, xmax, ymax])
                    mean_width_bar += (xmax - xmin)
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=2)
                    cv2.putText(img, 'bar', (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0),
                                thickness=1)
                else:
                    dont_add = False
                    mean_coord_new_bar = (xmin + xmax) / 2
                    for b in detected_bars:
                        mean_coord_bar = (b[0] + b[2]) / 2
                        if (abs(mean_coord_bar - mean_coord_new_bar) < (mean_width_bar / len(detected_bars) / 2)):
                            dont_add = True
                    if (not dont_add):
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=2)
                        cv2.putText(img, 'bar', (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0),
                                    thickness=1)
                        detected_bars.append([xmin, ymin, xmax, ymax])
                        mean_width_bar += (xmax - xmin)
            else:  # stacked
                if (not [xmin, ymin, xmax, ymax] in detected_bars):
                    dont_add = False
                    mean_width_bar += (xmax - xmin)
                    new_bar = [xmin, ymin, xmax, ymax]
                    for b in detected_bars:
                        if ((abs(b[0] - new_bar[0]) + abs(b[1] - new_bar[1])) <= 6) and (
                                (abs(b[2] - new_bar[2]) + abs(b[3] - new_bar[3])) <= 6):
                            dont_add = True
                    if not dont_add:
                        detected_bars.append([xmin, ymin, xmax, ymax])
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=1)
                        cv2.putText(img, 'bar', (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0),
                                    thickness=1)


        elif label == 1 and score < 0.7:
            other_detected_bars.append([xmin, ymin, xmax, ymax])

        elif label == 2 and score > 0.7:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 153, 0), thickness=2)
            cv2.putText(img, 'text_bar', (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 153, 0),
                        thickness=1)
            n_labels_barras += 1
            detected_bar_labels.append([xmin, ymin, xmax, ymax])

        elif label == 3 and score > 0.7:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)
            cv2.putText(img, 'text_num', (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0),
                        thickness=1)
            detected_numbers.append([xmin, ymin, xmax, ymax])

        elif label == 4 and score > 0.7:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (102, 102, 255), thickness=2)
            cv2.putText(img, 'text_label', (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (102, 102, 255),
                        thickness=1)
            text_label.append([xmin, ymin, xmax, ymax])

        elif label == 5 and score > 0.7:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 102, 153), thickness=2)
            cv2.putText(img, 'text_title', (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 102, 153),
                        thickness=1)
            text_title.append([xmin, ymin, xmax, ymax])
        elif (label == 6 and score > 0.7) or (chart_type == 2 and label == 6 and score > 0.5):
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 0), thickness=2)
            cv2.putText(img, 'legenda_cor', (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 0),
                        thickness=1)
            legenda_cor.append([xmin, ymin, xmax, ymax])

    # sometimes there are bar_labels recognized as axis label
    # if the axis label is at the same height then its a bar_label
    axis_label = []
    mean_Y_bar_labels = 0
    mean_height_bar_labels = 0
    for l in detected_bar_labels:
        mean_height_bar_labels += (l[3] - l[1]) / 2
        mean_Y_bar_labels += (l[3] + l[1]) / 2
    if detected_bar_labels:
        mean_height_bar_labels = mean_height_bar_labels / len(
            detected_bar_labels)  # ponto medio Y das labels das barras
        mean_Y_bar_labels = mean_Y_bar_labels / len(detected_bar_labels)

    for l in text_label:
        # se o ponto medio Y da text_label for igual a media das bar_labels entao foi detetado mal
        med = (l[3] + l[1]) / 2
        if ((med > (mean_Y_bar_labels - mean_height_bar_labels / 2)) and (
                med < (mean_Y_bar_labels + mean_height_bar_labels / 2))):
            # not an axis label but a bar label
            detected_bar_labels.append(l)
        else:
            axis_label.append(l)

    if detected_bars:
        mean_width_bar = mean_width_bar / len(detected_bars)

    if chart_type == 1:  # grouped bar chart

        # X axis label recognized as bar label
        for i in range(len(detected_bar_labels)):
            safe = 0
            for j in range(len(detected_bar_labels)):
                if j != i:
                    if (detected_bar_labels[i][1] < detected_bar_labels[j][3]):
                        safe = 1
            if (safe == 0):  # this 'detected_bar_labels' is actually a X axis label
                text_label.append(detected_bar_labels[i])
        for l in text_label:
            if l in detected_bar_labels:
                detected_bar_labels.remove(l)

        # divide bars in groups
        bars_in_groups = []
        bars_per_group = len(legenda_cor)
        detected_bars = sorted(detected_bars, key=itemgetter(0))  # ordenar pelo index 0 que é o X da barra

        # check distance between consecutive bars
        distance_between_bars = []  # diferenca do ponto max com o min da seguinte-> xmax + xmin / 2
        for i in range(1, len(detected_bars)):
            distance_between_bars.append(detected_bars[i][0] - detected_bars[i - 1][2])
        min_dist = min(distance_between_bars)
        if min_dist < 2:
            min_dist = 2
        cut = 0
        for i in range(len(distance_between_bars)):
            dist = distance_between_bars[i]
            if dist > (min_dist * 3.5):  # diff. group:  
                bars_in_groups.append(detected_bars[cut:i + 1])
                cut = i + 1
        bars_in_groups.append(detected_bars[cut:])

        # se numero de barras < numero labels(cor)   ou   numero barras diferente entre grupos
        # entao ha missing bars
        incongruity = 0
        it = iter(bars_in_groups)
        the_len = len(next(it))

        items = []
        for g in bars_in_groups:
            items.append(len(g))
        median_bars = statistics.median(items)
        if len(items) % 2 == 0:
            median_bars = items[math.ceil(len(items) / 2)]

        if not all(len(l) == the_len for l in it):
            # not all groups have same length
            incongruity = 1

            for group in bars_in_groups:
                if len(group) != len(legenda_cor):
                    incongruity = 1

        #### ACRESCENTAR AS BARRAS QUE FALTAM  
        detected_bar_labels = sorted(detected_bar_labels, key=itemgetter(0))
        if (incongruity):
            bars_in_groups = [[] for i in range(len(detected_bar_labels))]
            for bar in detected_bars:
                dist_to_label = []
                for group_label in detected_bar_labels:
                    dist_to_label.append(abs((bar[0] + bar[2]) / 2 - (group_label[0] + group_label[2]) / 2))

                index = dist_to_label.index(min(dist_to_label))
                bars_in_groups[index].append(bar)
            for i in range(len(bars_in_groups)):
                add_bar = []
                group = bars_in_groups[i]
                for j in range(1, len(group)):
                    if len(group) < len(legenda_cor):
                        # missing bar in this group
                        if (group[j][0] - group[j - 1][2]) > (mean_width_bar / 2):
                            how_many_bars = math.floor((group[j][0] - group[j - 1][2]) / mean_width_bar)
                            for k in range(how_many_bars):
                                if not add_bar:
                                    add_bar.append(
                                        [int(group[j - 1][2]), group[j - 1][3], group[j - 1][2] + int(mean_width_bar),
                                         group[j - 1][3]])
                                else:
                                    add_bar.append(
                                        [int(add_bar[-1][2]), add_bar[-1][3], add_bar[-1][2] + int(mean_width_bar),
                                         add_bar[-1][3]])
                bars_in_groups[i] = group + add_bar
                bars_in_groups[i] = sorted(bars_in_groups[i],
                                           key=itemgetter(0))  # ordenar pelo index 0 que é o X da barra

            # if still missing bars => beginning or end of group
            it = iter(bars_in_groups)
            the_len = len(next(it))
            if not all(len(l) == the_len for l in it):
                # not all groups have same length
                # bars before or after the group
                for i in range(len(bars_in_groups)):
                    group = bars_in_groups[i]
                    if len(group) > 0:
                        while len(group) < median_bars:
                            group_label = detected_bar_labels[i]
                            dist_left = abs((group[0][0] + group[0][2]) / 2 - (group_label[0] + group_label[2]) / 2)
                            dist_right = abs((group[-1][0] + group[-1][2]) / 2 - (group_label[0] + group_label[2]) / 2)
                            if (dist_left > dist_right):  # add to the right
                                group.append(
                                    [int(group[-1][2]), group[-1][3], group[-1][2] + int(mean_width_bar), group[-1][3]])
                            else:  # add to the left
                                group.insert(0, [int(group[0][0] - int(mean_width_bar)), group[0][3], group[0][0],
                                                 group[0][3]])

        all_bars = bars_in_groups

    ################# STACKED #################
    if chart_type == 2:
        # divide in groups
        bars_in_groups = []
        detected_bars = sorted(detected_bars, key=itemgetter(0))  # ordenar pelo index 0 que é o X da barra
        for bar in detected_bars:
            if (bars_in_groups == []):
                bars_in_groups.append([bar])
            else:  # percorrer grupos e encaixar no grupo ou criar um novo
                done = 0
                for group in bars_in_groups:
                    if ((not done) and abs(bar[2] - group[0][2]) < mean_width_bar / 2):
                        group.append(bar)
                        done = 1
                if not done:
                    bars_in_groups.append([bar])

        #### REMOVE BARS WITH THE SAME COLOR IN THE SAME GROUP
        for group in bars_in_groups:
            remove_bar = []
            for b1 in range(len(group)):
                for b2 in range(len(group)):

                    if b1 != b2:
                        center_x = int((group[b1][2] + group[b1][0]) / 2)
                        center_y = int((group[b1][3] + group[b1][1]) / 2)
                        center_x_other = int((group[b2][2] + group[b2][0]) / 2)
                        center_y_other = int((group[b2][3] + group[b2][1]) / 2)
                        color_1 = original_img[center_y, center_x]
                        color_2 = original_img[center_y_other, center_x_other]

                        if not color_diferent(color_1.tolist(), color_2.tolist()):
                            remove_bar.append(group[b2])
            for r in remove_bar:
                group.remove(r)

        it = iter(bars_in_groups)
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
            # not all groups have same length
            for group in bars_in_groups:
                remove_bar = []
                if len(group) != len(legenda_cor):
                    for b in group:
                        center_x = int((b[2] + b[0]) / 2)
                        center_y = int((b[3] + b[1]) / 2)
                        # cor do primeiro 1/3 != cor segundo 1/3 ---> wrong
                        color_1 = original_img[int((center_y + b[3]) / 2), center_x]
                        color_2 = original_img[int((center_y + b[1]) / 2), center_x]

                        if color_diferent(color_1.tolist(), color_2.tolist()):
                            remove_bar.append(b)

                for r in remove_bar:
                    group.remove(r)

            # sort
            for i in range(len(bars_in_groups)):
                bars_in_groups[i] = sorted(bars_in_groups[i], key=itemgetter(3))
                bars_in_groups[i].reverse()  # ordenar pelo index 3 que é o y da barra

            # ### check colors --> IF A GROUP HAS ALL COLORS:
            median_num_colors = statistics.median(new_list)
            if (median_num_colors > len(legenda_cor)):
                numbercolors = median_num_colors
            else:
                numbercolors = len(legenda_cor)

            for group in bars_in_groups:
                colors = []
                if len(group) == numbercolors:
                    for c in group:
                        center_x = int((c[2] + c[0]) / 2)
                        center_y = int((c[3] + c[1]) / 2)
                        colors.append(original_img[center_y, center_x].tolist())
                    break

            if colors == []:
                ######### ---------- GRAPH APPROACH
                bars_in_groups_colors = []

                for group in bars_in_groups:
                    colors = []
                    for c in group:
                        center_x = int((c[2] + c[0]) / 2)
                        center_y = int((c[3] + c[1]) / 2)
                        colors.append(original_img[center_y, center_x].tolist())
                    bars_in_groups_colors.append(colors)

                edges = []  # [['a', 'b'], ['b', 'c'], ...
                for group in bars_in_groups_colors:
                    for i in range(len(group)):
                        append = True
                        append2 = True
                        for (a, b) in edges:

                            if i == 0 and len(group) > 1:  # first color
                                if (not color_diferent(a, group[i])) and (not color_diferent(b, group[i + 1])):
                                    append = False
                            elif (i == len(group) - 1) and (len(group) > 1):
                                if (not color_diferent(a, group[i - 1])) and (not color_diferent(b, group[i])):
                                    append2 = False
                            elif len(group) > 2:
                                if (not color_diferent(a, group[i])) and (not color_diferent(b, group[i + 1])):
                                    append = False
                                if (not color_diferent(a, group[i - 1])) and (not color_diferent(b, group[i])):
                                    append2 = False

                        if append and i != len(group) - 1:
                            edges.append([group[i], group[i + 1]])
                        if append2 and i != 0:
                            edges.append([group[i - 1], group[i]])

                color_to_number = {}
                new_edges = []
                for edge in edges:
                    a = edge[0]
                    b = edge[1]
                    a_color = -1
                    b_color = -1
                    for k in color_to_number:
                        if not color_diferent(color_to_number[k], a):
                            a_color = k
                        if not color_diferent(color_to_number[k], b):
                            b_color = k
                    if a_color == -1:
                        a_color = len(color_to_number)
                        color_to_number[a_color] = a
                    if b_color == -1:
                        b_color = len(color_to_number)
                        color_to_number[b_color] = b

                    new_edges.append([a_color, b_color])

                first_color = new_edges[0][0]

                G = defaultdict(list)
                for (s, t) in new_edges:
                    G[s].append(t)

                all_paths = DFS(G, first_color)
                max_len = max(len(p) for p in all_paths)
                max_paths = [p for p in all_paths if len(p) == max_len]

                if (max_len > len(legenda_cor)):
                    numbercolors = max_len
                else:
                    numbercolors = len(legenda_cor)

                # TRANSFORMAR OS NUMEROS NAS CORES
                colors = []
                for x in range(len(max_paths[0])):
                    colors.append(color_to_number[max_paths[0][x]])

            ### missing bars
            for group in bars_in_groups:
                if len(group) != numbercolors:  # missing bar

                    # which color?
                    j = 0
                    missing = []
                    for i in range(len(colors)):
                        try:
                            b = group[j]
                            center_x = int((b[2] + b[0]) / 2) - 1
                            center_y = int((b[3] + b[1]) / 2)

                            if color_diferent(colors[i], original_img[center_y, center_x].tolist()):
                                # missing color i
                                missing.append(j)
                            else:
                                j += 1
                        except:
                            missing.append(j)
                    for index in missing:
                        if index != 0:
                            group.insert(index, [group[index - 1][0], group[index - 1][1] + 1, group[index - 1][2],
                                                 group[index - 1][1] + 2])
                        elif index != len(group) - 1 and len(group) != 0:
                            group.insert(index, [group[index + 1][0], group[index + 1][1] - 2, group[index + 1][2],
                                                 group[index + 1][1] - 1])
                        elif index == 0 and len(group) != 0:
                            group.insert(index,
                                         [group[index][0], group[index][1] - 2, group[index][2], group[index][1] - 1])
                        elif len(group) == 0:
                            ind = bars_in_groups.index(group)  # não é o primeiro grupo
                            if ind != 0:
                                other = bars_in_groups[ind - 1]
                                group.insert(index, [other[index - 1][0] + mean_width_bar * 2, other[0][1] - 2,
                                                     other[index - 1][2] + mean_width_bar * 2, other[0][1] - 1])
                            elif ind == 0 and len(bars_in_groups) > 1:
                                other = bars_in_groups[1]
                                group.insert(index, [other[index - 1][0] - mean_width_bar * 2, other[0][1] - 2,
                                                     other[index - 1][2] - mean_width_bar * 2, other[0][1] - 1])

        all_bars = bars_in_groups

    ####### SIMPLE #########
    if chart_type == 0:
        # If more #labels than #bars
        add_these_bars = []
        labels_with_no_bar = []
        # if len(detected_bar_labels) > len(detected_bars):
        if True:
            # find which label has a bar missing
            # if a label doesnt have a bar at a mean_width_bar/2 distance -> the correspondent bar is missing
            for label in detected_bar_labels:

                mean_coord_label = (label[0] + label[2]) / 2  # (xmin+xmax) / 2
                has_correspondent_bar = False
                for bar in detected_bars:
                    mean_coord_bar = (bar[0] + bar[2]) / 2
                    # se ponto medio (x) da barra está a uma distancia mean_width_bar/2 do ponto medio (x) label
                    if (abs(mean_coord_label - mean_coord_bar) < mean_width_bar / 2):
                        has_correspondent_bar = True

                if not has_correspondent_bar:
                    found = False
                    # search in "other_detected_bars" list
                    for other_bar in other_detected_bars:
                        mean_coord_bar_other = (other_bar[0] + other_bar[2]) / 2
                        # se ponto medio (x) da barra está a uma distancia mean_width_bar/2 do ponto medio (x) label
                        if ((abs(mean_coord_label - mean_coord_bar_other) < mean_width_bar / 2) and not found):
                            add_these_bars.append(other_bar)  # add new bar
                            found = True
                    if not found:
                        labels_with_no_bar.append(label)

        all_bars = detected_bars + add_these_bars

        all_bars = sorted(all_bars, key=itemgetter(0))  # ordenar pelo index 0 que é o X da barra

        add_more_bars = []
        label_has_now_bar = []

        if len(all_bars) > 1:  # if >1 bars found:

            # check if distance between first two bars and the rest makes sense
            distance_between_bars = []  # diferenca dos pontos medios -> xmax + xmin / 2
            for i in range(1, len(all_bars)):
                distance_between_bars.append(
                    (all_bars[i][2] + all_bars[i][0]) / 2 - (all_bars[i - 1][2] + all_bars[i - 1][0]) / 2)
            # calculate min (distance_between_bars)
            if distance_between_bars:  # se lista nao for vazia
                min_distance = min(distance_between_bars)
            # if there is a distance higher than the (min + half bar_width) ---> some bar missing here
            for i in range(len(distance_between_bars)):
                d = distance_between_bars[i]
                if (d > (min_distance + mean_width_bar / 2)):
                    # missing bar between i and i+1 bar
                    # if there is a label_with_no_bar between i and i+1 -> there is a missing bar there
                    for l in labels_with_no_bar:
                        if ((all_bars[i][2] < (l[0] + l[2]) / 2) and (all_bars[i + 1][2] > (l[0] + l[2]) / 2)):
                            # there is a bar in these coords
                            # i know the center point in X of the new bar
                            center_X = (((all_bars[i][2] + all_bars[i][0]) / 2) + (
                                    all_bars[i + 1][2] + all_bars[i + 1][0]) / 2) / 2
                            maxY = int((all_bars[i][3] + all_bars[i + 1][3]) / 2)  # ymax
                            minY = maxY - 2  # ALTERAR DEPOIS - VOU METER QUE É SÓ 2 PIXELS
                            if not ([int(center_X - mean_width_bar / 2), minY, int(center_X + mean_width_bar / 2),
                                     maxY] in add_more_bars):
                                add_more_bars.append(
                                    [int(center_X - mean_width_bar / 2), minY, int(center_X + mean_width_bar / 2),
                                     maxY])
                                label_has_now_bar.append(l)

            for found in label_has_now_bar:
                labels_with_no_bar.remove(found)

            add_these_bars = add_these_bars + add_more_bars  # just to draw the new ones
            # add the extra ones discovered
            all_bars = all_bars + add_more_bars
            all_bars = sorted(all_bars, key=itemgetter(0))  # - ordenar pelo index 0 que é o X da barra

            # if there are still labels without bars - são ou as primeiras ou ultimas:
            missing_bar_before = []
            missing_bar_after = []
            for l in labels_with_no_bar:
                if l[2] < all_bars[0][2]:  # xmax label < xmax primeira bar -> label é das primeiras
                    missing_bar_before.append(l)
                elif l[2] > all_bars[-1][2]:  # xmax label > xmax last bar
                    missing_bar_after.append(l)

            # bars before are ordered backwards and bars after are ordered normally - by X
            missing_bar_after = sorted(missing_bar_after, key=itemgetter(0))
            missing_bar_before = sorted(missing_bar_before, key=itemgetter(0))
            missing_bar_before.reverse()

            for m in missing_bar_after:
                dist = m[0] - all_bars[-1][0]
                if ((dist < (min_distance + mean_width_bar / 2)) and (dist > (
                        min_distance - mean_width_bar / 2))):  # if the distance between the last bar and this label is 

                    labels_with_no_bar.remove(m)
                    center_X = (m[0] + m[2]) / 2  # centro da barra é o centro da label
                    maxY = all_bars[-1][3]  # ymax
                    minY = maxY - 2  # 2 pixels altura
                    all_bars.append(
                        [int(center_X - mean_width_bar / 2), minY, int(center_X + mean_width_bar / 2), maxY])
                    add_these_bars.append(
                        [int(center_X - mean_width_bar / 2), minY, int(center_X + mean_width_bar / 2), maxY])
            for m in missing_bar_before:
                dist = all_bars[0][0] - m[0]
                if ((dist < (min_distance + mean_width_bar / 2)) and (dist > (
                        min_distance - mean_width_bar / 2))):  # if the distance between the last bar and this label is 

                    labels_with_no_bar.remove(m)
                    center_X = (m[0] + m[2]) / 2  # centro da barra é o centro da label
                    maxY = all_bars[0][3]  # ymax igual ao da barra seguinte
                    minY = maxY - 2  # 2 pixels altura
                    all_bars.insert(0, [int(center_X - mean_width_bar / 2), minY, int(center_X + mean_width_bar / 2),
                                        maxY])
                    add_these_bars.insert(0,
                                          [int(center_X - mean_width_bar / 2), minY, int(center_X + mean_width_bar / 2),
                                           maxY])

        for new_bars in add_these_bars:
            xmin, ymin, xmax, ymax = new_bars
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 102, 153), thickness=2)
            cv2.putText(img, 'bar', (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0),
                        thickness=1)

        # order bars and labels by X coordinate (xmin)
        all_bars = sorted(all_bars, key=itemgetter(0))  # ordenar pelo index 0 que é o X da barra

    detected_bar_labels = sorted(detected_bar_labels, key=itemgetter(0))
    # order numbers by Y coordinate (ymin) - ao contrario
    detected_numbers = sorted(detected_numbers, key=itemgetter(1))
    detected_numbers.reverse()

    # verify if "title" is a number misdetected
    distance_between_numbers = []
    for i in range(1, len(detected_numbers)):
        distance_between_numbers.append((detected_numbers[i - 1][3] + detected_numbers[i - 1][1]) / 2 - (
                detected_numbers[i][3] + detected_numbers[i][1]) / 2)
    if (len(distance_between_numbers) > 1):
        mean_dist = sum(distance_between_numbers) / len(distance_between_numbers)
        for title in text_title:
            distance_highest_label = abs(
                (title[3] + title[1]) / 2 - (detected_numbers[-1][3] + detected_numbers[-1][1]) / 2)
            height_of_label = (title[3] - title[1]) / 2
            # ponto medio Y do title a distancia mean_dist da label mais acima?
            if ((distance_highest_label < (mean_dist + height_of_label)) and (
                    distance_highest_label > (mean_dist - height_of_label))):
                # ponto medio X está perto da label anterior?
                label_x_mean = (title[2] + title[0]) / 2
                last_label_x_mean = (detected_numbers[-1][2] + detected_numbers[-1][0]) / 2
                largura_last_label = (detected_numbers[-1][2] - detected_numbers[-1][0])

                # mightbelabel < label + 1/2 largura_last_label
                if ((label_x_mean < (last_label_x_mean + largura_last_label / 2)) and (
                        label_x_mean > (last_label_x_mean - largura_last_label / 2))):
                    detected_numbers.append(title)

        for d in detected_numbers:
            if d in text_title:
                text_title.remove(d)

    detected_numbers = sorted(detected_numbers, key=itemgetter(1))
    detected_numbers.reverse()

    # Se o numero com menor ymin (mais acima) tiver xmin > os outros todos xmax => é a notação cientifica => nao interessa para ja
    scientific_notation = True
    for i in range(len(detected_numbers) - 1):
        if (detected_numbers[i][2] > detected_numbers[-1][0]):
            scientific_notation = False

    ### OPTICAL CHARACTER RECOGNITION - OCR
    maxi = 0
    numslist = []
    coordslist = []

    for i in range(len(detected_numbers)):
        num = detected_numbers[i]
        if (num[0] > 5):  # sometimes the beginning of the number is cropped
            number_cropped = original_img[num[1]:num[3], num[0] - 4:num[2] + 4]
            if num[1] > 5:
                number_cropped = original_img[num[1] - 4:num[3], num[0] - 4:num[2] + 4]
        else:
            number_cropped = original_img[num[1]:num[3], num[0]:num[2] + 2]

        # Add white padding to the image
        pad = 12
        number_cropped = add_padding(pad, number_cropped)

        # save image
        plt.imshow(number_cropped)
        plt.axis('off')

        # OCR
        IMAGE_PATH = os.path.join('/tmp/', (str(i) + ".png"))
        plt.savefig(IMAGE_PATH)
        result = reader.readtext(IMAGE_PATH)

        if (result):
            # erros comuns do OCR
            res = result[0][1].replace("Q", "0").replace("#", "0").replace("D", "0").replace("C", "0").replace("o", "0")

            if ((res.replace(",", "").replace(".", "")).isdecimal()):
                # quando reconhece 080 sem o ponto -> 0.80
                if ((res[0] == '0') and (not ("," in res)) and (not ("." in res))):
                    maxi = float(res[0] + '.' + res[1:])

                elif (float(res.replace(",", "").replace(".", "")) != 0 and (
                        ("." in str(maxi)) or maxi < 1)):  # o numero anterior é decimal entao nao substituir a virgula 
                    maxi = float(res.replace(",", "."))

                else:  # numero muito grande
                    maxi = float(res.replace(",", ""))

                numslist.append(maxi)
                coordslist.append((num[1] + num[3]) / 2)

    ### Check first by number of digits
    remove_index = []
    if sorted(numslist) != numslist:
        if (len(numslist) > 2):
            # percorrer lista e se num. anterior e num. seguinte tem os mesmos digitos mas o num. atual nao: not correct 
            for i in range(len(numslist) - 2):
                if ((len(str(int(numslist[i]))) == len(str(int(numslist[i + 2])))) and (
                        len(str(int(numslist[i + 1]))) != len(str(int(numslist[i]))))):
                    remove_index.append(i + 1)

    for ind in remove_index:
        numslist.pop(ind)
        coordslist.pop(ind)

    pixel_data_ratio = 0

    #### RANSAC regressor - check outliers
    X = np.array(numslist)

    X = X.reshape(-1, 1)
    y = np.array(coordslist)
    if (len(X) > 1):
        # Robustly fit linear model with RANSAC algorithm
        ransac = linear_model.RANSACRegressor()
        ransac.fit(X, y)
        inlier_mask = ransac.inlier_mask_  # [ True  True  True  True False]
        outlier_mask = np.logical_not(inlier_mask)
        # Predict data of estimated models
        line_X = np.arange(X.min(), X.max())[:, np.newaxis]
        line_y_ransac = ransac.predict(line_X)

        # discard outliers
        X = X[inlier_mask]
        y = y[inlier_mask]
        X = X.reshape(1, -1)
        X = X[0]

        ### pixel_data_ratio
        pixel_data_ratio_list = []
        for i in range(len(X) - 1):
            pixel_data_ratio_list.append(abs(X[i + 1] - X[i]) / abs(y[i + 1] - y[i]))
        if pixel_data_ratio_list:
            pixel_data_ratio = float(sum(pixel_data_ratio_list) / len(pixel_data_ratio_list))
        else:
            pixel_data_ratio = 0

    labels_ocr = []
    ##### DETETAR O TEXTO PRESENTE NAS LABELS X DAS BARRAS
    for i in range(len(detected_bar_labels)):
        label = detected_bar_labels[i]
        if (label[0] > 9 and label[2] + 8 < original_img.shape[1]):  # sometimes the beginning of the label is cropped
            label_cropped = original_img[label[1]:label[3], label[0] - 8:label[2] + 8]
            if label[1] > 7:
                label_cropped = original_img[label[1] - 6:label[3], label[0] - 6:label[2] + 6]
        else:
            label_cropped = original_img[label[1]:label[3], label[0]:label[2]]

        # Add white padding to the image
        pad = 12
        label_cropped = add_padding(pad, label_cropped)

        # save image
        plt.imshow(label_cropped)
        plt.axis('off')
        # OCR
        IMAGE_PATH = os.path.join('/tmp/', (str(i) + ".png"))
        plt.savefig(IMAGE_PATH)
        # plt.show() ########
        result = reader.readtext(IMAGE_PATH)
        result_pt = reader_pt.readtext(IMAGE_PATH)

        if result and result_pt:
            if result[0][2] > result_pt[0][2]:
                labels_ocr.append(result[0][1])
            else:
                labels_ocr.append(result_pt[0][1])
        elif result:
            labels_ocr.append(result[0][1])
        elif result_pt:
            labels_ocr.append(result_pt[0][1])
        else:
            labels_ocr.append("")

    if (chart_type == 0 and (len(all_bars) != len(labels_ocr))):
        labels_ocr = []
    if (chart_type == 1 and (len(all_bars) != len(labels_ocr))):
        labels_ocr = []

    ##### DETETAR O TEXTO PRESENTE NAS LABELS DO EIXO
    axis_ocr = []
    axis_label = sorted(axis_label, key=itemgetter(1))
    for i in range(len(axis_label)):
        label = axis_label[i]
        if (label[0] > 9 and label[2] + 8 < original_img.shape[1]):  # sometimes the beginning of the label is cropped
            label_cropped = original_img[label[1]:label[3], label[0] - 8:label[2] + 8]
            if label[1] > 8:
                label_cropped = original_img[label[1] - 7:label[3], label[0] - 7:label[2] + 6]
        else:
            label_cropped = original_img[label[1]:label[3], label[0]:label[2]]

        # Add white padding to the image
        pad = 12
        label_cropped = add_padding(pad, label_cropped)

        # rotation
        if i == 0:
            label_cropped = ndimage.rotate(label_cropped, -90, cval=255)

        # save image
        plt.imshow(label_cropped)
        plt.axis('off')
        # OCR
        IMAGE_PATH = os.path.join('/tmp/', (str(i) + ".png"))
        plt.savefig(IMAGE_PATH)
        result = reader.readtext(IMAGE_PATH)
        result_pt = reader_pt.readtext(IMAGE_PATH)

        if result and result_pt:
            if result[0][2] > result_pt[0][2]:
                axis_ocr.append(result[0][1])
            else:
                axis_ocr.append(result_pt[0][1])
        elif result:
            axis_ocr.append(result[0][1])
        elif result_pt:
            axis_ocr.append(result_pt[0][1])
        else:
            axis_ocr.append("")
    if len(axis_ocr) != 2:
        axis_ocr = []

    #########
    # CALCULAR TAMANHO DAS BARRAS

    if (pixel_data_ratio > 0):
        variable = 0
        if filename.endswith(".jpg"):
            csv_path = os.path.join('/tmp/', filename).replace(".jpg", ".csv")
        else:
            csv_path = os.path.join('/tmp/', filename).replace(".png", ".csv")
        compute_bar_heights3(chart_type, filename, csv_path, scientific_notation, pixel_data_ratio, all_bars, maxi,
                             labels_ocr, axis_ocr)

    if (show):
        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    if chart_type == 1:
        return np.shape(np.array(all_bars, dtype=object))
    if chart_type != 2:
        numero_barras = len(all_bars)
    else:
        numero_barras = 0
        new_list = []
        for g in bars_in_groups:
            numero_barras += len(g)
            new_list.append(len(g))
    return numero_barras


def chart_to_table(file, max, min):
    classes = ['grouped', 'simple', 'stacked']
    directory = r"/content/"
    image_name = "UEFA Nations League 2021.png"
    c = 0
    w = 0

    filename = os.path.join(directory, image_name)


    barray = Image.open(io.BytesIO(bytearray(file.read())))
    pred = predict_image(barray.convert("RGB"))
    classification_chart_type = pred

    device_cpu = torch.device('cpu')
    model_cpu = get_instance_segmentation_model(7)

    # SIMPLE BAR CHARTS
    if classification_chart_type == 1:
        model_cpu.load_state_dict(
            torch.load(r'/home/vmadmin/Distinguisher/web/data/src/models/model_simple_bar_charts_dict.pt',
                       map_location=device_cpu))
        classification = 0

    # GROUPED BAR CHARTS
    elif classification_chart_type == 0:
        model_cpu.load_state_dict(
            torch.load(r'/home/vmadmin/Distinguisher/web/data/src/models/model_grouped_bar_charts_dict.pt',
                       map_location=device_cpu))
        classification = 1

    # STACKED BAR CHARTS
    elif classification_chart_type == 2:
        model_cpu.load_state_dict(
            torch.load(r'/home/vmadmin/Distinguisher/web/data/src/models/model_stacked_bar_charts_dict.pt',
                       map_location=device_cpu))
        classification = 2

    # all
    device = device_cpu

    model_loaded = model_cpu

    rgba = np.array(barray)

    if (len(rgba.shape) == 3 and rgba.shape[2] == 4):  # (Height,Width,4)
        rgba[rgba[..., -1] == 0] = [255, 255, 255, 0]
        outside_img = Image.fromarray(rgba).convert("RGB")
    else:  # (Height,Width)
        outside_img = barray.convert("RGB")

    trans = get_transform()
    outside_img, x = trans(outside_img, None)
    n_bars = extract_chart_values(model_loaded, outside_img, True, "output_img", chart_type=classification)

    return n_bars
