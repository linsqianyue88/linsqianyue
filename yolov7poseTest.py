# !usr/bin/env python
# encoding:utf-8
import matplotlib.pyplot as plt
import torch
import cv2
import torchvision
from torchvision import transforms
import numpy as np
from mss import mss
import time


def box_iou(box1, box2):
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
    # 绘制coco数据集的骨架和关键点
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255], [
                            51, 153, 255], [255, 153, 153],
                        [255, 102, 102], [255, 51, 51], [153, 255, 153], [
                            102, 255, 102], [51, 255, 51], [0, 255, 0],
                        [0, 0, 255], [255, 0, 0], [255, 255, 255]])
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
                [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7,
                               0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16,
                              16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 9  # 节点圆点半径
    num_kpts = len(kpts) // steps  # 三个一组 51/3=17
    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]  # 关键点坐标
        print(x_coord, y_coord)
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)),
                       radius, (int(r), int(g), int(b)), -1)
    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0] - 1) * steps]),
                int(kpts[(sk[0] - 1) * steps + 1]))
        pos2 = (int(kpts[(sk[1] - 1) * steps]),
                int(kpts[(sk[1] - 1) * steps + 1]))
        if steps == 3:
            conf1 = kpts[(sk[0] - 1) * steps + 2]
            conf2 = kpts[(sk[1] - 1) * steps + 2]
            if conf1 < 0.5 or conf2 < 0.5:
                continue
        if pos1[0] % 640 == 0 or pos1[1] % 640 == 0 or pos1[0] < 0 or pos1[1] < 0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0] < 0 or pos2[1] < 0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=6)  # 骨架厚度


def output_to_keypoint(output):
    targets = []
    for i, o in enumerate(output):
        kpts = o[:, 6:]
        o = o[:, :6]
        for index, (*box, conf, cls) in enumerate(o.detach().cpu().numpy()):
            targets.append(
                [i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf, *list(kpts.detach().cpu().numpy()[index])])
    return np.array(targets)


def non_max_suppression_kpt(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                            multi_label=False, labels=(), kpt_label=False, nc=None, nkpt=None):
    if nc is None:
        # number of classes
        nc = prediction.shape[2] - \
            5 if not kpt_label else prediction.shape[2] - 56
    xc = prediction[..., 4] > conf_thres  # candidates
    # (pixels) minimum and maximum box width and height
    min_wh, max_wh = 2, 4096
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS
    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)
              ] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)
        if not x.shape[0]:
            continue
        x[:, 5:5 + nc] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        box = xywh2xyxy(x[:, :4])
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            if not kpt_label:
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[
                    conf.view(-1) > conf_thres]
            else:
                kpts = x[:, 6:]
                conf, j = x[:, 5:6].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), kpts), 1)[
                    conf.view(-1) > conf_thres]
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float(
            ) / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded
    return output


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / \
            shape[0]  # width, height ratios
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


# 人体姿态估计处理函数
def pose_estimation():
    screenshot = mss()  # 截图
    # img = screenshot_value.grab({"left": 960 - 200, "top": 540 - 200, "width": 400, "height": 400})
    img = screenshot.grab({"left": 1, "top": 1, "width": 800, "height": 800})
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)  # 颜色空间转换(原图,格式)
    image = letterbox(img, 960, stride=64, auto=True)[0]  # 通过填充灰色 调整大小满足等比例调整
    image = transforms.ToTensor()(image)  # 转换图片到Tensor格式
    image = torch.tensor(np.array([image.numpy()]))  # 张量(数组([图片转矩阵]))
    if torch.cuda.is_available():  # 判断cuda可用
        image = image.half().to(device)  # 单精度float32转为半精度float16
    output, _ = model(image)  # 图片传入模型推演
    # 在推断结果上运行非最大抑制(NMS)(去冗余重复结果)
    output = non_max_suppression_kpt(
        output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
    with torch.no_grad():  # 不后向传播 不更新网络
        output = output_to_keypoint(output)  # 将模型输出结果转换为关键点格式
    # 图片分割 交换维度(RGB三个通道被pytorch存储tensor数据时打乱)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)  # 转换数据类型
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)  # 颜色空间转换
    for idx in range(output.shape[0]):  # 遍历推断结果
        # 鼻子 - 0, 脖子 - 1，右肩 - 2，右肘 - 3，右手腕 - 4，左肩 - 5，左肘 - 6，左手腕 - 7，右臀 - 8，右膝盖 - 9，
        # 右脚踝 - 10，左臀 - 11，左膝盖 - 12，左脚踝 - 13，右眼 - 14，左眼 - 15，有耳朵 - 16，左耳朵 - 17，背景 - 18.
        # 绘制coco数据集的骨架和关键点(图片,关键点,步长)
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
    # pyplot进行绘图
    plt.figure(figsize=(8, 8), dpi=100, facecolor=None, edgecolor=None)  # 输出尺寸
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.axis('off')  # 标尺线
    plt.imshow(nimg)
    plt.savefig('./output.jpg')
    # plt.show()


'''
https://github.com/WongKinYiu/yolov7
https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt
'''
# 加载已学习模型。
device = torch.device("cuda:0" if torch.cuda.is_available()
                      else "cpu")  # 识别CPU或GPU
weigths = torch.load('./yolov7-w6-pose.pt')  # 加载权重模型
model = weigths['model'].half().to(device)  # 移动所有模型参数和缓冲区到GPU
_ = model.eval()  # train()训练  eval()测试
pose_estimation()
