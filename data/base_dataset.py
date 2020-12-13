import torch.utils.data as data
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import numpy as np
import random
import cv2
import util.util as util

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

global_sample_index = 0


def snapshot_img(img):
    global global_sample_index
    Image.fromarray(util.tensor2im(img)).save(f"samples/sample{global_sample_index}.png")
    global_sample_index +=1
    return img


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    rand_w = w
    rand_h = h

    if 'resize_and_crop' in opt.resize_or_crop:
        new_h = new_w = opt.loadSize
    elif 'scale_width' in opt.resize_or_crop:
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    if "random_load_size" in opt.resize_or_crop:
        rand_w = random.randint(opt.fineSize, opt.loadSize)
        rand_h = new_w * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))

    flip = random.random() > 0.5

    landscape = random.random() < opt.crop_prob

    rotate = random.randint(0, 359)

    erase = random.random() > 0.5
    jitter = random.random() > 0.5

    return {"random_load_size": (rand_w, rand_h),
            'crop_pos': (x, y),
            'flip': flip,
            "landscape": landscape,
            "rotate": rotate,
            "jitter": jitter,
            "erase": {"enabled": erase,
                      "i": random.randint(0, np.maximum(0, new_w - opt.fineSize)),
                      "j": random.randint(0, np.maximum(0, new_h - opt.fineSize)),
                      "h": random.randint(5, 20),
                      "w": random.randint(5, 20)
                      }}


def get_transform(opt, params,
                  method=Image.BICUBIC,
                  normalize=True,
                  is_A=False,
                  is_aug=False):
    transform_list = []

    if is_aug:
        transform_list.append(transforms.ToPILImage())

    did_crop_resize = False

    if 'stretch' in opt.resize_or_crop and is_A:
        did_crop_resize = True
        x = random.randint(0, 10)
        y = random.randint(0, 10)
        fine_mod = random.randint(0, 10)

        # print(f"will crop and resize {x}, {y} to {fine_mod}")
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize, method)))
        transform_list.append(transforms.Lambda(
            lambda img: __crop(img, (x, y), opt.loadSize - fine_mod)))

    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.resize_or_crop:
        rsize = opt.loadSize

        if 'random_load_size' in opt.resize_or_crop:
            rsize, _ = params['random_load_size']

        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, rsize, method)))

    if 'crop' in opt.resize_or_crop and not did_crop_resize:
        transform_list.append(transforms.Lambda(
            lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if 'shrink' in opt.resize_or_crop and is_A:
        x = random.randint(0, 10)
        transform_list.append(
            transforms.Lambda(lambda img: resize_border(img, x)))

    if "jitter" in opt.resize_or_crop and is_A:
        transform_list.append(transforms.ColorJitter(0.5, 0.5, 0.5))

    if "rotate" in opt.resize_or_crop and is_A:
        transform_list.append(transforms.RandomRotation([-2, 2]))

    if "landscape" in opt.resize_or_crop and params['landscape']:
        transform_list.append(
            transforms.Lambda(lambda img: landscape_crop(img, params['crop_pos'],
                                                         opt.fineSize, is_A)))
    elif "landscape" in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize, method)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(
            lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(
            lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if is_aug:
        rotate = transforms.Lambda(
            lambda img: transforms.functional.rotate(img, params["rotate"]))
        erase = transforms.Lambda(
            lambda img: transforms.functional.erase(img, params["erase"]['i'],
                                                    params["erase"]['j'],
                                                    params["erase"]['h'],
                                                    params["erase"]['w'],
                                                    0))

        if params["jitter"]:
            # transform_list.append(transforms.ColorJitter(0.5, 0.5, 0.5))
            pass

        transform_list.append(rotate)

        if params["erase"]["enabled"]:
            transform_list.append(erase)

        transform_list.append(transforms.Lambda(
            lambda img: __flip(img, params['flip'])))

        # transform_list.append(transforms.Lambda(snapshot_img))

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def resize_border(img, border):
    old_width, old_height = img.size[:2]

    width = old_width - (border * 2)
    height = old_height - (border * 2)

    img = img.resize((height, width))

    new_img = ImageOps.expand(img, border=border, fill='black')

    return new_img


def delandscape(img_orig, width, height):

    img = np.asarray(img_orig)

    old_size = img.shape[:2]
    h, w = old_size
    ratio_w = float(width) / w
    ratio_h = float(height) / h

    if w > h:
        new_size = (int(h * ratio_w), int(w * ratio_w))
    else:
        new_size = (int(h * ratio_h), int(w * ratio_h))

    img = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = abs(width - new_size[1])
    delta_h = abs(height - new_size[0])
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT,
                                 value=color)

    return Image.fromarray(new_img)


def landscape(img_orig):

    img = np.asarray(img_orig)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    if not len(contours):
        print(f"no contours found")
        return img_orig

    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    if (w < 64 or h < 64):
        print(f"image crop width ({w}) too small")
        print("using last good crop values")
        return img_orig

    crop = img[y:y + h, 0:img_orig.size[0]]

    return Image.fromarray(crop)


def landscape_crop(img, pos, fine_size, is_A):
    # img_l = landscape(img)

    new_h = img.size[1] / 1.773

    y_pos = int(new_h / 2)

    img_l = __crop(img, (0, y_pos), new_h, img.size[0])

    # if img_l.size[1] * 1.77 < img.size[1]:
    #    print("just scaling...")
    #    print(img_l.size)
    #    print(img.size)
    #    return __scale_width(img, fine_size)

    aspect = img_l.size[0] / fine_size
    h_aspect = img_l.size[1] / fine_size

    img = __crop(img_l, pos, int(
        img_l.size[1] * h_aspect), img_l.size[0] * h_aspect)

    # img.save(f"tmp/img_{'A' if is_A else 'B'}.png")

    # print(f"is A: {is_A}")
    # print(pos)
    # print(img_l.size)
    # print(img.size)
    # assert is_A

    img = delandscape(img, fine_size, fine_size)

    # print(f"final size: {img.size}")

    return img


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size, size_w=None):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size

    if size_w is not None:
        tw = size_w

    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return transforms.functional.hflip(img)
    return img
