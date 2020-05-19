import math
import os
import numpy as np

import torch
from PIL import Image


def get_vocab(txt_dir_path, filter_tags=None, top=None):
    # Also supports .vocab file:
    if os.path.isfile(txt_dir_path):
        with open(txt_dir_path, 'r') as f:
            return np.sort(np.array(f.read().split(' ')))

    vocab = []
    occ = {}
    print(filter_tags)
    for root, dirs, files in os.walk(txt_dir_path):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), "r") as f:
                    try:
                        tags = f.read().split(', ')
                        for tag in tags:

                            if tag not in occ:
                                occ[tag] = 0

                            occ[tag] += 1

                            if filter_tags is not None:
                                for filter in filter_tags:
                                    if filter not in tag:
                                        continue
                                    else:
                                        vocab.append(tag)
                            else:
                                vocab.append(tag)
                    except Exception:
                        print(f"error processing {file}")

    vocab = list(set(vocab))

    if top is not None:
        new_vocab = []
        for k in sorted(occ, key=occ.get, reverse=True):
            if k in vocab:
                new_vocab.append(k)

            if len(new_vocab) >= top:
                break

        vocab = new_vocab

    return np.sort(np.array(vocab))


def txt_to_onehot(vocab, txt, split=", ", trim=[" ", '\n']):

    if not isinstance(txt, str):
        raise ValueError("txt_to_onehot() expects a string blob as text input")

    onehot = np.zeros((len(vocab),), dtype=np.float)

    for t in txt.split(split):
        for tr in trim:
            t = t.replace(tr, "")

        match = np.where(vocab == t)[0]

        if len(match) == 0:
            continue

        match = match[0]

        onehot[match] = 1

    return onehot

    one_hot = {}

    # setup the hash
    for word in vocab:
        one_hot[word] = 0

    txt = txt.split(split)

    for word in txt:
        one_hot[word] = 1

    return np.array(list(one_hot.values()))


def txt_from_onehot(vocab, onehot, thresh=0.2, return_confidence=False):

    filtered = np.array(onehot) >= thresh

    txt = []
    conf = []

    for index in range(len(onehot)):
        if filtered[index]:
            txt.append(vocab[index])
            conf.append(onehot[index])

    if return_confidence:
        return txt, conf

    return txt


def onehot_to_image(onehot, img_shape):
    sqrt = round(math.pow(len(onehot), 0.5))

    sqr_y = sqrt * sqrt

    y = np.concatenate((onehot, np.zeros((sqr_y - len(onehot)))))

    y = np.reshape(y, (-1, sqrt))

    img = Image.fromarray(y * 127.5 + 1)
    img = img.convert("L").convert("RGB")
    img = img.resize(img_shape, Image.BICUBIC)

    return np.asarray(img)


def image_combine(img1, img2):
    img2_mask = np.any(img2 != [0, 0, 0], axis=-1)  # any non-black pixel

    final_img = onehot_to_image(img1, (512, 512)).copy()

    final_img[img2_mask] = img2[img2_mask]

    return final_img


def get_images(path, exts=None):

    images = []
    if exts is None:
        exts = [".png", ".jpg", ".webp", ".jpeg"]

    for root, _, files in os.walk(path):
        for file in files:
            for ext in exts:
                if file.endswith(ext):
                    images.append(os.path.join(root, file))

    return images


def load_image(img_file, shape, normalize=True, pixel_format='RGB'):
    # print(img_file)
    img = Image.open(img_file).convert(pixel_format)
    img = img.resize(shape,
                     Image.BICUBIC)
    if normalize:
        img = np.asarray(img) / 127.5 - 1

    return img


class OthersMap:
    bn_others = {}


def get_txt_from_img_fn(img_fn, others):
    return get_other(img_fn, others)


def get_other(source, others):

    img_bn = os.path.basename(source)
    img_bn = os.path.splitext(img_bn)[0]

    if source in OthersMap.bn_others:
        return OthersMap.bn_others[source]

    for other in others:
        bn = os.path.basename(other)
        bn = os.path.splitext(bn)[0]

        # print(f"comparing {bn} ==? {img_bn}")

        if bn == img_bn:
            OthersMap.bn_others[source] = other
            return other


def encode_txt(txt_file, max_size=512, model='gpt2'):
    from transformers import GPT2Tokenizer

    import logging
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("transformers.tokenization_gpt2").setLevel(logging.ERROR)
    loggers = [logging.getLogger()]  # get the root logger
    loggers = loggers + [logging.getLogger(name)
                         for name in logging.root.manager.loggerDict]

    for logger in loggers:
        logger.setLevel(logging.ERROR)

    with open(txt_file, 'r') as f:
        data = f.read()
        inst_tensor = data
        inst_tensor = inst_tensor.replace('\n', '')

        inst_tensor = inst_tensor.split(',')

        # print(f"trying to use tokenizer from {model}")

        tokenizer = GPT2Tokenizer.from_pretrained(model,
                                                  pad_token='0')

        inst_tensor = tokenizer.encode(inst_tensor,
                                       add_prefix_space=True,
                                       max_length=max_size,
                                       pad_to_max_length=True,
                                       return_tensors='pt')

        # print(data)
        # print(inst_tensor)

        pad_len = max_size - inst_tensor.size(1)

        padding = torch.zeros((inst_tensor.size(0), pad_len),
                              device=inst_tensor.device,
                              dtype=torch.float)
        # print(f"padding shape: {padding.size()}")
        inst_tensor = torch.cat((inst_tensor.float(), padding), dim=1)

        return inst_tensor


def test_onehot_to_image():

    onehot = [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]

    import cv2

    image = onehot_to_image(onehot, (256, 256))

    cv2.imshow("muh onehot", image)
    cv2.waitKey(1000)


def test_vocab():

    vocab_dir = "."

    vocab = get_vocab(vocab_dir, top=1000)

    assert len(vocab) == 1000
