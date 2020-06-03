import copy
import os
import numpy as np

import torch

from generate_options import GenerateOptions
from models.models import create_model
import util.util as util
from PIL import Image
from torchvision import transforms

from data.gan_utils import get_txt_from_img_fn, encode_txt, get_images


def load_image(img_file, shape):
    img = Image.open(img_file).convert('RGB')

    preprocess = transforms.Compose([
        transforms.Scale(shape),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    return preprocess(img) * 255.0


def is_image(fn):

    return (fn.endswith(".png") or
            fn.endswith(".jpg") or
            fn.endswith(".webp") or
            fn.endswith(".jpeg"))


def do_generate(opt, model=None):
    img_file = opt.image
    out_dir = opt.output
    shape = (opt.loadSize, opt.loadSize)

    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.label_nc = 0
    opt.no_instance = True

    if model is None:
        model = create_model(opt)

    if os.path.isfile(img_file):
        img_files = [img_file]
    else:
        img_files = get_images(img_file)

    label_files = []

    if opt.label is not None:
        if os.path.isfile(opt.label):
            label_files = [opt.label]
        else:
            label_files = get_images(opt.label, exts=['.txt'])

    if not is_image(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)

    print(f"Generating {len(img_files)} images...")

    for img_file in img_files:

        img_out_fn = out_dir

        if not is_image(img_out_fn):
            # print(f"output {img_out_fn} is not an image. assuming directory")
            img_out_fn = os.path.basename(img_file)
            img_out_fn = os.path.join(out_dir,
                                      img_out_fn)

        img_out = os.path.basename(img_out_fn)

        if not opt.replace and os.path.isfile(img_out_fn):
            print(f"skipping {img_out_fn}")
            continue

        img = load_image(img_file, shape)

        label = torch.Tensor([0]).cuda()

        if opt.cond:
            label = get_txt_from_img_fn(img_out, label_files)
            if label is None:
                print(f"could not find label for {img_out}")

            label = encode_txt(label, img.size(2), model=opt.tokenizer)

        generated = model.inference(img.view(1, 3, *shape), label)

        img_out = Image.fromarray(
            util.tensor2im(generated.data[0]))

        print(f"creating output for {img_out_fn}")
        img_out.save(img_out_fn)


def do_template(opt):
    import yaml

    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.label_nc = 0
    opt.no_instance = True

    with open(opt.template) as f:
        template = yaml.load(f, Loader=yaml.FullLoader)

    template_dirname = os.path.dirname(opt.template)
    shape = (512, 512)

    label_files = []

    if "labels_path" in template:
        labels_dir = os.path.join(template_dirname, template['labels_path'])
        opt.label = labels_dir

    original_opt = copy.deepcopy(opt)

    model_opts = {}

    if "model_options" in template:
        for model in template["model_options"]:
            model_opts[model] = template["model_options"][model]
            # print(model_opts[model])

    for m in template["models"]:

        for model_name, imgs in m.items():
            print(f'model filename: {model_name}')

            opt.name = model_name

            if model_name in model_opts.keys():
                for key, val in model_opts[model_name].items():
                    try:
                        print(f"setting option {key}: {val}")
                        opt.__setattr__(key, val)
                    except Exception:
                        print(f"no such option {key}")
            else:
                print(f"not using options for {model_name}")
                print(f"available options: {model_opts.keys()}")

            model = create_model(opt)

            for img_set in imgs:
                for img_in, img_out_fn in img_set.items():

                    fn = os.path.join(template_dirname, img_in)

                    opt.image = fn

                    img_out_fn = os.path.join(template_dirname,
                                              img_out_fn)

                    print(f"creating output for {img_out_fn}")

                    opt.output = img_out_fn

                    do_generate(opt, model)

            opt = copy.deepcopy(original_opt)


def generate():
    opt = GenerateOptions().parse(save=False)

    if opt.template is not None:
        do_template(opt)

    else:
        do_generate(opt)


if __name__ == "__main__":
    generate()
