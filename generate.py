import os
import numpy as np

import torch

from generate_options import GenerateOptions
from models.models import create_model
import util.util as util
from PIL import Image
from torchvision import transforms


def get_images(path):

    images = []
    exts = [".png", ".jpg", ".webp", ".jpeg"]

    for root, dirs, files in os.walk(path):
        for file in files:
            for ext in exts:
                if file.endswith(ext):
                    images.append(os.path.join(root, file))

    return images


def load_image(img_file, shape):
    img = Image.open(img_file).convert('RGB')

    preprocess = transforms.Compose([
        transforms.Scale(shape),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    return preprocess(img) * 255.0


def do_generate(opt):
    img_file = opt.image
    out_dir = opt.output
    shape = (512, 512)

    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.label_nc = 0
    opt.no_instance = True

    model = create_model(opt)

    if os.path.isfile(img_file):
        img_files = [img_file]
    else:
        img_files = get_images(img_file)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    for img_file in img_files:
        img = load_image(img_file, shape)
        generated = model.forward((img.view(1, 3, *shape),
                                  torch.Tensor([0]).cuda()))

        img_out = Image.fromarray(
            util.tensor2im(generated.data[0]))

        img_out_fn = os.path.basename(img_file)

        img_out_fn = os.path.join(out_dir,
                                  img_out_fn)

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

    for m in template["models"]:

        for model_filename, imgs in m.items():
            print(f'model filename: {model_filename}')

            opt.name = model_filename
            model = create_model(opt)

            for img_set in imgs:
                for img_in, img_out_fn in img_set.items():

                    fn = os.path.join(template_dirname, img_in)
                    try:
                        img = load_image(fn, shape)
                    except FileNotFoundError:
                        print(f"Error: could not find {fn}")
                        continue

                    generated = model.forward(
                        (img.view(1, 3, *shape), torch.Tensor([0]).cuda()))

                    img_out = Image.fromarray(
                        util.tensor2im(generated.data[0]))

                    img_out_fn = os.path.join(template_dirname,
                                              img_out_fn)

                    print(f"creating output for {img_out_fn}")
                    img_out.save(img_out_fn)


def generate():
    opt = GenerateOptions().parse(save=False)

    if opt.template is not None:
        do_template(opt)

    else:
        do_generate(opt)


if __name__ == "__main__":
    generate()
