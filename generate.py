import copy
import os
import numpy as np
import time

import torch

from generate_options import GenerateOptions
from models.models import create_model
import util.util as util
from PIL import Image
from torchvision import transforms

from data.gan_utils import get_txt_from_img_fn, encode_txt, get_images, get_vocab, \
    txt_to_onehot, txt_from_onehot


def do_preprocess(img, shape):
    preprocess = transforms.Compose([
        transforms.Resize(shape),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    return preprocess(img) * 255.0


def load_image(img_file, shape):
    img = Image.open(img_file).convert('RGB')

    return do_preprocess(img, shape)


def is_image(fn):

    return (fn.endswith(".png") or
            fn.endswith(".jpg") or
            fn.endswith(".webp") or
            fn.endswith(".jpeg"))


def get_v4l2_loopback_device(dev, opt):
    import fcntl
    import v4l2

    format = v4l2.v4l2_format()
    format.type = v4l2.V4L2_BUF_TYPE_VIDEO_OUTPUT
    format.fmt.pix.field = v4l2.V4L2_FIELD_NONE
    format.fmt.pix.pixelformat = v4l2.V4L2_PIX_FMT_RGB24
    format.fmt.pix.width = opt.loadSize
    format.fmt.pix.height = opt.loadSize
    format.fmt.pix.bytesperline = opt.loadSize * 3
    format.fmt.pix.sizeimage = opt.loadSize * opt.loadSize * 3

    if not os.path.exists(dev):
        print(f"Error {dev} does not exist. did you 'sudo modprobe v4l2loopback video_nr=69'?")
        return

    device = open(dev, 'wb')

    fcntl.ioctl(device, v4l2.VIDIOC_S_FMT, format)

    return device


def video_source_generate(opt, model=None):
    import cv2

    out_dir = opt.output
    shape = (opt.loadSize, opt.loadSize)

    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.label_nc = 0
    opt.no_instance = True
    opt.replace = True

    vocab = get_vocab(opt.tokenizer, top=opt.loadSize)
    model = create_model(opt)

    ret = True

    cap = cv2.VideoCapture(opt.image, cv2.CAP_GSTREAMER)
    cap.open(opt.image)

    writer = None

    if opt.output:
        if "/dev/video" in opt.output:
            writer = get_v4l2_loopback_device(opt.output, opt)
        else:
            writer = cv2.VideoWriter(opt.output,
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     20,
                                     shape)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("failed to read from camera... reopening...")
            time.sleep(5)
            cap = cv2.VideoCapture(opt.image, cv2.CAP_GSTREAMER)
            cap.open(opt.image)

        if frame is None:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = do_preprocess(Image.fromarray(frame), shape)

        img_out = do_inference(img, opt, model, vocab, device=opt.device)

        img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

        if writer is not None:
            writer.write(img_out)

        cv2.imshow("window2", img_out)
        cv2.waitKey(1)


def live_generate(opt, model=None):
    import signal
    from PyQt5.QtCore import QFileSystemWatcher
    from PyQt5.QtCore import QCoreApplication, QTimer

    app = QCoreApplication([])

    out_dir = opt.output
    shape = (opt.loadSize, opt.loadSize)

    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.label_nc = 0
    opt.no_instance = True
    opt.replace = True

    model = create_model(opt)

    def on_file_changed(path):
        print(f"file {path} changed")

        if path not in watcher.files():
            if os.path.exists(path):
                print("new file(s). adding to watch")
                if os.path.isdir(path):
                    paths = get_images(path)
                else:
                    paths = [path]
                watcher.addPaths(paths)

        opt.image = path
        do_generate(opt, model)

    imgs = []

    if os.path.isdir(opt.image):
        imgs = get_images(opt.image)

    watcher = QFileSystemWatcher([opt.image, *imgs])
    watcher.directoryChanged.connect(on_file_changed)
    watcher.fileChanged.connect(on_file_changed)

    timer = QTimer()
    timer.start(500)  # You may change this if you wish.
    timer.timeout.connect(lambda: None)  # Let the interpreter run each 500 ms.

    def sigint_handler(*args):
        QCoreApplication.quit()

    signal.signal(signal.SIGTERM, sigint_handler)

    app.exec_()


def do_inference(img, opt, model, vocab=None,
                 img_out_file=None, feature_image=None, label_files=None,
                 device='cuda'):
    label = torch.Tensor([0]).to(device)

    shape = (opt.loadSize, opt.loadSize)

    if opt.cond or opt.feature_image:
        # print("using label!!!")
        label = None
        label_file = opt.label

        if label_file is not None:
            tmp_label = txt_to_onehot(vocab, opt.label,
                                      size=opt.vocab_size)

            if np.any(tmp_label):
                label = torch.from_numpy(tmp_label).float()

        if label is None:
            if label_file is not None and os.path.isfile(label_file):
                with open(label_file, 'r') as f:
                    data = f.read()
                    label = txt_to_onehot(vocab, data,
                                          size=opt.vocab_size)
                    label = torch.from_numpy(label).float()

            elif img_out_file is not None:
                label_file = get_txt_from_img_fn(img_out_file, label_files)

                if label_file is None:
                    print(f"could not find label for {img_out_file}")
                else:
                    with open(label_file, 'r') as f:
                        data = f.read()
                        label = txt_to_onehot(vocab, data,
                                              size=opt.vocab_size)
                        label = torch.from_numpy(label).float()

    if feature_image is not None:
        feature_image = feature_image.to(device)

    if label is None:
        label = torch.zeros((1, opt.vocab_size)).to(device)

    generated = model.inference(img.view(1, 3, *shape).to(device),
                                label.to(device),
                                feature_image)

    img_out = util.tensor2im(generated.data[0])

    return img_out


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
    opt.gpu_ids = []
    opt.isTrain = False
    device = opt.device

    feature_image = None
    if opt.feature_image is not None:
        feature_image = load_image(opt.feature_image, shape).view(1, 3, *shape)
        opt.use_encoded_image = True
        opt.instance_feat = True

    if model is None:
        model = create_model(opt).to(device)

    if os.path.isfile(img_file):
        img_files = [img_file]
    else:
        img_files = get_images(img_file)

    label_files = []

    if opt.label is not None:
        if os.path.isdir(opt.label):
            label_files = get_images(opt.label, exts=['.txt'])
        else:
            label_files = get_images(os.path.dirname(opt.image), exts=['.txt'])

    if not is_image(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)

    if opt.cond:
        vocab = get_vocab(opt.vocab, top=opt.loadSize)
    else:
        vocab = None

    if opt.vocab_size is None and vocab is not None:
        opt.vocab_size = len(vocab)

    print(f"Image source {img_file}")
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
        try:
            img = load_image(img_file, shape)
        except Exception as ex:
            print(f"could not load img: {img_file}")
            print(ex)
            continue

        img_out = do_inference(img, opt, model, vocab, img_out, feature_image,
                               label_files=label_files,
                               device=device)
        img_out = Image.fromarray(img_out)

        print(f"creating output for {img_out_fn}")
        if opt.webp:
            ext = os.path.splitext(img_out_fn)[1]

            if ext != ".webp":
                img_out_fn = img_out_fn.replace(ext, ".webp")

            img_out.save(img_out_fn, "WEBP", lossless=True)
        else:
            img_out.save(img_out_fn)

    del model


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

    print(f"torch device: {opt.device}")

    for m in template["models"]:

        for model_name, imgs in m.items():
            print(f'model filename: {model_name}')

            if opt.name is None:
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
    elif opt.live:
        live_generate(opt)
    elif opt.video_src:
        video_source_generate(opt)
    else:
        do_generate(opt)


if __name__ == "__main__":
    generate()
