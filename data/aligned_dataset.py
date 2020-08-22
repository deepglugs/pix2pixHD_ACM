import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
from data.gan_utils import get_txt_from_img_fn, encode_txt, get_vocab, txt_to_onehot


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        # input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        # input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
            self.B_paths = sorted(make_dataset(self.dir_B))

        self.label_files = []

        if opt.cond or opt.instance_feat:

            self.dir_tags = os.path.join(opt.dataroot, "tags")
            self.label_files = sorted(make_dataset(self.dir_tags,
                                                   exts=['.txt']))

            tags = self.dir_tags

            if os.path.isfile(self.opt.vocab):
                tags = self.opt.vocab

            self.vocab = get_vocab(tags, top=opt.loadSize)

            assert len(self.label_files), f"Could not find any label files in {opt.dataroot}"

        # instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        # load precomputed instance-wise encoded features
        if opt.load_features:
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' %
                  self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):
        # input A (label maps)
        A_path = self.A_paths[index]
        try:
            A = Image.open(A_path)
        except Exception as ex:
            print(f"error opening {A_path}: {ex}")
            raise Exception(ex)

        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params, is_A=True)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(
                self.opt, params, method=Image.NEAREST, normalize=False,
                is_A=True)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        # input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]
            try:
                B = Image.open(B_path).convert('RGB')
            except Exception as ex:
                print(f"error opening {B_path}: {ex}")
                raise Exception(ex)
            transform_B = get_transform(self.opt, params)
            B_tensor = transform_B(B)

        # if using instance maps
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))

        label_file = get_txt_from_img_fn(self.A_paths[index],
                                         self.label_files)

        if label_file is not None:
            # inst_tensor = encode_txt(label_file, A_tensor.size(2),
            #                         model=self.opt.tokenizer)
            with open(label_file, 'r') as f:
                inst_tensor = txt_to_onehot(self.vocab, f.read(),
                                            size=A_tensor.size(2))
        elif self.opt.cond:
            raise Exception(f"label file for {self.A_paths[index]} not found")

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'
