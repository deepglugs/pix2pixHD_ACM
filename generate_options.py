from options.test_options import TestOptions


class GenerateOptions(TestOptions):
    def initialize(self):
        TestOptions.initialize(self)
        self.parser.add_argument('--template', help="template file",
                                 default=None)
        self.parser.add_argument('--image', help="input image file",
                                 default=None)
        self.parser.add_argument('--output', help="output directory",
                                 default=None)
        self.parser.add_argument('--label', help="input label file",
                                 default=None)
        self.parser.add_argument('--replace', action="store_true",
                                 help="overwrite existing generated images if they exist")
        self.parser.add_argument('--live', action="store_true",
                                 help="do live generation. use with image and output")
        self.parser.add_argument('--feature_image', help="feature image file",
                                 default=None)
        self.parser.add_argument('--video_src', action="store_true",
                                 help="do live generation from a video source")
        self.parser.add_argument('--webp', action="store_true",
                                 help="save as webp lossless file")
