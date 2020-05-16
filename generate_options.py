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
