from configparser import ConfigParser, ExtendedInterpolation
import os

class Parser(object):
    def __init__(self, config_path):
        assert os.path.exists(config_path), '{} not exists.'.format(config_path)
        self.config = ConfigParser(
            delimiters='=',
            interpolation=ExtendedInterpolation())
        self.config.read(config_path)

    @property
    def image_height(self):
        return self.config.getint('data', 'image_height')

    @property
    def image_width(self):
        return self.config.getint('data', 'image_width')

    @property
    def max_label(self):
        return self.config.getint('data', 'max_label')

    @property
    def batch_size(self):
        return self.config.getint('data', 'batch_size')
