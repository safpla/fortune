from configparser import ConfigParser, ExtendedInterpolation
import json
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
    def image_channels(self):
        return self.config.getint('data', 'image_channels')

    @property
    def max_label(self):
        return self.config.getint('data', 'max_label')

    @property
    def batch_size(self):
        return self.config.getint('data', 'batch_size')

    @property
    def pretrain_model(self):
        return self.config.get('env', 'pretrain_model')

    @property
    def pretrain_model_file(self):
        return os.path.expanduser(self.config.get('env', 'pretrain_model_file'))

    @property
    def grad_lim(self):
        return self.config.getfloat('train', 'grad_lim')

    @property
    def lr_policy_params(self):
        params = self.config.get('train', 'lr_policy_params', fallback='{}')
        return json.loads(params)

    @property
    def train_data(self):
        return os.path.expanduser(self.config.get('data', 'train_data'))

    @property
    def valid_data(self):
        return os.path.expanduser(self.config.get('data', 'valid_data'))

    @property
    def test_data(self):
        return os.path.expanduser(self.config.get('data', 'test_data'))

    @property
    def timedelay_num(self):
        return self.config.getint('train', 'timedelay_num')

    @property
    def max_step(self):
        return self.config.getint('train', 'max_step')

    @property
    def summary_steps(self):
        return self.config.getint('train', 'summary_steps')
    @property
    def exp_dir(self):
        return os.path.expanduser(self.config.get('env', 'exp_dir'))

    @property
    def data_dir(self):
        return os.path.expanduser(self.config.get('env', 'data_dir'))


if __name__ == '__main__':
    config_path = '/home/leo/GitHub/fashionAI/config/skirt_length.cfg'
    config = Parser(config_path)
    print(config.data_dir)
    print(config.pretrain_model_file)
