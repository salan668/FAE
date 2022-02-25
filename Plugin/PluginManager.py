"""
All rights reserved. 
Author: Yang SONG (songyangmri@gmail.com)
"""
import json
from pathlib import Path


class Plugin(object):
    def __init__(self):
        self.format = 'EXE'
        self.name = None
        self.path = None
        self.figure = None
        self.description = None

    def LoadConfig(self, config_path):
        with open(str(config_path), 'r') as f:
            info = json.load(f)
        if 'path' not in info.keys():
            raise KeyError('path is not in the config path: {}'.format(config_path))
        self.path = config_path.parent / info['path']

        if 'name' not in info.keys():
            raise KeyError('name is not in the config path: {}'.format(config_path))
        self.name = info['name']

        if 'format' not in info.keys():
            self.format = 'EXE'
        else:
            self.format = info['format']

        if 'figure' not in info.keys():
            self.figure = None
        else:
            self.figure = config_path.parent / info['figure']

        if 'description' not in info.keys():
            self.description = None
        else:
            self.description = config_path.parent / info['description']


class PluginManager(object):
    def __init__(self):
        self.plugins = {}

    def LoadPlugin(self, plugin_folder: Path):
        self.plugins = {}
        if not plugin_folder.exists():
            return

        for one in plugin_folder.iterdir():
            if one.is_dir():
                config_path = one / 'config.json'
                if not config_path.exists():
                    print('{} does not include config.json'.format(one))
                    continue

                one_plugin = Plugin()
                one_plugin.LoadConfig(config_path)
                self.plugins[one_plugin.name] = one_plugin

