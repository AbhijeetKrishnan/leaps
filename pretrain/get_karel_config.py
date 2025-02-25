import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from pretrain.customargparse import flatten_keys, args_to_dict
from fetch_mapping import fetch_mapping

# Ref.: https://stackoverflow.com/a/34997118
class obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)

def dict2obj(d):
    return json.loads(json.dumps(d), object_hook=obj)

def get_karel_task_config(karel_task: str, seed: int, num_demo_per_program=None):
    if karel_task == 'cleanHouse':
        from pretrain.leaps_cleanhouse import config
    elif karel_task == 'harvester':
        from pretrain.leaps_harvester import config
    elif karel_task == 'fourCorners':
        from pretrain.leaps_fourcorners import config
    elif karel_task == 'randomMaze':
        from pretrain.leaps_maze import config
    elif karel_task == 'stairClimber':
        from pretrain.leaps_stairclimber import config
    elif karel_task == 'topOff':
        from pretrain.leaps_topoff import config
    
    flattened_config = {key: value for key, value in flatten_keys(config)}
    flattened_config['seed'] = seed
    if num_demo_per_program:
        flattened_config['num_demo_per_program'] = num_demo_per_program
    # print(flattened_config)
    flattened_config_obj = dict2obj(flattened_config)

    flattened_config_obj.task_file = config['rl']['envs']['executable']['task_file']
    flattened_config_obj.grammar = config['dsl']['grammar']
    flattened_config_obj.use_simplified_dsl = config['dsl']['use_simplified_dsl']
    flattened_config_obj.task_definition = config['rl']['envs']['executable']['task_definition']
    flattened_config_obj.execution_guided = config['rl']['policy']['execution_guided']
    _, _, flattened_config_obj.dsl_tokens, _ = fetch_mapping(os.path.join(BASE, 'mapping_karel2prl.txt'))
    flattened_config_obj.use_simplified_dsl = False

    unflattened_config_dict = args_to_dict(flattened_config_obj)
    # print(unflattened_config_dict)

    karel_task_config = { **unflattened_config_dict, 'args': flattened_config_obj }
    
    return karel_task_config