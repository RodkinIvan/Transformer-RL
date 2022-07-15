import numpy as np
import argparse
import os
import json
import time
import torch


HPARAMS_REGISTRY = {}


class Hyperparams(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value


vmpo_none = Hyperparams()
# model
vmpo_none.model = 'vmpo'
vmpo_none.state_rep = 'none'
# env
vmpo_none.env_name = 'rooms_watermaze'
vmpo_none.action_dim = 9
vmpo_none.action_list = np.array([
    [0, 0, 0, 1, 0, 0, 0],    # Forward
    [0, 0, 0, -1, 0, 0, 0],   # Backward
    [0, 0, -1, 0, 0, 0, 0],   # Strafe Left
    [0, 0, 1, 0, 0, 0, 0],    # Strafe Right
    [-20, 0, 0, 0, 0, 0, 0],  # Look Left
    [20, 0, 0, 0, 0, 0, 0],   # Look Right
    [-20, 0, 0, 1, 0, 0, 0],  # Look Left + Forward
    [20, 0, 0, 1, 0, 0, 0],   # Look Right + Forward
    [0, 0, 0, 0, 1, 0, 0],    # Fire.
])
HPARAMS_REGISTRY['vmpo_none'] = vmpo_none


ppo_none = Hyperparams()
ppo_none.update(vmpo_none)
# model
ppo_none.model = 'ppo'
HPARAMS_REGISTRY['ppo_none'] = ppo_none



def parse_args_and_update_hparams(H, parser, s=None):
    # args = parser.parse_args(s)
    # valid_args = set(args.__dict__.keys())
    # hparam_sets = [x for x in args.hparam_sets.split(',') if x]
    # for hp_set in hparam_sets:
    #     hps = HPARAMS_REGISTRY[hp_set]
    #     for k in hps:
    #         if k not in valid_args:
    #             raise ValueError(f"{k} not in default args")
    #     parser.set_defaults(**hps)
    H.update(parser.parse_args(s).__dict__)


def add_arguments(parser):

    # utils
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--desc', type=str, default='test')
    parser.add_argument('--hparam_sets', '--hps', type=str)
    parser.add_argument('--gpu', type=str, default='0')

    # model
    parser.add_argument('--model', type=str, default='vmpo', help='{vmpo|ppo}')
    parser.add_argument('--state_rep', type=str, default='coberl', help='{none|lstm|trxl|gtrxl}')
    parser.add_argument('--n_latent_var', type=int, default=10)
    parser.add_argument('--n_layer', type=int, default=1)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--mem_len', type=int, default=20)
    parser.add_argument('--emb_size', type=int, default=8)

    # env
    parser.add_argument('--env_name', type=str, default='CartPole-v1')
    parser.add_argument('--state_dim', type=int, default=4)
    parser.add_argument('--action_dim', type=int, default=2)
    parser.add_argument('--log_interval', type=int, default=40)
    parser.add_argument('--max_episodes', type=int, default=5000)
    parser.add_argument('--max_timesteps', type=int, default=1000)
    parser.add_argument('--update_timestep', type=int, default=50)
    parser.add_argument('--action_list', type=list, default=[0, 1])

    # training
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--K_epochs', type=int, default=4)
    parser.add_argument('--eps_clip', type=float, default=0.2)

    return parser


def logger(log_prefix):
    'Prints the arguments out to stdout, .txt, and .jsonl files'

    jsonl_path = f'{log_prefix}.jsonl'
    txt_path = f'{log_prefix}.txt'

    def log(*args, pprint=False, **kwargs):

        t = time.ctime()
        argdict = {'time': t}
        if len(args) > 0:
            argdict['message'] = ' '.join([str(x) for x in args])
        argdict.update(kwargs)

        txt_str = []
        args_iter = sorted(argdict) if pprint else argdict
        for k in args_iter:
            val = argdict[k]
            if isinstance(val, np.ndarray):
                val = val.tolist()
            elif isinstance(val, np.integer):
                val = int(val)
            elif isinstance(val, np.floating):
                val = float(val)
            argdict[k] = val
            if isinstance(val, float):
                val = f'{val:.5f}'
            txt_str.append(f'{k}: {val}')
        txt_str = ', '.join(txt_str)

        if pprint:
            json_str = json.dumps(argdict, sort_keys=True)
            txt_str = json.dumps(argdict, sort_keys=True, indent=4)
        else:
            json_str = json.dumps(argdict)

        print(txt_str, flush=True)

        with open(txt_path, "a+") as f:
            print(txt_str, file=f, flush=True)
        with open(jsonl_path, "a+") as f:
            print(json_str, file=f, flush=True)

    return log


def setup_save_dirs(H):
    H.save_dir = os.path.join(H.save_dir, H.desc)
    os.makedirs(H.save_dir, exist_ok=True)
    H.logdir = os.path.join(H.save_dir, 'log')


def set_up_hyperparams(s=None):
    H = Hyperparams()
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    parse_args_and_update_hparams(H, parser, s=s)
    setup_save_dirs(H)
    logprint = logger(H.logdir)
    for i, k in enumerate(sorted(H)):
        logprint(type='hparam', key=k, value=H[k])
    np.random.seed(H.seed)
    torch.manual_seed(H.seed)
    torch.cuda.manual_seed(H.seed)
    logprint('training model', H.desc, 'on', H.dataset)
    return H, logprint
