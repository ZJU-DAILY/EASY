import argparse

import gc

from eval import *
from framework import FinalFantasy


def save_memory(dataset):
    dataset.x1 = dataset.x2 = None
    dataset.x2y_mask = dataset.y2x_mask = None
    # dataset.sim_x2y = dataset.sim_y2x = None
    return dataset


def train(name, dbp15k=True, args=None):
    nit = args.n_it
    epoch = args.epoch
    iter_type = args.iter_type
    refine_it = args.refine_begin
    device = args.device

    only_gnn = args.only_gnn
    cosine = args.no_neap
    # use_cache = not args.reload_sim
    lev = not args.no_lev

    print("Current dataset :", name)
    with torch.no_grad():
        prefix = "dataset/DBP15K/processed/" if dbp15k else "dataset/SRPRS/processed/"
        dataset = torch.load(prefix + name + ".pt")[0].to(device)
        dataset.ent1_len = dataset.x1.size(0)
        dataset.ent2_len = dataset.x2.size(0)
        if cosine:
            dataset.cosine_x2y = cosine_sim(dataset.x1, dataset.x2)
            dataset.cosine_y2x = dataset.cosine_x2y.t()
        dataset = save_memory(dataset)
        dataset_name = "dbp15k_" + name if dbp15k else "srprs_" + name

        print("-- Begin train")
        model_name = args.model
        model = FinalFantasy(dataset,
                             name=dataset_name,
                             model=model_name,
                             srp=not dbp15k,
                             fuse_semantic=not cosine,
                             fuse_lev_dist=lev).to(device)

    model.train_myself(num_it=nit, refine_begin=refine_it, epoch=epoch, iter_type=iter_type, only_gnn=only_gnn)
    return model.log_each_it


def save_result(r, path='result.json'):
    with open(path, 'w') as f:
        f.write(to_json(r))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pair', type=str, default='all_srprs',
                        help='[zh_en, ja_en, fr_en] for DBP15k, '
                             '[en_fr, en_de] for SRPRS, '
                             '[all_dbp15k, all_srprs, all] for running multiple datasets')
    parser.add_argument('--model', type=str, default='rrea',
                        help='backbone model for structure-based representation learning'
                             '\nsupported: [mraea, rrea, gcn-align]')
    parser.add_argument('--no_neap', action='store_true', default=False,
                        help='replace neap similarity by cosine similarity')
    parser.add_argument('--no_lev', action='store_true', default=False,
                        help='not fusing lev distance')
    parser.add_argument('--iter_type', type=str, default='ours',
                        help='iteration methods, supported: [ours, mraea, dat, th, mwgm, none]'
                             '\n mraea iteration only support [mraea, rrea] models')
    parser.add_argument('--n_it', type=int, default=20,
                        help='number of iteration')
    parser.add_argument('--epoch', type=int, default=30,
                        help='number of epochs to train the structure-based model')
    parser.add_argument('--refine_begin', type=int, default=1,
                        help='number of iteration that begin refine')
    parser.add_argument('--result_path', type=str, default='final_result.json',
                        help='path to save experiment results as JSON format')
    parser.add_argument('--random_seed', type=int, default=2021,
                        help='SIGIR 2021')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--only_gnn', default=False, action='store_true')
    # parser.add_argument('--reload_sim', default=False, action='store_true')

    args = parser.parse_args()
    set_seed(args.random_seed)
    pair = args.pair
    result = {}
    pairs = [pair]
    if 'all' in pair:
        pairs = DBP15K_PAIRS + SRPRS_PAIRS
        if 'dbp15k' in pair:
            pairs = DBP15K_PAIRS
        elif 'srprs' in pair:
            pairs = SRPRS_PAIRS
    for pair in pairs:
        dbp15k = pair in DBP15K_PAIRS
        result[pair] = train(pair, dbp15k, args)
        save_result(result, args.result_path)
        gc.collect()
