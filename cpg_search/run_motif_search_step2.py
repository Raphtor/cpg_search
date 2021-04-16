from cpg_search.run_motif_search import *
from cpg_search.utils import *
import sys
from argparse import ArgumentParser
import yaml
import shutil

if __name__ == "__main__":
    nengo.rc.set('decoder_cache', 'enabled', "False")
    
   
    
    
    parser = ArgumentParser()
    parser.add_argument('metadata')
    parser.add_argument('--in-path', required=True)
    parser.add_argument('--out-path', required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--config',default=None)
    args = parser.parse_args()
    path = args.in_path
    outpath = args.out_path
    if args.config is None:
        try:
            config = os.path.join(path, 'config.yaml')
            with open(config) as fp:
                cfg = yaml.load(fp)
        except:
            pass
    elif os.path.exists(args.config):
        config = args.config
        with open(config) as fp:
            cfg = yaml.load(fp)
    else:
        raise ValueError('No config file provided or found in {}'.format(path))
    cfg = eval_cfg(cfg, globals())
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    shutil.copy(config, os.path.join(outpath,os.path.basename(config)))
    raw_df = pd.read_pickle(args.metadata)
    imms = generate_intermodule_matrices(**cfg['generate_intermodule_matrices'])
    rows = raw_df.iterrows()

    futures = []
    

    pool = multiprocessing.Pool(processes=4)
    i = 0
    for i,((j,row), imm) in enumerate(product(rows,imms)):
        mm, lr = row['_metadata'][0]
        
        kwds = dict(module_matrix=mm, lr_matrix=lr,intermodule_matrix=imm, metadata=(mm,lr,imm),**cfg['create_and_run_model'])
        
        test_fn = "{}.pkl".format(make_hash(kwds['metadata']))
        if not os.path.exists(os.path.join(outpath, test_fn)) or args.overwrite:
            # create_and_run_model(**kwds)
            futures.append(pool.apply_async(create_and_run_model,kwds=kwds))
    pbar = tqdm(total=len(futures))
    while len(futures):
        for i,f in enumerate(futures):
            if f.ready():
                save_return(f.get(), path=outpath)
                futures.pop(i)
                pbar.update(1)