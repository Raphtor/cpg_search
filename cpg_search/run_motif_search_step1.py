from cpg_search.run_motif_search import *

from argparse import ArgumentParser
if __name__ == "__main__":
    nengo.rc.set('decoder_cache', 'enabled', "False")

    parser = ArgumentParser()
    parser.add_argument('--path', default='./data')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--mm-scale',type=int)
    parser.add_argument('--lr-scale',type=int)
    args = parser.parse_args()
    path =  args.path
    mms = generate_module_matrices(4,scale=args.mm_scale)
    lrs = generate_lr_matrices(4,lim=2,scale=args.lr_scale)
    if not os.path.exists(path):
        os.makedirs(path)


    futures = []
    pool = multiprocessing.Pool(processes=4)
    for i,(mm, lr) in enumerate(product(mms,lrs)):
        kwds = dict(module_matrix=mm, lr_matrix=lr, metadata=(mm,lr))
        test_fn = "{}.pkl".format(make_hash(kwds['metadata']))
        if not os.path.exists(os.path.join(path, test_fn)) or args.overwrite:
            futures.append(pool.apply_async(createa_and_run_model,kwds=kwds))
    pbar = tqdm(total=len(futures))
    while len(futures):
        for i,f in enumerate(futures):
            if f.ready():
                save_return(f.get(), path=path)
                futures.pop(i)
                pbar.update(1)
            
        
    
        