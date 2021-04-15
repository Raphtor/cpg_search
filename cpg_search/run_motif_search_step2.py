from run_motif_search import *
import sys
from argparse import ArgumentParser



if __name__ == "__main__":
    nengo.rc.set('decoder_cache', 'enabled', "False")
    
   
    
    
    parser = ArgumentParser()
    parser.add_argument('metadata')
    parser.add_argument('--in-path', default='./data')
    parser.add_argument('--out-path', default='./data_s2')
    parser.add_argument('--overwrite', action='store_true')

    parser.add_argument('--imm-scale',type=int)
    args = parser.parse_args()

    path = args.in_path
    outpath = args.out_path
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    raw_df = pd.read_pickle(args.metadata)
    imms = generate_intermodule_matrices(4,scale=args.imm_scale)
    rows = raw_df.iterrows()

    futures = []
    

    pool = multiprocessing.Pool(processes=4)
    i = 0
    for i,((j,row), imm) in enumerate(product(rows,imms)):
        mm, lr = row['_metadata'][0]
        
        kwds = dict(module_matrix=mm, lr_matrix=lr,intermodule_matrix=imm, modules=2, metadata=(mm,lr,imm))
        
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