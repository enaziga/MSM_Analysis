from msmbuilder.featurizer import DihedralFeaturizer
from msmbuilder.decomposition import tICA
from msmbuilder.cluster import MiniBatchKMeans, KMeans
from msmbuilder.msm import MarkovStateModel, implied_timescales
from msmbuilder.io import gather_metadata, save_meta, NumberedRunsParser
from msmbuilder.io import load_meta, preload_tops, save_trajs, save_generic, load_trajs, preload_top, backup, load_generic
from msmbuilder.io.sampling import sample_states, sample_msm, sample_dimension
from msmbuilder.lumping import PCCAPlus
import matplotlib
matplotlib.use('Agg')
import mdtraj as md
import msmexplorer as msme
from multiprocessing import Pool
import seaborn as sns
from matplotlib import pyplot as plt
import glob
import numpy as np
import os
import time
import locale
from optparse import OptionParser
import math

rs = np.random.RandomState(42)
sns.set_style('ticks')
colors = sns.color_palette()

parser = OptionParser(usage="usage: %prog [options] ")
parser.set_defaults(tica_n=4, tica_lag=240, nclusters=32, msm_lag=300, freeEnergy=True)

parser.add_option("-e", "--free-energy" , dest="freeEnergy", help="Generate a free-energy histogram instead of a frequency")
parser.add_option("-w", "--wt-directory", dest="wt_dir", help="Path to WT directory with trajectories to analyse")
parser.add_option("-t", "--wt-topology", dest="wt_top", help="Name of WT topology (PDB) which shouyld be in WT directory.")
parser.add_option("-m", "--mutant-directory", dest="mut_dir", help="Path to Mutant directory with trajectories to analyse")
parser.add_option("-p", "--mutant-topology", dest="mut_top", help="Name of mutant topology(PDB) which shouyld be in WT directory.")

(options, args) = parser.parse_args()

if (options.wt_dir == ""):
    print >> sys.stderr, "Directory path for at least the WT is required"
    sys.exit(1)

systems = []
topologies = []

if (options.mut_dir != ""):
   systems.append(options.wt_dir)
   systems.append(options.mut_dir)
   topologies.append(options.wt_top)
   topologies.append(options.mut_top)   
else:
   systems.append(options.wt_dir)
   topologies.append(options.wt_top)

def select_data_samples(Nsamples, run_id, is_rand):

    metadatas = []

    for i,system in enumerate(systems):

        sys_fnames = system + "*.xtc"
        sys_files = glob.glob(sys_fnames)

        parser_name = 'meta%01i' % i
        pickle_name = 'meta%01i.pandas.pickl' % i
        #topology    = system + "/" + topologies[i]
        topology    = topologies[i]

        parser = NumberedRunsParser(traj_fmt="heavyatoms_{run}.xtc", top_fn=topology, step_ps=40,)
        #path_to_files = sys_dir + "/*.xtc"
        meta_handle = gather_metadata(sys_fnames, parser)
        save_meta(meta_handle, meta_fn=pickle_name)
        print(parser_name)
        
        metadatas.append(meta_handle)

    return metadatas

# Helper Functions for Featurization

def feat0(irow):
    #tops = preload_tops(topologies[0])
    i, row = irow
    #traj = md.load(row['traj_fn'], top=tops[row['top_fn']])
    traj = md.load(row['traj_fn'], top=topologies[0])
    feat_traj = dihed_feat.partial_transform(traj)
    return i, feat_traj

def feat1(irow):
    i, row = irow
    traj = md.load(row['traj_fn'], top=topologies[1])
    feat_traj = dihed_feat.partial_transform(traj)
    return i, feat_traj

def featurize_dataset(metas):

    dihed_feats = []

    dihed_feat = DihedralFeaturizer(types=['phi', 'psi'],sincos=True)

    if len(metas) == 1:
        meta = metas[0]
        ## Do it in parallel
        with Pool() as pool:
            dihed_feat = dict(pool.imap_unordered(feat0, meta.iterrows() ))
        print("I got here")
        #save_generic(dihed_feat, pickle_name)
        dihed_feats.append(dihed_feat)

    if len(metas) == 2:
        
        meta1 = metas[0]
        meta2 = metas[1]
        ## Do it in parallel
        with Pool() as pool:
            dihed_feat1 = dict(pool.imap_unordered(feat0, meta1.iterrows() ))
            dihed_feat2 = dict(pool.imap_unordered(feat1, meta2.iterrows() ))
        print("I got here")

        #save_generic(dihed_feat, pickle_name)
        dihed_feats.append(dihed_feat1)
        dihed_feats.append(dihed_feat2)

    print("Trajectories saved")

    if len(metas) > 1:
        
        combined_feat_trajs = []
        keys = []
 
        for i,each_dihed_feat in enumerate(dihed_feats):

            model_keys_name = []

            for d_trajs1 in each_dihed_feat:
                combined_feat_trajs.append(each_dihed_feat[d_trajs1])
                model_keys_name.append(d_trajs1)

            keys.append(model_keys_name)

    return [combined_feat_trajs, keys]


def generate_msm(both_feat_trajs, keys, tica_n, tica_lag, nclusters, msm_lag):

    # tICA decomposition
    tica_decomp = tICA(n_components=tica_n, lag_time=tica_lag, kinetic_mapping=True,commute_mapping=False,shrinkage=None)
    combined_tica = tica_decomp.fit_transform(both_feat_trajs)

    if len(keys) == 2:
        
        tica_model1 = combined_tica[0:len(keys[0])]
        tica_model2 = combined_tica[len(keys[0]):len(keys[0])+len(keys[1])]
        tica_models = [tica_model1,tica_model2]
    else:

        tica_models = [combined_tica]
 
    # Joint kmeans
    kmeans = MiniBatchKMeans(batch_size=100, compute_labels=True, init='k-means++', max_iter=100, max_no_improvement=10, n_clusters=nclusters, \
         n_init=3, random_state=rs, reassignment_ratio=0.01, tol=0.0, verbose=0)
    assignments = kmeans.fit_transform(combined_tica)

    # build individual MSMs

    msm_models = []
    for tica_model in tica_models:
        msm = MarkovStateModel(lag_time=msm_lag, n_timescales=5, verbose=False, sliding_window=True, reversible_type='mle', prior_counts=0, ergodic_cutoff='on')
        msm_models.append(msm)

    if len(tica_models) == 2:

        model_clusters1 = assignments[0:len(tica_models[0])]
        model_clusters2 = assignments[len(tica_models[0]):len(tica_models[0])+len(tica_models[1])]
        model_clusters = [model_clusters1,model_clusters2]
    else:

        model_clusters = assignments

    fitted_msm_models = []
    
    for msm_model,model_cluster in zip(msm_models,model_clusters):
        fitted_msm_model = msm_model.fit_transform(model_cluster)
        fitted_msm_models.append(fitted_msm_model)

    return [tica_models,kmeans,msm_models,fitted_msm_models,model_clusters,assignments]


def calculate_free_energies(tica,msm,assignments,n_data):

    # Free energy data
    data = np.concatenate(tica, axis=0)
    pi_0 = msm.populations_[np.concatenate(assignments, axis=0)]

    FE = msme.plot_free_energy(data[:len(pi_0),:2], obs=(0, 1), n_samples=n_data, pi=pi_0,
                      random_state=rs,
                      cmap='jet',
                      alpha=0.9,
                      labelsize=18,
                      shade=True,
                      clabel=True,
                      xlabel="tIC 1",
                      ylabel="tIC 2",
                      clabel_kwargs={'fmt': '%.1f'},
                      cbar=True,
                      cbar_kwargs={'format': '%.1f', 'label': 'Free energy (kcal/mol)', 'drawedges': 'False'}
                      )
    
    return [FE,msm.populations_]

def generate_msm_trajs(run_id, keys, tica, msm, kmeans, metadata, topology):

    wt_dict =  dict(zip(keys, tica)) 
    inds = sample_msm(wt_dict, kmeans.cluster_centers_, msm, n_steps=20000, stride=50)
    save_generic(inds, str(run_id)+".pickl")
    metax = load_meta(meta_fn=metadata)
    traj = md.join( md.load_frame(metax.loc[traj_i]['traj_fn'], index=frame_i, top=topology) for traj_i, frame_i in inds )
    traj_fn = run_id + "_traj.xtc"
    backup(traj_fn)
    traj.save(traj_fn)

# Load data, do tICA, Clustering and MSM generation

metas = select_data_samples(100, 1, is_rand=False)
dihed_feat = DihedralFeaturizer(types=['phi', 'psi'],sincos=True)
combined_feat_trajs, keys = featurize_dataset(metas)
tica_models,kmeans,msm_models,fitted_msm_models,model_clusters,assignments = generate_msm(combined_feat_trajs, keys, tica_n=4, tica_lag=240, nclusters=32, msm_lag=300)

# Calculate free energies and generate MSM trajectories

#if freeEnergy == True:
n_data = 250000

for i,msm_model in enumerate(msm_models):
    FE_filename ='Free_Energy_Model_%01i.pdf' % i
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 8))
    freeEnergy,population = calculate_free_energies(tica_models[i],msm_models[i],fitted_msm_models[i],n_data)
    fig.tight_layout()
    fig.savefig(FE_filename)

for i,metadata in enumerate(metas):
    tops_meta = preload_top(metadata)
    generate_msm_trajs(i, keys[i], tica_models[i], msm_models[i], kmeans, metadata, tops_meta)

