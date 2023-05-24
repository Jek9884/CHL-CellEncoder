import scanpy as sc
from sklearn.metrics import silhouette_score, mean_squared_error
import numpy as np
from sklearn import metrics
import pandas as pd
from os import path
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.preprocessing import normalize

import plotly.graph_objs as go
import plotly.io as pio
from IPython.display import HTML

def prepare_data(data, norm=True, log=True, scale=True):
    """
        Apply pre-processing pipeline to input data
    """
    
    
    if norm:
        # Normalize counts per cell
        sc.pp.normalize_total(data, target_sum=1e4)
    if log:
        # Log-transform the data
        sc.pp.log1p(data)
    if scale:
        # Scale data
        sc.pp.scale(data)

    data.raw = data

    # Find variable genes based on log dispersion
    sc.pp.highly_variable_genes(data, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    return data


def find_clusters(anndata, n_pcs=50, metric='euclidean', resolution=1.0, init_pos = 'spectral', use_rep=None, n_neigh=15, seed=42, save=False, return_anndata=False):
    """
        Performs clustering on input data and computes scores
    """
    
    
    print(f"umap_{n_pcs}_{metric}_{resolution}_{init_pos}.png")
    
    # Prepare data
    data = anndata.copy()
    
    # Compute nearest neighbors using the first n_pcs principal components
    data = sc.pp.neighbors(data, use_rep=use_rep, n_pcs=n_pcs, n_neighbors=n_neigh, metric=metric, random_state=seed, copy=True)
        
    # Run the Louvain algorithm for clustering
    data = sc.tl.louvain(data, random_state=seed, resolution=resolution, copy=True)
    
    # Compute SSE score of clustering
    sse_score = compute_sse(data, use_rep)
    
    # Compute Silhouette score of clustering
    sl_score = compute_silhouette(data, use_rep)

    # Save the plot with the custom filename
    if save:
        # Visualize clusters
        data = sc.tl.umap(data, random_state=seed, init_pos=init_pos, copy=True)

        sc.pl.umap(data, color="louvain")

        # Set the custom filename
        filename = f"umap_{n_pcs}_{metric}_{resolution}_{init_pos}.png"
        plt.savefig(filename, dpi=300)
        
    print(f"Silhouette {sl_score}")
    print(f"SSE {sse_score}")
    
    if return_anndata:
        return data
    else:
        return sl_score, sse_score
    
def compute_sse(adata, use_rep, cluster_col = 'louvain'):
    """
        Compute sse score
    """
    
    if not use_rep:
        use_rep="X_scVI"
        
    unique_labels = np.unique(adata.obs[cluster_col])
    centers = np.zeros((len(unique_labels), np.shape(adata.obsm[use_rep])[1]))
    
    # Compute centroids
    for i, label in enumerate(unique_labels):
        cluster_mask = (adata.obs[cluster_col] == label)
        centers[i, :] = adata[cluster_mask].obsm[use_rep].mean(axis=0)
    
    # Assign labels
    label_ints = np.zeros(adata.shape[0], dtype=int)
    for i, label in enumerate(unique_labels):
        label_ints[adata.obs[cluster_col] == label] = i

    # Compute sse score
    sse = ((adata.obsm[use_rep] - centers[label_ints, :])**2)
    sse = sse.sum()
    
    return sse

def compute_silhouette(adata, use_rep):
    """
        Compute silhouette score
    """
    
    if not use_rep:
        use_rep="X_scVI"
    
    cluster_assignments = adata.obs['louvain'].values
    return silhouette_score(adata.obsm[use_rep], cluster_assignments)

def compute_purity(adata_og, adata_pred):
    """
        Compute purity score
    """
    
    # compute contingency matrix
    contingency_matrix = metrics.cluster.contingency_matrix(adata_og, adata_pred)

    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def gen_purity_hist(adata_og, adata_pred, save_path=None):
    """
        Generate the histogram of the purity score where each bar corresponds to a predicted cluster
    """
    
    # compute contingency matrix
    contingency_matrix = metrics.cluster.contingency_matrix(adata_og, adata_pred)
    
    # get the max value for each column
    max_cols = np.amax(contingency_matrix, axis=0)
    sum_row = 0.0
    res = []
    
    # compute score for each cluster
    for i in range(np.shape(contingency_matrix)[1]):
        sum_row = float(np.sum(contingency_matrix[:,i]))
        res.append(max_cols[i] / sum_row)
        
    plt.bar(range(len(res)), res)
    plt.ylim([0, 1])
    
    # save plot
    if save_path:
        plt.savefig(save_path+'bar_plot_purity.png')
    else:
        plt.show()
        
def gen_heatmap(matrix, save_path=None):
    """
        Generate the heatmap of an input matrix
    """
    matrix_normed = normalize(matrix, axis=1, norm='l2')
    
    ax = sn.heatmap(matrix_normed, linewidth=0.5)
    
    if save_path:
        plt.savefig(save_path+'heatmap.png')
    else:
        plt.show()

def plot_sankey(ground_truth, predicted, field_og = 'cluster.ids', field_pred = 'louvain', filename = "sankey_plot", from_cls = 0, to_cls = 3):
    """
        Generate a Sankey plot w.r.t. the ground truth labels and the predicted values
    """
    pio.renderers.default = "plotly_mimetype+notebook"
    
    # compute contingency matrix
    contingency_matrix = metrics.cluster.contingency_matrix(ground_truth[field_og], predicted.obs[field_pred])
    contingency_matrix = contingency_matrix[from_cls:to_cls, :] 
    
    # create the data structures for the sankey plot
    n_gt = len(ground_truth[field_og].unique())
    n_pred = len(predicted.obs[field_pred].unique())
    targets = np.array([[i for i in range(0, n_pred)] for j in range(0, n_gt)]).flatten()
    sources = np.array([[ j for i in range(0, n_pred)] for j in range(0, n_gt)]).flatten()
    
    # List of colors to apply
    colors_list = ['rgba(158,1,66, 0.8)', 'rgba(254,224,139, 0.8)', 'rgba(255,0,255, 0.8)', 'rgba(50,136,189, 0.8)', 'rgba(244,109,67, 0.8)', 'rgba(102,194,165, 0.8)', 'rgba(94,79,162, 0.8)', 'rgba(254,224,139, 0.8)', 'rgba(255,0,255, 0.8)', 'rgba(213,62,79, 0.8)', 'rgba(171,221,164, 0.8)']
    
    diff = np.abs(to_cls-from_cls)
    
    cl = []
    label_gt = []
    for i in range(diff):
        cl_idx = i % len(colors_list)
        # Assign colors
        cl = cl + ([colors_list[cl_idx]]*n_pred)
        # Assing labels
        label_gt = label_gt + [f"C{i}_gt"]
    
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = label_gt + [f"C{i}_pred" for i in range(0, n_pred)],
          color = "blue"
        ),
        link = dict(
          source = sources.flatten()[:n_pred*diff],
          target = np.array([targets+to_cls]).flatten()[:n_pred*diff],
          value = contingency_matrix.flatten()[:n_pred*diff],
          color = cl
      ))])

    fig.write_html(f"{filename}.html", default_width=1200, default_height=1200)

    # Show the figure
    HTML(filename=f"{filename}.html")
    

def grid_clustering(anndata, n_pcs=50, **kwargs):
    """
        Perform grid search on clustering parameters
    """
    
    metrics = kwargs['metrics']
    resolutions = kwargs['resolutions']
    init_pos = kwargs['init_pos']
    use_rep = kwargs['use_rep']
    
    if 'filename' in kwargs:
        filename = kwargs['filename']
    else:
        filename = "raw_clustering.csv"
    
    dim = len(metrics) + len(resolutions)
 
    for metric in kwargs['metrics']:
        for resolution in kwargs['resolutions']:
            for pos in kwargs['init_pos']: 
                for n_neigh in kwargs['n_neigh']:
                            #Perform clustering
                            res_dict = {}
                            res_dict['metric'] = metric
                            res_dict['resolution'] = resolution
                            res_dict['init_pos'] = pos
                            res_dict['n_neighbors'] = n_neigh
                            res_dict['silhouette'], res_dict['sse'] = find_clusters(anndata, n_pcs = n_pcs, metric = metric, 
                                                                                    resolution = resolution, init_pos = pos, use_rep=use_rep, n_neigh=n_neigh)
                            
                            # Save results of config on dataframe
                            for key, value in res_dict.items():
                                res_dict[key] = [res_dict[key]]
                            
                            df = pd.DataFrame(res_dict)    
    
                            # Store results (if file already exists, append the results otherwise create the .csv file)
                            df.to_csv(filename, mode='a', sep='#', index=False, 
                                    header=False if path.exists(filename) else True)