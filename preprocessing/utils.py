import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage


def clean_dataset(df):
    '''
    Clean the dataset 'df' (pandas.DataFrame) removing undocumented samples and correct the values of the attributes 'PAY_*'
    '''
    # the documented values for MARRIAGE are [1,2,3], there are 54 samples labelled as 0
    df = df[df.MARRIAGE.isin([1,2,3])]
    # the documented values for EDUCATION are [1,2,3,4], there are 345 samples labelled as 0, 5, 6
    df = df[df.EDUCATION.isin([1,2,3,4])]
    # correct values of the PAY_* columns
    df.loc[:, df.columns.str.startswith('PAY_')] += 1    
    return df


def split_dataset(df, num_samples_qpu):
    '''
    Split the dataset 'df'(pandas.DataFrame) into samples that are processed by quantum evolution in emulation 'X_emu' (pandas.DataFrame)
    and those processed by Aquila QPU 'X_qpu' (pandas.DataFrame). The division is stratified according to the labels, thus it maintains
    the classes distributions'. 'num_samples_qpu' is an integer indicating how many samples are going to be in the 'X_qpu'dataset.
    '''
    # stratified split
    df_emu, df_qpu = train_test_split(df, test_size=num_samples_qpu, random_state=7, stratify=df.default_payment_next_month)
    # all columns are features, apart from the last one which contains the class labels
    X_emu = df_emu.loc[:, df_emu.columns[:-1]]
    y_emu = df_emu.loc[:, df_emu.columns[-1]]
    X_qpu = df_qpu.loc[:, df_qpu.columns[:-1]]
    y_qpu = df_qpu.loc[:, df_qpu.columns[-1]]
    return X_emu, X_qpu, y_emu, y_qpu


def scale_features(X_emu, X_qpu, categorical_cols): 
    '''
    Scale the features in the datasets 'X_emu', 'X_qpu' (pandas.DataFrame) with values in [-1,+1].
    Categorical features 'categorical_cols' (List of String) are one-hot-encoded.
    '''     
    # scale all non-categorical features
    scaler = MaxAbsScaler()
    numerical_cols = X_emu.columns[~X_emu.columns.isin(categorical_cols)]
    scaler.fit(X_emu[numerical_cols])
    X_emu[numerical_cols] = scaler.transform(X_emu[numerical_cols])
    X_qpu[numerical_cols] = scaler.transform(X_qpu[numerical_cols])
    # One-hot encode categorical columns and remove redundant columns
    X_emu = pd.get_dummies(X_emu, columns=categorical_cols)  
    X_emu = X_emu.drop(columns=['SEX_2', 'MARRIAGE_3']) 
    X_qpu = pd.get_dummies(X_qpu, columns=categorical_cols)
    X_qpu = X_qpu.drop(columns=['SEX_2', 'MARRIAGE_3']) 
    return X_emu, X_qpu


def pca_correlation_graph(X, variables_names):    
    '''
    Apply correlation circle method to aggregate features of the dataset 'X' (pandas.DataFrame). 'variables_names' are the columns name of 'X'.
    '''
    # apply PCA on X
    X = np.array(X).astype(float)
    num_col=X.shape[1]
    pca = PCA(n_components=num_col)
    pca.fit(X)
    X_pca = pca.transform(X)
    explained_var = pca.explained_variance_ratio_   
    # select the number of pricipal components according to the increment in the variance
    num_col=((explained_var[0:-1]-explained_var[1:])>1e-3).sum()
    print('Number of PC: {}'.format(num_col))
    # fit PCA with the optimal number of components
    pca = PCA(n_components=num_col, whiten=True).fit(X)
    X_pca = pca.transform(X)
    clustering_var = pca.components_.T   
     
    # method that computed the correlation between matrix A and B
    def corr2_coeff(A, B):
        A, B = np.array(A), np.array(B)
        # Rowwise mean of input arrays & subtract from input arrays themeselves
        A_mA = A - A.mean(1)[:, None]
        B_mB = B - B.mean(1)[:, None]
        # Sum of squares across rows
        ssA = (A_mA**2).sum(1)
        ssB = (B_mB**2).sum(1)
        # Finally get corr coeff
        return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None])) 
    
    def create_correlation_table(A, B, names_cols_A, names_cols_B):
        corrs = corr2_coeff(A.T, B.T).T
        df_corrs = pd.DataFrame(corrs,  columns=names_cols_A, index=names_cols_B)
        return df_corrs
    
    # compute correlation scores between PCA-transformed dataset and 'plain' dataset
    correlation_matrix = create_correlation_table(X_pca, X, ['Dim ' + str(i + 1) for i in range(num_col)], variables_names)
    # cluster the features of X according to the PCA scores
    clusters_all = AgglomerativeClustering(n_clusters=clustering_var.shape[0], compute_distances=True)
    clusters_all.fit(clustering_var)
    threshold = (clusters_all.distances_.min()+clusters_all.distances_.max())/2
    clusters = AgglomerativeClustering(distance_threshold=threshold, n_clusters=None)
    clusters.fit(clustering_var)
    correlation_matrix['cluster']=clusters.labels_
    # clustering results: show which features are aggregated in the same cluster
    num_cluster = clusters.n_clusters_
    agg_dict = {}
    for i in range(num_cluster):
        samples = np.where(clusters.labels_==i)[0]
        features = correlation_matrix.index[samples]
        agg_dict[i]=list(features)
        print('Cluster {} -> features {}'.format(i, features))
    # plot the correlation circle
    cmap = plt.cm.tab20
    cNorm  = colors.Normalize(vmin=0, vmax=num_cluster-1)
    scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cmap)
    plt.Circle((0, 0), radius=1, color='k', fill=False)
    circle1 = plt.Circle((0, 0), radius=1, color='k', fill=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle1)
    correlation_matrix=correlation_matrix.sort_values('cluster')
    cl_list=[]
    for _, row in correlation_matrix.iterrows():
        label=None
        cl = int(row['cluster'])
        if cl not in cl_list:
            label='cluster '+str(cl)
            cl_list.append(cl)
        x = row['Dim 1']
        y = row['Dim 2']
        plt.arrow(0.0, 0.0, x, y, color=scalarMap.to_rgba(row['cluster']), length_includes_head=True, head_width=.03, label=label)
        plt.plot([0.0, x], [0.0, y], '-', color=scalarMap.to_rgba(row['cluster']))
    # Plotting vertical lines
    plt.plot([-1.1, 1.1], [0, 0], 'k--')
    plt.plot([0, 0], [-1.1, 1.1], 'k--')
    # Setting limits and title
    plt.xlim((-1.1, 1.1))
    plt.ylim((-1.1, 1.1))
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    os.makedirs('results/preprocessing', exist_ok=True)
    plt.savefig('results/preprocessing/corr_circle.png')
    plt.close()
    return agg_dict


def aggregate_features(agg_dict, X_emu, X_qpu):
    '''
    Aggregate the features in the dataset 'X_emu', 'X_qpu', according to the clustering defined in 'agg_dict' dictionary.
    '''
    for key, value in agg_dict.items():
        # aggregate features by mean value
        X_emu[key]=X_emu[value].mean(axis=1)
        X_qpu[key]=X_qpu[value].mean(axis=1)
        X_emu = X_emu.drop(labels=value, axis=1)
        X_qpu = X_qpu.drop(labels=value, axis=1)
    return X_emu, X_qpu