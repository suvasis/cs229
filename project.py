'''
Created on Dec 6, 2018
CS229

@author: suvmukhe mukherjee and minakshi mukherjee
'''

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from operator import itemgetter
import pandas as pd
from collections import defaultdict

from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
import numpy as np
import pylab as pl

import matplotlib.pyplot as plt


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score


from sklearn.mixture import GMM

x = None
y = None

def clean_dataset(df):
    
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

def elastic_net():
    #https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html
    data = pd.read_excel('leave_one_out_results_linearregression.xlsx')#X_train.csv')#.head(150)
    data.columns = data.columns.to_series().apply(lambda x: x.strip())
    data= clean_dataset(data)
    data[data==np.inf]=np.nan
    data.fillna(data.mean(), inplace=True)
    #data = pd.read_csv('X_train.csv')#X_train.csv')#.head(150)
    print(data.columns)
    
    

#     columns =[ 'sertraline', 'venlafaxine', 'escitalopram', 'age', 'gender',
#        'education', 'SOFAS_baseline', 'SOFAS_6',  'SOFAS_response', 'Amygdala_Clust1', 'Insula_Clus1', 'Insula_Clust2',
#        'Nac_Clust1', 'Nac_Clust2']
    columns =[  'Amygdala_Clust1', 'Insula_Clus1','Insula_Clust2','Nac_Clust1', 'Nac_Clust2']
    
    #x = data.loc[:, data.columns != 'HDRS17_baseline']
    x = data.loc[:,columns]

    y = data.loc[:,'HDRS17_baseline']
    clean_dataset(x)    
    alpha = .3
    enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

    kf = KFold(len(x), n_folds=15)
    err = 0
    for train,test in kf:
        enet.fit(x.iloc[train],y.iloc[train])
        p = enet.predict(x.iloc[test])
        e = p-y.iloc[test]
        err += np.dot(e,e)
        y_pred_enet = enet.fit(x.iloc[train],y.iloc[train]).predict(x.iloc[test])
        r2_score_enet = r2_score(test, y_pred_enet)
        print(enet)
        print("r^2 on test data : %f" % r2_score_enet)
        
        #lasso
        alpha = 0.1
        lasso = Lasso(alpha=alpha)
        
        y_pred_lasso = lasso.fit(x.iloc[train],y.iloc[train]).predict(x.iloc[test])
        r2_score_lasso = r2_score(y.iloc[test], y_pred_lasso)
        print(lasso)
        print("r^2 on test data : %f" % r2_score_lasso)
    
        plt.plot(enet.coef_, color='lightgreen', linewidth=2,
                 label='Elastic net coefficients')
        plt.plot(lasso.coef_, color='gold', linewidth=2,
                 label='Lasso coefficients')
        plt.legend(loc='best')
        plt.title("Lasso R^2: %f, Elastic Net R^2: %f"
                  % (r2_score_lasso, r2_score_enet))
        plt.show()

def em_gmm():
    data = pd.read_excel('leave_one_out_results_linearregression.xlsx')#X_train.csv')#.head(150)
    data.columns = data.columns.to_series().apply(lambda x: x.strip())
    data= clean_dataset(data)
    data[data==np.inf]=np.nan
    data.fillna(data.mean(), inplace=True)
    #data = pd.read_csv('X_train.csv')#X_train.csv')#.head(150)
    print(data.columns)
    
    

    columns =[ 'sertraline', 'venlafaxine', 'escitalopram', 'age', 'gender',
       'education', 'SOFAS_baseline', 'SOFAS_6',  'SOFAS_response', 'Amygdala_Clust1', 'Insula_Clus1', 'Insula_Clust2',
       'Nac_Clust1', 'Nac_Clust2']
    #columns =[  'Amygdala_Clust1', 'Insula_Clus1']
    
    #x = data.loc[:, data.columns != 'HDRS17_baseline']
    x = data.loc[:,columns]

    #y = data.loc[:,'SOFAS_baseline']#'HDRS17_baseline']
    y = data.loc[:,'HDRS17_baseline']
    clean_dataset(x)
    
    gmm = GMM(n_components=4).fit(x)
    labels = gmm.predict(x)
    print(labels)
    plt.scatter(x.iloc[:, 10], y, c=labels, s=40, cmap='viridis');
    #plt.scatter(x.iloc[:, ], x.iloc[:, 1], c=labels, s=40, cmap='viridis');
    plt.xlabel('Amygdala_Clust1')
    #plt.xlabel('Nac_Clust1')
    plt.ylabel('HDRS17_baseline');#'SOFAS_baseline'
    #plt.ylabel('Nac_Clust2')#'Insula_Clus2')#'Amygdala_Clust1')
    plt.title('Gaussian Mixture Model')
    plt.show()


def plot_gmm():
    data = pd.read_excel('leave_one_out_results_linearregression.xlsx')#X_train.csv')#.head(150)
    data.columns = data.columns.to_series().apply(lambda x: x.strip())
    data= clean_dataset(data)
    data[data==np.inf]=np.nan
    data.fillna(data.mean(), inplace=True)
    #data = pd.read_csv('X_train.csv')#X_train.csv')#.head(150)
    print(data.columns)
    
    

    columns =[ 'sertraline', 'venlafaxine', 'escitalopram', 'age', 'gender',
       'education', 'SOFAS_baseline', 'SOFAS_6',  'SOFAS_response', 'Amygdala_Clust1', 'Insula_Clus1', 'Insula_Clust2',
       'Nac_Clust1', 'Nac_Clust2']
    #columns =[  'Amygdala_Clust1', 'Insula_Clus1']
    
    #x = data.loc[:, data.columns != 'HDRS17_baseline']
    x = data.loc[:,columns]

    y = data.loc[:,'HDRS17_baseline']
    clean_dataset(x)
    n_clusters=4 
    rseed=0 
    ax=None
    kmeans = KMeans(n_clusters=4, random_state=0)
    labels = kmeans.fit_predict(x)
    gmm = GMM(n_components=4).fit(x)
    labels = gmm.predict(x)
    
#    gmm = GMM(n_components=4, covariance_type='full', random_state=42)
#    plot_gmm(gmm, X_stretched)

    probs = gmm.predict_proba(x)
    size = 50 * probs.max(1) ** 2  # square emphasizes differences
    #plt.scatter(x.iloc[:, 10], x.iloc[:, 11], c=labels, cmap='viridis', s=size);
    plt.scatter(x.iloc[:, 12], y, c=labels, cmap='viridis', s=size);
    #plt.xlabel('Amygdala_Clust1')
    plt.xlabel('Nac_Clust1')
    plt.ylabel('HDRS17_baseline');#'SOFAS_baseline'
    #plt.ylabel('Nac_Clust2')#'Insula_Clus2')#'Amygdala_Clust1')
    plt.title('Gaussian Mixture Model')
    #plt.colorbar();  # show color scale
    plt.show()

def plot_kmeans():
    data = pd.read_excel('leave_one_out_results_linearregression.xlsx')#X_train.csv')#.head(150)
    data.columns = data.columns.to_series().apply(lambda x: x.strip())
    data= clean_dataset(data)
    data[data==np.inf]=np.nan
    data.fillna(data.mean(), inplace=True)
    #data = pd.read_csv('X_train.csv')#X_train.csv')#.head(150)
    print(data.columns)
    
    

#     columns =[ 'sertraline', 'venlafaxine', 'escitalopram', 'age', 'gender',
#        'education', 'SOFAS_baseline', 'SOFAS_6',  'SOFAS_response', 'Amygdala_Clust1', 'Insula_Clus1', 'Insula_Clust2',
#        'Nac_Clust1', 'Nac_Clust2']
    columns =[ 'sertraline', 'venlafaxine', 'escitalopram', 'Amygdala_Clust1', 'Insula_Clus1', 'Insula_Clust2',  'Nac_Clust1', 'Nac_Clust2']
    
    #x = data.loc[:, data.columns != 'HDRS17_baseline']
    x = data.loc[:,columns]

    y = data.loc[:,'HDRS17_baseline']
    #clean_dataset(x)
    n_clusters=4
    rseed=0 
    ax=None
    kmeans = KMeans(n_clusters, random_state=0)
    labels = kmeans.fit_predict(x)

    # plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    #print(x.iloc[:, 0], x.iloc[:, 1])
    #return
    ax.scatter(x.iloc[:, 0], x.iloc[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    #ax.scatter(x[:, 0], y, c=labels, s=40, cmap='viridis', zorder=2)

    #
    kmeans = KMeans(n_clusters=4, random_state=0).fit(x)
    # plot the representation of the KMeans model
    centers = kmeans.cluster_centers_
    radii = [cdist(x[labels == i], [center]).max()
             for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))
    plt.show()  
#not working
#http://facweb.cs.depaul.edu/mobasher/classes/csc478/Notes/IPython%20Notebook%20-%20Regression.html
def linearRegression():
    # use this dataset: - leave_one_out_results_linearregression.xlsx
    #data = pd.read_csv('C:/tools/workspace/cs224wproject/project/X_train.csv')#X_train.csv')#.head(150)
    data = pd.read_excel('leave_one_out_results_linearregression.xlsx')#X_train.csv')#.head(150)
    data.columns = data.columns.to_series().apply(lambda x: x.strip())
    data=clean_dataset(data)
    data[data==np.inf]=np.nan
    data.fillna(data.mean(), inplace=True)
    columns =['CONNID', 'sertraline', 'venlafaxine', 'escitalopram', 'age', 'gender',
       'education', 'SOFAS_baseline', 'SOFAS_6',  'SOFAS_response', 'Amygdala_Clust1', 'Insula_Clus1', 'Insula_Clust2',
       'Nac_Clust1', 'Nac_Clust2']

    #x = data.loc[:, data.columns != 'HDRS17_baseline']
    x = data.loc[:,columns]
    #print(x)
    y = data.loc[:,'HDRS17_baseline']

    #print(y.head())
    
    linreg = LinearRegression()
    LinearRegression(copy_X=True, fit_intercept=True, normalize=False)

    linreg.fit(x,y)
    print( linreg.predict(x))
    p = linreg.predict(x)
    err = abs(p-y)
    total_error = np.dot(err,err)
    rmse_train = np.sqrt(total_error/len(p))
    print (rmse_train)
    print ('Regression Coefficients: \n', linreg.coef_)
    pl.plot(p, y,'ro')
    pl.plot([0,50],[0,50], 'g-')
    pl.xlabel('predicted')
    pl.ylabel('real')
    pl.show()
    print("test")
    
def methodcomparison():
    data = pd.read_excel('leave_one_out_results_linearregression.xlsx')#X_train.csv')#.head(150)
    data.columns = data.columns.to_series().apply(lambda x: x.strip())
    data= clean_dataset(data)
    data[data==np.inf]=np.nan
    data.fillna(data.mean(), inplace=True)
    #data = pd.read_csv('X_train.csv')#X_train.csv')#.head(150)
    print(data.columns)
    
    

    columns =[ 'sertraline', 'venlafaxine', 'escitalopram', 'age', 'gender',
       'education', 'Amygdala_Clust1', 'Insula_Clus1', 'Insula_Clust2',
       'Nac_Clust1', 'Nac_Clust2']
    #columns =[  'Amygdala_Clust1', 'Insula_Clus1']
    
    #x = data.loc[:, data.columns != 'HDRS17_baseline']
    x = data.loc[:,columns]

    y = data.loc[:,'HDRS17_baseline']
    clean_dataset(x)
    a = 0.3
    for name,met in [
            ('linear regression', LinearRegression()),
            ('lasso', Lasso(fit_intercept=True, alpha=a)),
            ('ridge', Ridge(fit_intercept=True, alpha=a)),
            ('elastic-net', ElasticNet(fit_intercept=True, alpha=a))
            ]:
        met.fit(x,y)
        # p = np.array([met.predict(xi) for xi in x])
        p = met.predict(x)
        e = p-y
        total_error = np.dot(e,e)
        rmse_train = np.sqrt(total_error/len(p))
    
        kf = KFold(len(x), n_folds=10)
        err = 0
        for train,test in kf:
            met.fit(x.iloc[train],y.iloc[train])
            p = met.predict(x.iloc[test])
            e = p-y.iloc[test]
            err += np.dot(e,e)
    
        rmse_10cv = np.sqrt(err/len(x))
        print('Method: %s' %name)
        print('RMSE on training: %.4f' %rmse_train)
        print('RMSE on 10-fold CV: %.4f' %rmse_10cv)
        print( "\n")

def ridge():
    data = pd.read_excel('leave_one_out_results_linearregression.xlsx')#X_train.csv')#.head(150)
    data.columns = data.columns.to_series().apply(lambda x: x.strip())
    data= clean_dataset(data)
    data[data==np.inf]=np.nan
    data.fillna(data.mean(), inplace=True)
    #data = pd.read_csv('X_train.csv')#X_train.csv')#.head(150)
    print(data.columns)
    
    

#     columns =[ 'sertraline', 'venlafaxine', 'escitalopram', 'age', 'gender',
#        'education', 'SOFAS_baseline', 'SOFAS_6',  'SOFAS_response', 'Amygdala_Clust1', 'Insula_Clus1', 'Insula_Clust2',
#        'Nac_Clust1', 'Nac_Clust2']
    columns =[ 'sertraline', 'venlafaxine', 'escitalopram', 'Amygdala_Clust1', 'Insula_Clus1', 'Insula_Clust2',  'Nac_Clust1', 'Nac_Clust2']
    
    #x = data.loc[:, data.columns != 'HDRS17_baseline']
    x = data.loc[:,columns]

    y = data.loc[:,'HDRS17_baseline']
    clean_dataset(x)
    # Create linear regression object with a ridge coefficient 0.5
    ridge = Ridge(fit_intercept=True, alpha=0.5)

    # Train the model using the training set
    ridge.fit(x,y)
    
    # Compute RMSE on training data
    # p = np.array([ridge.predict(xi) for xi in x])
    p = ridge.predict(x)
    err = p-y
    total_error = np.dot(err,err)
    rmse_train = np.sqrt(total_error/len(p))
    
    # Compute RMSE using 10-fold x-validation
    kf = KFold(len(x), n_folds=10)
    xval_err = 0
    for train,test in kf:
        ridge.fit(x.iloc[train],y.iloc[train])
        p = ridge.predict(x.iloc[test])
        e = p-y[test]
        xval_err += np.dot(e,e)
    rmse_10cv = np.sqrt(xval_err/len(x))
    
    method_name = 'Ridge Regression'
    print('Method: %s' %method_name)
    print('RMSE on training: %.4f' %rmse_train)
    print('RMSE on 10-fold CV: %.4f' %rmse_10cv)
    print('Ridge Regression')
    print('alpha\t RMSE_train\t RMSE_10cv\n')
    alpha = np.linspace(.01,20,50)
    t_rmse = np.array([])
    cv_rmse = np.array([])
    
    for a in alpha:
        ridge = Ridge(fit_intercept=True, alpha=a)
        
        # computing the RMSE on training data
        ridge.fit(x,y)
        p = ridge.predict(x)
        err = p-y
        total_error = np.dot(err,err)
        rmse_train = np.sqrt(total_error/len(p))
    
        # computing RMSE using 10-fold cross validation
        kf = KFold(len(x), n_folds=10)
        xval_err = 0
        for train, test in kf:
            ridge.fit(x.iloc[train], y.iloc[train])
            p = ridge.predict(x.iloc[test])
            err = p - y[test]
            xval_err += np.dot(err,err)
        rmse_10cv = np.sqrt(xval_err/len(x))
        
        t_rmse = np.append(t_rmse, [rmse_train])
        cv_rmse = np.append(cv_rmse, [rmse_10cv])
        print('{:.3f}\t {:.4f}\t\t {:.4f}'.format(a,rmse_train,rmse_10cv))    
    pl.plot(alpha, t_rmse, label='RMSE-Train')
    pl.plot(alpha, cv_rmse, label='RMSE_XVal')
    pl.legend( ('RMSE-Train', 'RMSE_XVal') )
    pl.ylabel('RMSE')
    pl.xlabel('Alpha')
    pl.show()    
    
def kflod():
    data = pd.read_excel('leave_one_out_results_linearregression.xlsx')#X_train.csv')#.head(150)
    data.columns = data.columns.to_series().apply(lambda x: x.strip())
    data= clean_dataset(data)
    data[data==np.inf]=np.nan
    data.fillna(data.mean(), inplace=True)
    #data = pd.read_csv('X_train.csv')#X_train.csv')#.head(150)
    print(data.columns)

#     columns =[ 'sertraline', 'venlafaxine', 'escitalopram', 'age', 'gender',
#        'education', 'SOFAS_baseline', 'SOFAS_6',  'SOFAS_response', 'Amygdala_Clust1', 'Insula_Clus1', 'Insula_Clust2',
#        'Nac_Clust1', 'Nac_Clust2']
    columns =[ 'sertraline', 'venlafaxine', 'escitalopram', 'age', 'gender',
       'education',  'Amygdala_Clust1', 'Insula_Clus1', 'Insula_Clust2',
       'Nac_Clust1', 'Nac_Clust2']
    #columns =[  'Amygdala_Clust1', 'Insula_Clus1']

    #x = data.loc[:, data.columns != 'HDRS17_baseline']
    x = data.loc[:,columns]

    y = data.loc[:,'HDRS17_baseline']
    clean_dataset(x)
    # Now let's compute RMSE using 10-fold x-validation
    kf = KFold(len(x), n_folds=15)
    xval_err = 0
    linreg = LinearRegression()
    
    for train,test in kf:
        linreg.fit(x.iloc[train],y.iloc[train])
        # p = np.array([linreg.predict(xi) for xi in x[test]])
        p = linreg.predict(x.iloc[test])
        e = p-y[test]
        xval_err += np.dot(e,e)
    
    rmse_10cv = np.sqrt(xval_err/len(x))
    method_name = 'Simple Linear Regression'
    print('Method: %s' %method_name)
   # print('RMSE on training: %.4f' %rmse_train)
    print('RMSE on 10-fold CV: %.4f' %rmse_10cv)
    

def loaddata():
    # use this dataset: - leave_one_out_results_linearregression.xlsx
    #data = pd.read_csv('C:/tools/workspace/cs224wproject/project/X_train.csv')#X_train.csv')#.head(150)
    data = pd.read_excel('leave_one_out_results_linearregression.xlsx')#X_train.csv')#.head(150)
    data.columns = data.columns.to_series().apply(lambda x: x.strip())
    data=clean_dataset(data)
    data[data==np.inf]=np.nan
    data.fillna(data.mean(), inplace=True)
    columns =['CONNID', 'sertraline', 'venlafaxine', 'escitalopram', 'age', 'gender',
       'education', 'SOFAS_baseline', 'SOFAS_6',  'SOFAS_response', 'Amygdala_Clust1', 'Insula_Clus1', 'Insula_Clust2',
       'Nac_Clust1', 'Nac_Clust2']

    #x = data.loc[:, data.columns != 'HDRS17_baseline']
    x = data.loc[:,columns]
    print(x)
    y = data.loc[:,'HDRS17_baseline']

    print(y.head())
    

if __name__ == '__main__':
    loaddata()
    #linearRegression()
    #kflod()
    #ridge()
    #methodcomparison()
    #plot_kmeans()
    #plot_gmm()
    em_gmm()
    #elastic_net()
