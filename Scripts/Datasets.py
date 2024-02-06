import numpy as np
import pandas as pd
from scipy.io import mmread

def Scdata(names, path, only_labels=False):
    if only_labels == False: 
        X=pd.read_csv(path+names[0],header=0,index_col=0)
        y=pd.read_csv(path+names[1],header=0,index_col=0)
        return X,y
    else:
        y=pd.read_csv(path+names[1],header=0,index_col=0)
        return y
    
    
    

def ScMOdata(names, path, only_labels=False):
    if only_labels == False: 
        x1=pd.read_csv(path+names[0],header=0,index_col=0).T
        x2=pd.read_csv(path+names[1],header=0,index_col=0).T
        y=pd.read_csv(path+names[2],header=0,index_col=0)
        return x1, x2, y
    else:
        y=pd.read_csv(path+names[2],header=0,index_col=0)
        return y

    
    
    

def LoadData(dataname, only_labels=False):
    dir = '/home/chayan/NeuroMDAVIS/Data/'
    
    # -----------Single cell datasets (Single MOdality)-------------------
    if dataname == 'PBMC3k':
        names = ['PBMC3k.csv', 'PBMC3k_groundTruth.csv']
        path = dir + 'Single Omics/PreprocessedData/PBMC3k/'
        return Scdata(names, path, only_labels)
        
    elif dataname == 'Zeisel':
        names = ['ZeiselScanpy.csv', 'Zeisel_groundTruth.csv']
        path = dir + 'Single Omics/PreprocessedData/Zeisel/'
        return Scdata(names, path, only_labels)
    
    elif dataname == 'Usoskin':
        names = ['UsoskinScanpy.csv', 'Usoskin_groundTruth.csv']
        path = dir + 'Single Omics/PreprocessedData/Usoskin/'
        return Scdata(names, path, only_labels)
    
    elif dataname == 'Jurkat':
        names = ['JurkatScanpy.csv', 'Jurkat_groundTruth.csv']
        path = dir + 'Single Omics/PreprocessedData/Jurkat/'
        return Scdata(names, path, only_labels)
    
    elif dataname == 'Kolodziejczyk':
        names = ['KolodziejczykScanpy.csv', 'Kolodziejczyk_groundTruth.csv']
        path = dir + 'Single Omics/PreprocessedData/Kolodziejczyk/'
        return Scdata(names, path, only_labels)
    
    elif dataname == 'Quake':
        names = ['QuakeScanpy.csv', 'Quake_groundTruth.csv']
        path = dir + 'Single Omics/PreprocessedData/Quake/'
        return Scdata(names, path, only_labels)
    
    elif dataname == 'Blakeley':
        names = ['BlakeleyScanpy.csv', 'Blakeley_groundTruth.csv']
        path = dir + 'Single Omics/PreprocessedData/Blakeley/'
        return Scdata(names, path, only_labels)
    
    elif dataname == 'Wong':
        names = ['Wong.csv', 'Wong_groundTruth.csv']
        path = dir + 'Single Omics/PreprocessedData/Wong/'
        return Scdata(names, path, only_labels)
    
    
    elif dataname == 'E-GEOD-139324':
        path = dir + 'Single Omics/PreprocessedData/E-GEOD-139324/'
        if only_labels == False:
            x = mmread(path+'E-GEOD-139324.aggregated_filtered_normalised_counts.mtx').toarray().T
            y = pd.read_csv(path+'ExpDesign-E-GEOD-139324.tsv', sep='\t')
            return x, y
        else :
            y = pd.read_csv(path+'ExpDesign-E-GEOD-139324.tsv', sep='\t')
            return y
        
    elif dataname == 'E-GEOD-123046':
        path = dir + 'Single Omics/PreprocessedData/E-GEOD-123046/'
        if only_labels == False:
            x = mmread(path+'E-GEOD-123046.aggregated_filtered_normalised_counts.mtx').toarray().T
            y = pd.read_csv(path+'ExpDesign-E-GEOD-123046.tsv', sep='\t')
            return x, y
        else :
            y = pd.read_csv(path+'ExpDesign-E-GEOD-123046.tsv', sep='\t')
            return y
        
    elif dataname == 'E-MTAB-6701':
        path = dir + 'Single Omics/PreprocessedData/E-MTAB-6701/'
        if only_labels == False:
            x = mmread(path+'E-MTAB-6701.aggregated_filtered_normalised_counts.mtx').toarray().T
            y = pd.read_csv(path+'ExpDesign-E-MTAB-6701.tsv', sep='\t')
            return x, y
        else :
            y = pd.read_csv(path+'ExpDesign-E-MTAB-6701.tsv', sep='\t')
            return y
        
    #-----------------Time series data---------------------------------------
    elif dataname == 'TSD1':
        path = dir + 'Single Omics/PreprocessedData/TimeSeriesData1_(E-MTAB-7901)/'
        if only_labels == False:
            x = mmread(path+'E-MTAB-7901.aggregated_filtered_normalised_counts.mtx').toarray().T
            y = pd.read_csv(path+'ExpDesign-E-MTAB-7901.tsv', sep='\t')
            return x, y
        else :
            y = pd.read_csv(path+'ExpDesign-E-MTAB-7901.tsv', sep='\t')
            return y
    
    elif dataname == 'TSD2':
    	return 0    
    
    
    
    # -----------Single cell datasets (Multiple Modalities)-------------------
    
    elif dataname == 'pbmc5k':
        names = ['rna_scaled_pbmc5K.csv', 'protein_scaled_pbmc5K.csv', 'labels_pbmc5K_groundTruth.csv']
        path = dir + 'Multi Omics/pbmc5k/'
        return ScMOdata(names, path, only_labels)
        
    elif dataname == 'pbmc10k_MALT':
        names = ['pbmc10k_MALT_rna_scaled.csv', 'pbmc10k_MALT_adt_scaled.csv', 'pbmc10k_MALT_groundTruth.csv']
        path = dir + 'Multi Omics/pbmc10k_MALT/'
        return ScMOdata(names, path, only_labels)
    
    elif dataname == 'cbmc8k':
        names = ['cbmc8k_rna_scaled.csv', 'cbmc8k_adt_scaled.csv', 'cbmc8k_groundTruth.csv']
        path = dir + 'Multi Omics/cbmc8k/'
        return ScMOdata(names, path, only_labels)
        
    elif dataname == 'bmcite30k':
        names = ['bmcite30k_rna_scaled.csv', 'bmcite30k_adt_scaled.csv', 'bmcite30k_groundTruth.csv']
        path = dir + 'Multi Omics/bmcite30k/'
        return ScMOdata(names, path, only_labels)
    
    elif dataname == 'gayoso30k':
        names = ['gayoso30k_rna_scaled.csv', 'gayoso30k_adt_scaled.csv', 'gayoso30k_groundTruth.csv']
        path = dir + 'Multi Omics/gayoso30k/'
        return ScMOdata(names, path, only_labels)
        
    
    elif dataname == 'kotliarov50k':
        names = ['kotliarov50k_rna_scaled.csv', 'kotliarov50k_adt_scaled.csv', 'kotliarov50k_groundTruth.csv']
        path = dir + 'Multi Omics/kotliarov50k/'
        return ScMOdata(names, path, only_labels)
    
    elif dataname == 'pbmc10k_atacseq':
        names = ['pbmc10k_rna_hvg_matched_cells.csv', 'pbmc10k_atac_hvg_matched_cells.csv', 'pbmc10k_groundTruth_rna.csv']
        path = dir + 'Multi Omics/pbmc10k_atacseq/'
        return ScMOdata(names, path, only_labels)
    
    else:
        print('Invalid dataname')
        return pd.DataFrame(), pd.DataFrame()
