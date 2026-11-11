from utils import *
import pandas as pd

    
def main():
    filename = 'dataset_aquila/default_of_credit_card_clients.csv'
    input_df = pd.read_csv(filename, index_col='ID')
    
    categ_cols = ['SEX', 'MARRIAGE']  
    num_samples_qpu = 2571
    
    cleaned_df = clean_dataset(input_df)    
    X_emulation, X_qpu, y_emulation, y_qpu = split_dataset(cleaned_df, num_samples_qpu)      
    X_emu_scaled, X_qpu_scaled = scale_features(X_emulation, X_qpu, categ_cols)  
    agg_dict = pca_correlation_graph(X_emu_scaled, X_emu_scaled.columns)
    X_emu_agg, X_qpu_agg = aggregate_features(agg_dict, X_emu_scaled, X_qpu_scaled)
  
    X_emu_agg.to_csv('dataset_aquila/emulation_samples.csv')
    X_qpu_agg.to_csv('dataset_aquila/qpu_samples.csv')    
    y_emulation.to_csv('dataset_aquila/labels_emu.csv')
    y_qpu.to_csv('dataset_aquila/labels_qpu.csv')
    

if __name__ == "__main__":
    main()