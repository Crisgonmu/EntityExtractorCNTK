# Training a Neural Entity Detector using Pubmed Word Embeddings
import os
from DataReader import DataReader
from EntityExtractor import EntityExtractor


################################################################################### 
#  Train the network on the prepared data and obtain the predictions on the test set
###################################################################################
def main():
    print("Running on BIO-NLP data\n\n")
    
    home_dir = "../dl4nlp"
   
    # The hyper-parameters of the word embedding trained model 
    window_size = 5
    embed_vector_size = 50
    min_count = 1000

    # Define the data files 
    data_folder = os.path.join("..\\", "sample_data","drugs_and_diseases")
    train_file_path = os.path.join(data_folder, "Drug_and_Disease_train.txt")
    test_file_path = os.path.join(data_folder, "Drug_and_Disease_test.txt")    
    data_file_path = os.path.join(data_folder, "unlabeled_test_sample.txt")    
    resources_pickle_file = os.path.join(home_dir, "models", "resources.pkl")
    embedding_pickle_file = os.path.join(home_dir, "models", "w2vmodel_pubmed_vs_{}_ws_{}_mc_{}.pkl" \
            .format(embed_vector_size, window_size, min_count))
    print("embedding_pickle_file= {}".format(embedding_pickle_file))

    # The hyperparameters of the LSTM trained model         
    #network_type= 'unidirectional'
    network_type= 'bidirectional'
    num_layers = 2
    num_hidden_units = 150
    num_epochs = 10
    batch_size = 50
    dropout = 0.2
    reg_alpha = 0.0

    model_file_path = os.path.join(home_dir,'models','lstm_{}_model_units_{}_lyrs_{}_epchs_{}_vs_{}_ws_{}_mc_{}.h5'.\
                  format(network_type, num_hidden_units, num_layers,  num_epochs, embed_vector_size, window_size, min_count))    
    
    print("Training the model... num_epochs = {}, num_layers = {}, num_hidden_units = {}".\
            format(num_epochs, num_layers,num_hidden_units))

    reader = DataReader() 
    
    entityExtractor = EntityExtractor(reader, embedding_pickle_file)
    
    entityExtractor.train (train_file_path, \
        output_resources_pickle_file = resources_pickle_file, \
        network_type = network_type, \
        num_epochs = num_epochs, \
        batch_size = batch_size, \
        dropout = dropout, \
        reg_alpha = reg_alpha, \
        num_hidden_units = num_hidden_units, \
        num_layers = num_layers)                

    #Save the model
    entityExtractor.save(model_file_path)

    print("Done.")     
   
if __name__ == "__main__":
    main()

