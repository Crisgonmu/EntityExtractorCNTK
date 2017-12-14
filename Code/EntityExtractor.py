import numpy as np
import pandas as pd
import sys
from sklearn.metrics import confusion_matrix, classification_report
import cntk as C
from cntk.logging.graph import plot
from cntk.layers import For

# For reproducibility
np.random.seed(42)

class EntityExtractor:

    def __init__ (self, reader, embedding_pickle_file=None):
        
        self.reader = reader
        self.model = None       
        
        if not (embedding_pickle_file is None):
            self.wordvecs = self.reader.load_embedding_lookup_table(embedding_pickle_file)
 

    def load (self, filepath):
        self.model = C.ops.functions.load_model(filepath)
        
    def save (self, filepath):        
        self.model.save(filepath)

    def print_summary (self):
        print(self.model.summary())
  
    def __create_model(self, features, num_classes, num_hidden_units, dropout):      
        embedding = C.layers.Embedding(weights=self.wordvecs)(features)
        unidirectional1 = C.layers.Recurrence(C.layers.LSTM(num_hidden_units))(embedding)
        bidirectional1 = C.layers.Recurrence(C.layers.LSTM(num_hidden_units), go_backwards=True)(embedding)
        splice1 = C.splice(unidirectional1, bidirectional1)
        dropout1 = C.layers.Dropout(dropout)(splice1)
        unidirection2 = C.layers.Recurrence(C.layers.LSTM(num_hidden_units))(dropout1)
        bidirectional2 = C.layers.Recurrence(C.layers.LSTM(num_hidden_units), go_backwards=True)(dropout1)
        splice2 = C.splice(unidirection2, bidirectional2)
        dropout2 = C.layers.Dropout(dropout)(splice2)
        last = C.sequence.last(dropout2)
        model = C.layers.Dense(num_classes)(last)

        return model

    ##################################################
    # train
    ##################################################
    def train (self, train_file, output_resources_pickle_file, \
        network_type = 'unidirectional', \
        num_epochs = 1, batch_size = 50, \
        dropout = 0.2, reg_alpha = 0.0, \
        num_hidden_units = 150, num_layers = 1):
        
        train_X, train_Y = self.reader.read_and_parse_training_data(train_file, output_resources_pickle_file) 

        print("Data Shape: ")
        print(train_X.shape) # (15380, 613)
        print(train_Y.shape) # (15380, 613, 8)      
        #self.wordvecs.shape (66962, 50)
        
        print("Hyper parameters:")
        print("output_resources_pickle_file = {}".format(output_resources_pickle_file))
        print("network_type = {}".format(network_type))
        print("num_epochs= {}".format(num_epochs ))
        print("batch_size = {}".format(batch_size ))
        print("dropout = ".format(dropout ))
        print("reg_alpha = {}".format(reg_alpha ))
        print("num_hidden_units = {}".format(num_hidden_units))
        print("num_layers = {}".format(num_layers ))

        # Instantiate the model function;
        features = C.sequence.input_variable(self.wordvecs.shape[0])
        labels = C.input_variable(train_Y.shape[2], dynamic_axes=[C.Axis.default_batch_axis()])
        self.model = self.__create_model(features, train_Y.shape[2], num_hidden_units, dropout)

        plot_path = "./lstm_model.png"
        plot(self.model, plot_path)        
        
        # Instantiate the loss and error function
        loss = C.cross_entropy_with_softmax(self.model, labels)
        error = C.classification_error(self.model, labels)

        # LR schedule
        learning_rate = 0.02
        lr_schedule = C.learning_parameter_schedule(learning_rate)
        momentum_schedule = C.momentum_schedule(0.9, minibatch_size=batch_size)
        learner = C.fsadagrad(self.model.parameters, lr = lr_schedule, momentum = momentum_schedule, unit_gain = True)        

        # Setup the progress updater
        progress_printer = C.logging.ProgressPrinter(freq=100, first=10, tag='Training', num_epochs=num_epochs)

        # Instantiate the trainer. We have all data in memory. https://github.com/Microsoft/CNTK/blob/master/Manual/Manual_How_to_feed_data.ipynb
        print('Start training')       
        train_summary = loss.train((train_X.astype('float32'), train_Y.astype('float32')), parameter_learners=[learner], callbacks=[progress_printer])


    ##################################################
    # train
    ##################################################
    def train_keras (self, train_file, output_resources_pickle_file, \
        network_type = 'unidirectional', \
        num_epochs = 1, batch_size = 50, \
        dropout = 0.2, reg_alpha = 0.0, \
        num_hidden_units = 150, num_layers = 1):
        
        train_X, train_Y = self.reader.read_and_parse_training_data(train_file, output_resources_pickle_file)       

        print("Data Shape: ")
        print(train_X.shape)
        print(train_Y.shape)        
        
        print("Hyper parameters:")
        print("output_resources_pickle_file = {}".format(output_resources_pickle_file))
        print( "network_type = {}".format(network_type))
        print( "num_epochs= {}".format(num_epochs ))
        print("batch_size = {}".format(batch_size ))
        print("dropout = ".format(dropout ))
        print("reg_alpha = {}".format(reg_alpha ))
        print("num_hidden_units = {}".format(num_hidden_units))
        print("num_layers = {}".format(num_layers ))         
                
        self.model = Sequential()        
        self.model.add(Embedding(self.wordvecs.shape[0], self.wordvecs.shape[1], \
                                 input_length = train_X.shape[1], \
                                 weights = [self.wordvecs], trainable = False))                

        for i in range(0, num_layers):
            if network_type == 'unidirectional':
                # uni-directional LSTM
                self.model.add(LSTM(num_hidden_units, return_sequences = True))
            else:
                # bi-directional LSTM
                self.model.add(Bidirectional(LSTM(num_hidden_units, return_sequences = True)))
        
            self.model.add(Dropout(dropout))

        self.model.add(TimeDistributed(Dense(train_Y.shape[2], activation='softmax')))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        print(self.model.summary())

        self.model.fit(train_X, train_Y, epochs = num_epochs, batch_size = batch_size)