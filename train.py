import keras, tensorflow as tf, numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import Input, Embedding
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, CSVLogger

np.random.seed(1000)

import os, glob, argparse, time
from util.helpers import Tree_Dataset as Dataset, make_dir_if_not_exists as mkdir
from util.helpers import TestCallback

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

if __name__ == '__main__':
  
    parser = argparse.ArgumentParser(description='Train a test result prediction CNN.')

    parser.add_argument('data_directory', help='Data directory')
    parser.add_argument('checkpoints_directory', help='Checkpoints directory')
    parser.add_argument('-b', "--batch_size", type=int, help="batch size", default=32)
    parser.add_argument('-ed', "--embedding_dim", type=int, help="embedding_dim", default=32)
    parser.add_argument('-d', '--dropout', help='Probability to use for dropout', type=float, default=0.2)
    parser.add_argument('-e', "--epochs", type=int, help="epochs", default=40)
    parser.add_argument('-ot', "--only_test", help="only run predictions on test data", action="store_true")

    args = parser.parse_args()

    ## Load data
    data_directory = args.data_directory
    checkpoints_directory = args.checkpoints_directory
    mkdir(checkpoints_directory)
    batch_size = args.batch_size
    embedding_dim = args.embedding_dim


    dataset = Dataset(data_directory)
    tl_dict = dataset.get_tl_dict()
    rev_tl_dict = dataset.get_rev_tl_dict()

    num_train, num_validation, num_test, num_all = dataset.data_size
    print 'Training:', num_train, 'examples', '\nValidation:', num_validation, 'examples', '\nTest:',     num_test, 'examples', '\nAll:', num_all, 'examples' 
    print 'vocabulary size:', dataset.vocab_size

    vocab_size = dataset.vocab_size
    cnt_problem_ids = dataset.cnt_problem_IDs
    test_suite_size = dataset.test_suite_size
    max_subtrees = dataset.max_subtrees_per_program
    max_nodes = dataset.max_nodes_per_subtree

    if not args.only_test:
        # get training data
        programs, program_lengths, subtree_lengths, problem_ids, test_ids,  verdicts, buggy_subtrees, _ = dataset.get_batch(start=0, end=num_train, which='train')
        train_x = [programs, np.array(problem_ids), np.array(test_ids)]
        train_y = keras.utils.to_categorical(verdicts, num_classes=2)

        # get validation data
        v_programs, v_program_lengths, v_subtree_lengths, v_problem_ids, v_test_ids,  v_verdicts, v_buggy_subtrees, _ = dataset.get_batch(start=0, end=num_train, which='valid')
        valid_x = [v_programs, np.array(v_problem_ids), np.array(v_test_ids)]
        valid_y = keras.utils.to_categorical(v_verdicts, num_classes=2)

    # get test data
    t_programs, t_program_lengths, t_subtree_lengths, t_problem_ids, t_test_ids,  t_verdicts, t_buggy_subtrees, _ = dataset.get_batch(start=0, end=num_test, which='test')
    test_x = [t_programs, np.array(t_problem_ids), np.array(t_test_ids)]
    test_y = keras.utils.to_categorical(t_verdicts, num_classes=2)

    ## Define model
    program_input = Input(shape=(max_subtrees, max_nodes), dtype='int32', name='program')
    embedded_program = Embedding(output_dim=embedding_dim, input_dim=vocab_size, name='program_embedding')(program_input)

    # Use convolution over program input
    base = Conv2D(64, (1, 1), padding='same', activation='relu', name='base')(embedded_program)

    tower_1 = Conv2D(64, (1, max_nodes), padding='valid', activation='relu', name='tower1')(base)
    tower_2 = Conv2D(64, (3, max_nodes), strides=(3, 1), padding='valid', activation='relu', name='tower2')(base)

    program_features = keras.layers.concatenate([tower_1, tower_2], axis=1)
    program_vector = Flatten()(program_features)

    # embed test_id, and problem_id input
    problem_id_input = Input(shape=(1,), dtype='int32', name='problem_id')
    embedded_problem_id = Embedding(output_dim=5, input_dim=cnt_problem_ids, name='problem_id_embedding')(problem_id_input)
    embedded_problem_id = keras.layers.Reshape((5,))(embedded_problem_id)

    test_id_input = Input(shape=(1,), dtype='int32', name='test_id')
    embedded_test_id = Embedding(output_dim=5, input_dim=test_suite_size, name='test_id_embedding')(test_id_input)
    embedded_test_id = keras.layers.Reshape((5,))(embedded_test_id)
    
    merged = keras.layers.concatenate([program_vector, embedded_test_id, embedded_problem_id])

    hidden = Dense(128, activation='relu')(merged)
    hidden = Dense(64, activation='relu')(hidden)
    hidden = Dense(32, activation='relu')(hidden)

    logits = Dense(2)(hidden)
    output = keras.layers.Softmax()(logits)
    model = Model(inputs=[program_input, problem_id_input, test_id_input], outputs=output)

    model.summary()

    # load weights
    ckpts = glob.glob(os.path.join(checkpoints_directory, "weights.*-*.hdf5"))
    best_checkpoint, initial_epoch = None, 0
    if len(ckpts) > 0:
        for ckpt in ckpts:
            ckpt_epoch = int(ckpt.split('-')[0].split('.')[1])
            if initial_epoch < ckpt_epoch:
                initial_epoch = ckpt_epoch
                best_checkpoint = ckpt
    
        model.load_weights(best_checkpoint)
        print 'loaded checkpoint:', best_checkpoint, '\nWill resume training from epoch:', initial_epoch

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if args.only_test:
        ## Test
        loss_and_metrics = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=1)
        print model.metrics_names, loss_and_metrics
    else:
        ## Train & validate
        filepath = os.path.join(checkpoints_directory, "weights.{epoch:02d}-{val_acc:.2f}.hdf5")
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
        csv_logger = CSVLogger(os.path.join(checkpoints_directory, 'log.csv'), append=True, separator=';')
        time_callback = TimeHistory()
        test_eval_callback = TestCallback((test_x, test_y))
        callbacks_list = [test_eval_callback,checkpoint,time_callback,csv_logger]
        model.fit(train_x, train_y, batch_size=batch_size, epochs=50, verbose=1, validation_data=(valid_x, valid_y), \
        initial_epoch=initial_epoch, shuffle=True, callbacks=callbacks_list)
        times = time_callback.times
        total_time_taken = 0
        for loop_epoch_id, time_taken in enumerate(times):
            print 'epoch:', loop_epoch_id+1, 'time_taken:', time_taken
            total_time_taken += time_taken
        print 'total_time_taken:', total_time_taken
        
        # Test
        loss_and_metrics = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=1)
        print model.metrics_names, loss_and_metrics

