import subprocess32 as subprocess
import os, tempfile, time, sys, random, copy, ast, re, py_compile, json, glob #, sqlite3, threading
import numpy as np
import pickle as cp
import tensorflow as tf
from keras.callbacks import Callback

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

def get_top_k(arr, k):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    top_k_indices = np.argpartition(arr, -k)[-k:]
    top_k_indices = list(reversed(top_k_indices[np.argsort(arr[top_k_indices])]))
    return top_k_indices, arr[top_k_indices]

def get_least_k(arr, k):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    least_k_indices = np.argpartition(arr, k)[:k]
    sorted_least_k_indices = least_k_indices[np.argsort(arr[least_k_indices])]
    return sorted_least_k_indices, arr[sorted_least_k_indices]

def prepend_line_numbers(program, offset=1):
    return '\n'.join(['[%-2d] ' % (line_number+offset) + line for line_number, line in enumerate(program.split('\n')) if line.strip() != ''])

def remove_non_ascii(text):
    return ''.join([i if ord(i) < 128 else '' for i in text])

def remove_empty_lines(program):
    lines = [line for line in program.split('\n') if len(line.strip()) > 0]
    return '\n'.join(lines)

def remove_all_white_space(line):
    return ''.join(line.split())

def normalize_brackets(program):
    program = program.replace('\r', '\n')
    lines = [line for line in program.split('\n') if len(line.strip()) > 0]
    
    if len(lines) == 1:
        raise ValueError()

    #for i, line in enumerate(lines):
    for i in range(len(lines)-1, -1, -1):
        line = lines[i]
        wsr_line = remove_all_white_space(line)
        if wsr_line == '}' or wsr_line == '}}' or wsr_line == '}}}' or wsr_line == '};' \
        or wsr_line == '}}}}' or wsr_line == '}}}}}' or wsr_line == '{' or wsr_line == '{{':
            if i > 0:
                lines[i-1] += ' ' + line.strip()
                lines[i]    = ''
            else:
                # can't handle this case!
                raise ValueError()
                return ''

    # Remove empty lines
    for i in range(len(lines)-1, -1, -1):
        if lines[i] == '':
            del lines[i]

    for line in lines:
        assert(lines[i].strip() != '')

    return '\n'.join(lines)

# a coin flip with given winning probability
def coin_flip(probability, rng=None):
    if not rng:
        return np.random.random_sample() < probability
    return rng.random_sample() < probability

def done(msg=''):
    if msg == '':
        print('done at', time.strftime("%d/%m/%Y"), time.strftime("%H:%M:%S"))
    else:
        print(msg, ',done at', time.strftime("%d/%m/%Y"), time.strftime("%H:%M:%S"))


def make_dir_if_not_exists(path):
    try:
        os.makedirs(path)
    except:
        pass

def get_curr_time_string():
    return time.strftime("%b %d %Y %H:%M:%S")

def get_variables(filename, program):
    if type(filename) == int:
        filename = str(filename) + '.c'
    elif not filename.endswith('.c'):
        filename += '.c'
    filename = os.path.join('/tmp', filename)
    # remove empty lines
    program = '\n'.join([line for line in program.split('\n') if len(line.strip())>0])
    with open(filename, 'w') as f:
        f.write(program)
        
    # ctags result sorted numerically on line numbers
    shell_string = "ctags -x --c-kinds=lvf %s | sort -k 3 -n" % filename

    try:
        result = subprocess.check_output(shell_string, timeout=30, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        result = e.output
        print result
        raise

    # print result
    lines = [line for line in result.split('\n') if len(line.strip()) > 0]
    return lines

class Color_Printer:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

    @staticmethod
    def highlight(with_what, *args):
        args = list(map(str, list(args)))
        msg = with_what + ' '.join(args) + Color_Printer.END
        print(msg)

    @staticmethod
    def bold(*args):
        Color_Printer.highlight(Color_Printer.BOLD, args)

    @staticmethod
    def warn(*args):
        Color_Printer.highlight(Color_Printer.RED, args)
        
    @staticmethod
    def yellow(*args):
        Color_Printer.highlight(Color_Printer.YELLOW, args)
        
    @staticmethod
    def green(*args):
        Color_Printer.highlight(Color_Printer.GREEN, args)

    @staticmethod
    def underline(*args):
        Color_Printer.highlight(Color_Printer.UNDERLINE, args)

class logger():
    def _open(self):
        if not self.open:
            try:
                self.handle = open(self.log_file, 'a+')
                self.open = True
            except Exception as e:
                print(os.getcwd())
                raise e
        else:
            raise RuntimeError('ERROR: Trying to open already opened log-file!')

    def close(self):
        if self.open:
            self.handle.close()
            self.open = False
        else:
            raise RuntimeError('ERROR: Trying to close already closed log-file!')

    def __init__(self, log_file, move_to_logs_dir=True):
        self.log_file = log_file + '.txt' if '.txt' not in log_file else log_file
        if move_to_logs_dir and not self.log_file.startswith('logs/'):
            self.log_file = os.path.join('logs', self.log_file)
        self.open = False
        self.handle = None
        self._open()

        self.terminal = sys.stdout

        self.log('\n\n-----------------------| Started logging at: {} |----------------------- \n'.format(get_curr_time_string()))

    # for backward compatibility
    def log(self, *msg_list):

        msg_list = list(map(str, msg_list))
        msg = ' '.join(msg_list)

        if not self.open:
            self._open()

        self.handle.write(msg + '\n')
        self.handle.flush()

        print(msg)
        self.terminal.flush()

    # set ** sys.stdout = logger(filename) ** and then simply use print call
    def write(self, message):
        if not self.open:
            self._open()

        self.handle.write(message)
        self.terminal.write(message)

        self.flush()

    @property
    def terminal(self):
        return self.terminal

    def flush(self):
        self.terminal.flush()
        self.handle.flush()

pass


# returns dict(value:key) for a give dict(key:value)
def get_rev_dict(dict_):
    # assert len(dict_) > 0, 'passed dict has size zero'
    rev_dict_ = {}
    for key, value in dict_.items():
        rev_dict_[value] = key

    return rev_dict_


def get_best_checkpoint(checkpoint_directory):

    def get_best_checkpoint_in_dir(checkpoint_dir):
        best_checkpoint = None
        for checkpoint_name in os.listdir(checkpoint_dir):
            if 'meta' in checkpoint_name:
                this_checkpoint = int(checkpoint_name[17:].split('.')[0])

                if best_checkpoint is None or this_checkpoint > best_checkpoint:
                    best_checkpoint = this_checkpoint

        return best_checkpoint

    bc = get_best_checkpoint_in_dir(os.path.join(checkpoint_directory, 'best'))
    if bc is None:
        bc = get_best_checkpoint_in_dir(checkpoint_directory)
    if bc is None:
        raise ValueError('No checkpoints found!')
    return bc


def prepare_program_batch(programs, max_subtrees_per_program, max_nodes_per_subtree):
    if type(programs) == str: programs = json.loads(programs)

    _batch_size = len(programs)
    _programs = np.zeros(shape=[_batch_size, max_subtrees_per_program, max_nodes_per_subtree], dtype=np.int32)
    _program_lengths = np.zeros(shape=[_batch_size], dtype=np.int32)
    _subtree_lengths = np.zeros(shape=[_batch_size, max_subtrees_per_program], dtype=np.int32)
    
    for p, prog in enumerate(programs):
        _program_lengths[p] = len(prog)
        for i, seq in enumerate(prog):
            _subtree_lengths[p, i] = len(seq)
            for j, element in enumerate(seq):
                    _programs[p, i, j] = element

    return _programs, _program_lengths, _subtree_lengths

def prepare_batch(sequences, msg=False):
    sequence_lengths = [len(seq) for seq in sequences]
    batch_size = len(sequences)
    max_sequence_length = max(sequence_lengths)

    if msg:
        print('max_sequence_length', max_sequence_length)

    inputs = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # initialize with _pad_ = 0

    for i, seq in enumerate(sequences):
        for j, element in enumerate(seq):
            try:
                inputs[i, j] = element
            except:
                print ('sequences:', sequences)
                print ('i:', i, 'j:', j, 'batch_size:', batch_size)
                raise

    return [inputs, np.array(sequence_lengths)]

def split_list(a_list, delimiter, keep_delimiter=True):
    output = []
    temp = []
    for each in a_list:
        if each == delimiter:
            if keep_delimiter:
                temp.append(delimiter)
            output.append(temp)
            temp = []
        else:
            temp.append(each)
    if len(temp) > 0:
        output.append(temp)
    return output

def get_single_cell(memory_dim, initializer, dropout, keep_prob, which):
    if which == 'LSTM':
        constituent_cell = tf.contrib.rnn.LSTMCell(memory_dim, initializer=initializer, state_is_tuple=True)
    elif which == 'GRU':
        constituent_cell = tf.contrib.rnn.GRUCell(memory_dim, kernel_initializer=initializer)
    else:
        raise ValueError('Unsupported rnn cell type: %s' % which)
    if dropout != 0:
        constituent_cell = tf.contrib.rnn.DropoutWrapper(constituent_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    return constituent_cell

def new_RNN_cell(memory_dim, num_layers, initializer, dropout=0, keep_prob=None, which='LSTM'):

    assert memory_dim is not None and num_layers is not None and dropout is not None, 'At least one of the arguments is passed as None'
    if num_layers > 1:
        return tf.contrib.rnn.MultiRNNCell([get_single_cell(memory_dim, initializer, dropout, keep_prob, which) for _ in range(num_layers) ])
    else:
        return get_single_cell(memory_dim, initializer, dropout, keep_prob, which)


#def load_dictionaries(destination, name='all_dicts.npy'):
#    tl_dict, rev_tl_dict = np.load(os.path.join(destination, name))
#    return tl_dict, rev_tl_dict


class Dataset:
    def _deserialize(self, data_folder):
        train_ex = np.load(os.path.join(
            data_folder, 'examples-train.npy'))
        valid_ex = np.load(os.path.join(
            data_folder, 'examples-validation.npy'))
        test_ex = np.load(os.path.join(
            data_folder, 'examples-test.npy'))

        assert train_ex is not None and valid_ex is not None and test_ex is not None
        return train_ex, valid_ex, test_ex

    def __init__(self, data_folder, load_real_test_data=False, load_seeded_test_data=False, load_only_dicts=False, shuffle=False, seed=1189):
        self.rng = np.random.RandomState(seed)
        self.tl_dict, self.rev_tl_dict = np.load(os.path.join(data_folder, 'all_dicts.npy'))

        assert self.tl_dict is not None and self.rev_tl_dict is not None

        if load_only_dicts:
            return

        if not shuffle:
            # Load originals
            self.train_ex, self.valid_ex, self.test_ex = self._deserialize(data_folder)
            print("Successfully loaded data.")

        else:
            try:  # to load pre-generated shuffled data
                self.train_ex, self.valid_ex, self.test_ex = self._deserialize(os.path.join(data_folder, 'shuffled'))
                print("Successfully loaded shuffled data.")

            # or generate it
            except IOError:
                print("Generating shuffled data...")
                sys.stdout.flush()

                # Load originals
                self.train_ex, self.valid_ex, self.test_ex = self._deserialize(data_folder)

                # Shuffle
                self.rng.shuffle(self.train_ex)
                self.rng.shuffle(self.valid_ex)
                self.rng.shuffle(self.test_ex)

                # Save for later
                make_dir_if_not_exists(os.path.join(data_folder, 'shuffled'))

                np.save(os.path.join(data_folder, 'shuffled',
                                     'examples-train.npy'), self.train_ex)
                np.save(os.path.join(data_folder, 'shuffled',
                                     'examples-validation.npy'), self.valid_ex)
                np.save(os.path.join(data_folder, 'shuffled',
                                     'examples-test.npy'), self.test_ex)
                

    def get_raw_data(self):
        return self.train_ex, self.valid_ex, self.test_ex

    def get_tl_dict(self):
        return self.tl_dict

    def get_rev_tl_dict(self):
        return self.rev_tl_dict

    @property
    def data_size(self):
        return len(self.train_ex), len(self.valid_ex), len(self.test_ex)

    @property
    def vocab_size(self):
        return len(self.tl_dict)

    @classmethod
    def prepare_batch(self, sequences, msg=False):
        sequence_lengths = [len(seq) for seq in sequences]
        batch_size = len(sequences)
        max_sequence_length = max(sequence_lengths)
        
        if msg:
            print 'max_sequence_length', max_sequence_length
        
        inputs = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # initialize with _pad_ = 0
        
        for i, seq in enumerate(sequences):
            for j, element in enumerate(seq):
                inputs[i,j] = element
                
        return [inputs, np.array(sequence_lengths)]
    
    def get_batch(self, start, end, which='train'):
        # print start, end, '\n'
        try:
            if which=='train':
                X, Y = zip(*self.train_ex[start:end])
            elif which=='valid':
                X, Y = zip(*self.valid_ex[start:end])
            elif which=='test':
                X, Y = zip(*self.test_ex[start:end])
            else:
                raise ValueError('choose one of train/valid/test for which')
        except:
            print self.train_ex[0:2]
            raise
        Xs, X_lens = self.prepare_batch(X)
        return Xs, X_lens, Y

class Tree_Dataset:
    def _deserialize(self, data_folder):
        # train_ex = np.load(os.path.join(
        #     data_folder, 'examples-train.pkl'))
        # valid_ex = np.load(os.path.join(
        #     data_folder, 'examples-validation.pkl'))
        # test_ex = np.load(os.path.join(
        #     data_folder, 'examples-test.pkl'))
        with open(os.path.join(data_folder, 'examples-train.pkl'), 'r') as f:
            train_ex = cp.load(f)
        with open(os.path.join(data_folder, 'examples-validation.pkl'), 'r') as f:
            valid_ex = cp.load(f)
        with open(os.path.join(data_folder, 'examples-test.pkl'), 'r') as f:
            test_ex = cp.load(f)

        assert train_ex is not None and valid_ex is not None and test_ex is not None
        return train_ex, valid_ex, test_ex

    def __init__(self, data_folder, load_only_dicts=False, seed=1189):
        self.rng = np.random.RandomState(seed)
        try:
            # self.tl_dict, self.test_dict, self.problem_id_dict, self.program_id_dict, self.info, self.var_name_dict = \
            #             np.load(os.path.join(data_folder, 'all_dicts.npy'))
            self.tl_dict, self.test_dict, self.problem_id_dict, self.info, self.var_name_dict = \
                        np.load(os.path.join(data_folder, 'all_dicts.npy'))
        except:
            self.org_tl_dict, self.normalized_tl_dict, self.test_dict, self.problem_id_dict, self.program_id_dict, \
             self.info, self.var_name_dict = np.load(os.path.join(data_folder, 'all_dicts.npy'))
            self.tl_dict = self.normalized_tl_dict
        self.rev_tl_dict = get_rev_dict(self.tl_dict)
        self.max_subtrees_per_program = self.info['max_subtrees_per_program']
        self.max_nodes_per_subtree = self.info['max_nodes_per_subtree']

        assert self.tl_dict is not None and self.test_dict is not None and self.problem_id_dict is not None

        if load_only_dicts:
            return

        # Load originals
        self.train_ex, self.valid_ex, self.test_ex = self._deserialize(data_folder)
        self.all_ex = None
        print("Successfully loaded data.")
        self.shuffle()

    def shuffle(self):
        self.rng.shuffle(self.train_ex)
        
    def get_raw_data(self):
        return self.train_ex, self.valid_ex, self.test_ex

    def get_tl_dict(self):
        return self.tl_dict

    def get_rev_tl_dict(self):
        return self.rev_tl_dict

    def get_test_dict(self):
        return self.test_dict
    
    def get_problem_id_dict(self):
        return self.problem_id_dict

    def get_program_id_dict(self):
        return self.program_id_dict

    @property
    def data_size(self):
        return len(self.train_ex), len(self.valid_ex), len(self.test_ex), len(self.train_ex)+len(self.valid_ex)+len(self.test_ex)

    @property
    def vocab_size(self):
        return len(self.tl_dict)

    @property
    def test_suite_size(self):
        return len(self.test_dict)
    
    @property
    def cnt_problem_IDs(self):
        return len(self.problem_id_dict)
    
    def prepare_batch(self, programs):        
        _batch_size = len(programs)
        _programs = np.zeros(shape=[_batch_size, self.max_subtrees_per_program, self.max_nodes_per_subtree], dtype=np.int32)
        _program_lengths = np.zeros(shape=[_batch_size], dtype=np.int32)
        _subtree_lengths = np.zeros(shape=[_batch_size, self.max_subtrees_per_program], dtype=np.int32)

        try:
            for p, prog in enumerate(programs):
                _program_lengths[p] = len(prog)
                for i, seq in enumerate(prog):
                    _subtree_lengths[p, i] = len(seq)
                    for j, element in enumerate(seq):
                            _programs[p, i, j] = element
        except:
            print 'len(prog):', len(prog)
            raise

        return _programs, _program_lengths, _subtree_lengths
    
    def get_batch(self, start, end, which='train'):
        # print start, end, '\n'
        try:
            if which=='train':
                problem_ids, program_ids, test_ids, programs, verdicts, buggy_subtrees = zip(*self.train_ex[start:end])
            elif which=='valid':
                problem_ids, program_ids, test_ids, programs, verdicts, buggy_subtrees = zip(*self.valid_ex[start:end])
            elif which=='test':
                problem_ids, program_ids, test_ids, programs, verdicts, buggy_subtrees = zip(*self.test_ex[start:end])
            elif which=='all':
                try:
                    problem_ids, program_ids, test_ids, programs, verdicts, buggy_subtrees = zip(*self.all_ex[start:end])
                except:
                    self.all_ex = self.train_ex + self.valid_ex + self.test_ex
                    problem_ids, program_ids, test_ids, programs, verdicts, buggy_subtrees = zip(*self.all_ex[start:end])
            else:
                raise ValueError('choose one of train/valid/test for which')
        except:
            print self.train_ex[0:2]
            raise
            
        try:
            programs, program_lengths, subtree_lengths = self.prepare_batch(programs)
        except:
            max_st, max_nd = 0, 0
            for program_id, program in zip(program_ids, programs):
                if len(program) > self.max_subtrees_per_program:
                    print 'program_id:', program_id, 'len(program):', len(program)
                max_st = max(max_st, len(program))
                max_nd = max(max_nd, max([len(subtree) for subtree in program]))
            print 'max_st:', max_st, 'max_nd:', max_nd
            raise

        if which == 'all':
            return programs, program_lengths, subtree_lengths, problem_ids, test_ids, verdicts, buggy_subtrees, program_ids
        else:
            return programs, program_lengths, subtree_lengths, problem_ids, test_ids, verdicts, buggy_subtrees, None


# This function returns the line where we indexed into the string
# [program_string] is a tokenized program
# [char_index] is an index into program_string
def isolate_line(program_string, char_index):
    begin = program_string[:char_index].rfind('~') - 2

    while begin - 2 > 0 and program_string[begin - 2] in [str(i) for i in range(10)]:
        begin -= 2

    if program_string[char_index:].find('~') == -1:
        end = len(program_string)
    else:
        end = char_index + program_string[char_index:].find('~') - 2

        while end - 2 > 0 and program_string[end - 2] in [str(i) for i in range(10)]:
            end -= 2

        end -= 1

    return program_string[begin:end]

# Extracts the line number for a tokenized line, e.g. '1 2 ~ _<token>_ ...' returns 12
def extract_line_number(line):
    number = 0
    never_entered = True

    for token in line.split('~')[0].split():
        never_entered = False
        number *= 10
        try:
            number += int(token) - int('0')
        except ValueError:
            raise

    if never_entered:
        raise FailedToGetLineNumberException(line)

    return number

# Input: tokenized program
# Returns: array of lines, each line is tokenized
def get_lines(program_string):
#     print 'start get_lines with arg:', '\n', program_string
    tokens = program_string.split()
    ignore_tokens = ['~'] + [chr(n + ord('0')) for n in range(10)]

    # Split token string into lines
    lines = []

    for token in tokens:
#         print 'token:', token
        if token in ignore_tokens and token == '~':
            if len(lines) > 0:
                lines[-1] = lines[-1].rstrip(' ')
            lines.append('')
        elif token not in ignore_tokens:
            lines[-1] += token + ' '
#         print 'lines:', lines

    #for line in lines:
    #    assert line.strip() != '', "program_string: \n" + program_string
#     print 'end get_lines()'
    return lines

# Input: output of get_lines() (tokenized lines)
# Result: Tokenized program
def recompose_program(lines):

#     for line in lines:
#         assert len(line.strip()) > 0, 'found an empty line in the input of fn:recompose_programs: \n' + '\n'.join(lines)
#     pass

    recomposed_program = ''

    for i, line in enumerate(lines):
        for digit in str(i):
            recomposed_program += digit + ' '

        recomposed_program += '~ '
        recomposed_program += line + ' '

    return recomposed_program

# Fetches one specific line from the program
def fetch_line(program_string, line_number, include_line_number=True):
    result = ''

    if include_line_number:
        for digit in str(line_number):
            result += digit + ' '

        result += '~ '

    result += get_lines(program_string)[line_number]
    #assert result.strip() != ''
    return result


# def tokens_to_source(tokens, name_dict, clang_format=False, name_seq=None, literal_seq=None):
#     result = ''
#     type_ = None

#     reverse_name_dict = {}
#     name_count = 0

#     for k,v in name_dict.iteritems():
#         reverse_name_dict[v] = k

#     for token in tokens.split():
#         try:
#             prev_type_was_op = (type_ == 'op')

#             type_, content = token.split('>_')
#             type_ = type_.lstrip('_<')

#             if type_ == 'id':
#                 if name_seq is not None:
#                     content = name_seq[name_count]
#                     name_count += 1
#                 else:
#                     try:
#                         content = reverse_name_dict[content.rstrip('@')]
#                     except KeyError:
#                         content = 'new_id_' + content.rstrip('@')
#             elif type_ == 'number':
#                 content = content.rstrip('#')

#             if type_ == 'directive' or type_ == 'include' or type_ == 'op' or type_ == 'type' or type_ == 'keyword' or type_ == 'APIcall':
#                 if type_ == 'op' and prev_type_was_op:
#                     result = result[:-1] + content + ' '
#                 else:
#                     result += content + ' '
#             elif type_ == 'id':
#                 result += content + ' '
#             elif type_ == 'number':
#                 if literal_seq is None:
#                     result += '0 '
#                 else:
#                     try:
#                         result += '%s ' % literal_seq[0]
#                         literal_seq = literal_seq[1:]
#                     except IndexError as e:
#                         print 'IndexError in tokens_to_source2, number literal parsing, using hack!'
#                         result += '0 '
#                     except Exception as e:
#                         print literal_seq
#                         raise
#             elif type_ == 'string':
#                 if literal_seq is None:
#                     result += '"String" '
#                 else:
#                     try:
#                         result += '%s ' % literal_seq[0]
#                         literal_seq = literal_seq[1:]
#                     except IndexError as e:
#                         print 'IndexError in tokens_to_source2, string literal parsing, using hack!'
#                         result += '"DeEpFiX" '
#             elif type_ == 'char':
#                 if literal_seq is None:
#                     result += "'c' "
#                 else:
#                     try:
#                         result += '%s ' % literal_seq[0]
#                         literal_seq = literal_seq[1:]
#                     except IndexError as e:
#                         print 'IndexError in tokens_to_source2, char literal parsing, using hack!'
#                         result += "'z' "
#         except ValueError:
#             if token == '~':
#                 result += '\n'

#     if not clang_format:
#         return result

#     source_file = tempfile.NamedTemporaryFile(suffix=".c", delete=False)
#     source_file.write(result)
#     source_file.close()

#     shell_string = 'clang-format %s' % source_file.name
#     clang_output = subprocess.check_output(shell_string, timeout=30, shell=True)
#     os.unlink(source_file.name)

#     return clang_output


def tokens_to_source(tokens, clang_format=False):
    result = ''
    type_ = None

    # name_count = 0

    for token in tokens.split():
        try:
            prev_type_was_op = (type_ == 'op')

            type_, content = token.split('>_')
            type_ = type_.lstrip('_<')

            if type_ == 'id':
                content = content.rstrip('@')
            elif type_ == 'number':
                content = content.rstrip('#')

            if type_ == 'directive' or type_ == 'include' or type_ == 'op' or type_ == 'type' or type_ == 'keyword' or type_ == 'APIcall':
                if type_ == 'op' and prev_type_was_op:
                    result = result[:-1] + content + ' '
                else:
                    result += content + ' '
            elif type_ == 'id' or type_ == 'number' or type_ == 'string' or type_ == 'char':
                result += content.replace('--sL-space--', ' ') + ' '

        except ValueError:
            if token == '~':
                result += '\n'
            #else:
            #    print 'token:', token, '\n\n', tokens.split()
            #    raise
                

    if not clang_format:
        return result.strip()

    source_file = tempfile.NamedTemporaryFile(suffix=".c", delete=False)
    source_file.write(result)
    source_file.close()

    shell_string = 'clang-format %s' % source_file.name
    clang_output = subprocess.check_output(shell_string, timeout=30, shell=True)
    os.unlink(source_file.name)

    return clang_output.strip()


def clang_format(source_file):
    shell_string = 'clang-format %s' % source_file
    clang_output = subprocess.check_output(shell_string, timeout=30, shell=True)
    return clang_output.strip()

def clang_format_from_source(source_string):
    name1 = int(time.time() * 10**6)
    name2 = np.random.random_integers(0, 1000)
    filename = '/tmp/tempfile_%d_%d.c' % (name1, name2)
    
    with open(filename, 'w+') as f:
        f.write(source_string)

    result = clang_format(filename)
    os.unlink(filename)
    return result

def get_error_list(error_message):
    error_list = []
    for line in error_message.splitlines():
        if 'error:' in line or 'Error:' in line:
            error_list.append(line)
    return error_list

def compile_file(filename, which='gcc'):
    assert filename.endswith('.c'), filename

    outfile = filename + '.out'

    if which == 'gcc':
        shell_string = "gcc -w -std=c99 -pedantic %s -lm -o %s" % (filename, outfile)
    else:
        shell_string = "clang -w -std=c99 -pedantic %s -lm -o %s" % (filename, outfile)

    try:
        result = subprocess.check_output(shell_string, timeout=5, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        result = e.output
    except subprocess.TimeoutExpired as e:
        result = '%s:0:0: error: compilation timedout' % filename
        result = '\n'.join([result]*20)

    result = remove_non_ascii(result)
    return outfile, result

def compilation_errors(program_id, program_string):
    name1 = int(time.time() * 10**6)
    name2 = np.random.random_integers(0, 1000)
    filename = '/tmp/%s_%d_%d.c' % (str(program_id), name1, name2)
    with open(filename, 'w+') as f:
        f.write(program_string)

    outfile, result = compile_file(filename)

    try:
        os.unlink(filename)
        os.unlink(outfile)
    except:
        pass

    error_list = get_error_list(result)
    return error_list
