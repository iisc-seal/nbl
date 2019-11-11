from abc import abstractmethod
from util.helpers import get_lines

class UnexpectedTokenException(Exception):
    pass

class EmptyProgramException(Exception):
    '''In fn tokenizer:get_lines(), positions are empty, most probably the input program \
       is without any newline characters or has a special character such as ^A'''
    pass
    
class FailedTokenizationException(Exception):
    '''Failed to create line-wise id_sequence or literal_sequence or both'''
    pass

class Tokenizer:

    @abstractmethod
    def tokenize(self, code, keep_format_specifiers=False, keep_names=True, \
                 keep_literals=False):
        return NotImplemented
    
    def enforce_restrictions(self, result):
        # Remove 2D arrays
        #if '_<op>_] _<op>_[' in result or '_<op>_* _<op>_*' in result:
        #    result = ''
        
        # (Don't) Remove programs with long lines
#         lines = get_lines(result)
#         
#         for line in lines:
#             if len(line.split()) > get_max_line_length():
#                 result = ''
    
        return result
    