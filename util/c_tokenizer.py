import collections
import regex as re
from helpers import get_lines, recompose_program
from tokenizer import Tokenizer, UnexpectedTokenException, EmptyProgramException

Token = collections.namedtuple('Token', ['typ', 'value', 'line', 'column'])

class C_Tokenizer(Tokenizer):
    _keywords = ['auto', 'break', 'case', 'const', 'continue', 'default', \
                'do', 'else', 'enum', 'extern', 'for', 'goto', 'if', \
                'register', 'return', 'signed', 'sizeof', 'static', 'switch', \
                'typedef', 'void', 'volatile', 'while', 'EOF', 'NULL', \
                'null', 'struct', 'union']
    _includes = ['stdio.h', 'stdlib.h', 'string.h', 'math.h', 'malloc.h', \
                'stdbool.h', 'cstdio', 'cstdio.h', 'iostream', 'conio.h']
    _calls    = ['printf', 'scanf', 'cin', 'cout', 'clrscr', 'getch', 'strlen', \
                'gets', 'fgets', 'getchar', 'main', 'malloc', 'calloc', 'free']
    _types    = ['char', 'double', 'float', 'int', 'long', 'short', 'unsigned']

    def _escape(self, string):
        return repr(string)[1:-1]

    def _tokenize_code(self, code):
        keywords = {'IF', 'THEN', 'ENDIF', 'FOR', 'NEXT', 'GOSUB', 'RETURN'}
        token_specification = [
            ('comment', r'\/\*(?:[^*]|\*(?!\/))*\*\/|\/\*([^*]|\*(?!\/))*\*?|\/\/[^\n]*'),
            ('directive', r'#\w+'),
            ('string', r'"(?:[^"\n]|\\")*"?'),
            ('char', r"'(?:\\?[^'\n]|\\')'"),
            ('char_continue', r"'[^']*"),
            ('number',  r'[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'),
            ('include',  r'(?<=\#include) *<([_A-Za-z]\w*(?:\.h))?>'),
            #('op',  r'\(|\)|\[|\]|{|}|->|<<|>>|\*\*|\|\||&&|--|\+\+|[-+*|&%\/=]=|[-<>~!%^&*\/+=?|.,:;#]'),
            ('op',  r'\(|\)|\[|\]|{|}|->|<<|>>|\*\*|\|\||&&|--|\+\+|[-+*|&%\/=<>]=|[-<>~!%^&*\/+=?|.,:;#]'),
            ('name',  r'[_A-Za-z]\w*'),
            ('whitespace',  r'\s+'),
            ('nl', r'\\\n?'),
            ('MISMATCH',r'.'),            # Any other character
        ]
        tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
        line_num = 1
        line_start = 0
        for mo in re.finditer(tok_regex, code):
            kind = mo.lastgroup
            value = mo.group(kind)
            if kind == 'NEWLINE':
                line_start = mo.end()
                line_num += 1
            elif kind == 'SKIP':
                pass
            elif kind == 'MISMATCH':
                yield UnexpectedTokenException('%r unexpected on line %d' % (value, line_num))
            else:
                if kind == 'ID' and value in keywords:
                    kind = value
                column = mo.start() - line_start
                yield Token(kind, value, line_num, column)

    def _sanitize_brackets(self, tokens_string):
        lines = get_lines(tokens_string)

        if len(lines) == 1:
            raise EmptyProgramException(tokens_string)

        #for i, line in enumerate(lines):
        for i in range(len(lines)-1, -1, -1):
            line = lines[i]
            
            if line.strip() == '_<op>_}' or line.strip() == '_<op>_} _<op>_}' \
               or line.strip() == '_<op>_} _<op>_} _<op>_}' or line.strip() == '_<op>_} _<op>_;' \
               or line.strip() == '_<op>_} _<op>_} _<op>_} _<op>_}' \
               or line.strip() == '_<op>_{' \
               or line.strip() == '_<op>_{ _<op>_{':
                if i > 0:
                    lines[i-1] += ' ' + line.strip()
                    lines[i]    = ''
                else:
                    # can't handle this case!
                    return ''

        # Remove empty lines
        for i in range(len(lines)-1, -1, -1):
            if lines[i] == '':
                del lines[i]

        for line in lines:
            assert(lines[i].strip() != '')

        return recompose_program(lines)

    def tokenize(self, code, keep_names=True, \
                 keep_literals=True, prepend_line_numbers=True):
        result = '0 ~ ' if prepend_line_numbers else ''

        names = ''
        line_count = 1
        name_dict = {}
        name_sequence = []

        regex = '%(d|i|f|c|s|u|g|G|e|p|llu|ll|ld|l|o|x|X)'
        isNewLine = True

        # Get the iterable
        my_gen = self._tokenize_code(code)

        while True:
            try:
                token = next(my_gen)
            except StopIteration:
                break

            if isinstance(token, Exception):
                return '', '', ''

            type_ = str(token[0])
            value = str(token[1])

            if value in self._keywords:
                result += '_<keyword>_' + self._escape(value) + ' '
                isNewLine = False

            elif type_ == 'include':
                result += '_<include>_' + self._escape(value).lstrip() + ' '
                isNewLine = False

            elif value in self._calls:
                result += '_<APIcall>_' + self._escape(value) + ' '
                isNewLine = False

            elif value in self._types:
                result += '_<type>_' + self._escape(value) + ' '
                isNewLine = False

            elif type_ == 'whitespace' and (('\n' in value) or ('\r' in value)):
                if isNewLine:
                    continue
                if prepend_line_numbers:
                    result += ' '.join(list(str(line_count))) + ' ~ '
                    line_count += 1
                isNewLine = True

            elif type_ == 'whitespace' or type_ == 'comment' or type_ == 'nl':
                pass

            elif 'string' in type_:
                # matchObj = [m.group().strip() for m in re.finditer(regex, value)]
                # if matchObj and keep_format_specifiers:
                #     for each in matchObj:
                #         result += each + ' '
                # else:
                #     result += '_<string>_' + ' '
                value = value.replace(' ', '--sL-space--')
                result += '_<string>_' + value + ' '
                isNewLine = False

            elif type_ == 'name':
                if keep_names:
                    if self._escape(value) not in name_dict:
                        name_dict[self._escape(value)] = str(len(name_dict) + 1)
                    
                    name_sequence.append(self._escape(value))
                    # result += '_<id>_' + name_dict[self._escape(value)] + '@ '
                    # names += '_<id>_' + name_dict[self._escape(value)] + '@ '
                    result += '_<id>_' + self._escape(value) + '@ '
                    names += '_<id>_' + self._escape(value) + '@ '
                else:
                    result += '_<id>_' + '@ '
                isNewLine = False

            # need to keep 'r' in the end for pointer deref mutation to work.
            elif type_ == 'number':
                if keep_literals:
                    result += '_<number>_' + self._escape(value) + '# '
                else:
                    result += '_<number>_' + '# '
                isNewLine = False

            elif 'char' in type_ or value == '':
                result += '_<' + type_.lower() + '>_' + value + ' '
                isNewLine = False

            else:
                converted_value = self._escape(value).replace('~', 'TiLddE')
                result += '_<' + type_ + '>_' + converted_value + ' '

                isNewLine = False

        result = result[:-1]
        names = names[:-1]

        if result.endswith('~'):
            idx = result.rfind('}')
            result = result[:idx+1]

        if prepend_line_numbers:
            return self.enforce_restrictions(self._sanitize_brackets(result))#, name_dict, name_sequence
        else:
            return result#, name_dict, name_sequence


def test_tokenize():
    test_prog = '''#include <stdio.h> 
#include <stdlib.h> 
double id_1 ( int id_2 ) 
{

{ 
int id_3 ; 
double id_4 ; 
for ( id_3 = 0 ; id_3 <= id_2 ; id_3 ++) 
id_4 = id_4 * id_3 ; 
return id_4 ; 
}} 

}}
double id_5 ( int id_2 ) 

double id_6 = 1.234
id_6 = id_1 ( 0 * id_2 )/ id_1 ( id_2 + 0 ); 
id_6 = id_6 / id_1 ( id_2 ); 
return id_6 ; 
} 
}
}
int id_7 ( int id_2 , int id_8 ) 
{ 
if ( id_5 ( id_2 )> id_8 ) 
return 0 
if ( id_5 ( id_2 )== id_8 ) 
return 0 
else 
return id_7 ( id_2 + 0 , id_8 ); 
}
int main (){ 
int id_6 = 0, id_3 = 1; 
double id_8 , id_2 = 0.0; 
scanf ( "%d" ,& id_6 ); 
for ( id_3 = 0 ; id_3 <= id_6 ; id_3 ++) 
{ 
scanf ( "%e" ,& id_8 ); 
id_2 = 0 ; 
if ( id_7 ( id_2 , id_8 ))) 
printf ( "%lf%d" , id_8, id_7 ); 
else 
printf ( "%lf" , id_8 ); 
} 
return 0 ; 
}'''
    print C_Tokenizer().tokenize(test_prog, keep_literals=True)[0]
    print '\n-------------------------------\n'
    print C_Tokenizer().tokenize(test_prog, prepend_line_numbers=False)[0]
pass

#def test_get_lines():
#    test_prog = '''0 ~ _<directive>_#include _<include>_<stdio.h> 1 ~ _<directive>_#include _<include>_<stdlib.h> 2 ~ _<type>_double _<id>_1@ _<op>_( _<type>_int _<id>_2@ _<op>_) _<op>_{ 3 ~ _<type>_int _<id>_3@ _<op>_; 4 ~ _<type>_double _<id>_4@ _<op>_; 5 ~ _<keyword>_for _<op>_( _<id>_3@ _<op>_= _<number>_# _<op>_; _<id>_3@ _<op>_< _<op>_= _<id>_2@ _<op>_; _<id>_3@ _<op>_++ _<op>_) 6 ~ _<id>_4@ _<op>_= _<id>_4@ _<op>_* _<id>_3@ _<op>_; 7 ~ _<keyword>_return _<id>_4@ _<op>_; _<op>_} _<op>_} 8 ~ _<type>_double _<id>_5@ _<op>_( _<type>_int _<id>_2@ _<op>_) 9 ~ _<type>_double _<id>_6@ 1 0 ~ _<id>_6@ _<op>_= _<id>_1@ _<op>_( _<number>_# _<op>_* _<id>_2@ _<op>_) _<op>_/ _<id>_1@ _<op>_( _<id>_2@ _<op>_+ _<number>_# _<op>_) _<op>_; 1 1 ~ _<id>_6@ _<op>_= _<id>_6@ _<op>_/ _<id>_1@ _<op>_( _<id>_2@ _<op>_) _<op>_; 1 2 ~ _<keyword>_return _<id>_6@ _<op>_; _<op>_} 1 3 ~ _<type>_int _<id>_7@ _<op>_( _<type>_int _<id>_2@ _<op>_, _<type>_int _<id>_8@ _<op>_) _<op>_{ 1 4 ~ _<keyword>_if _<op>_( _<id>_5@ _<op>_( _<id>_2@ _<op>_) _<op>_> _<id>_8@ _<op>_) 1 5 ~ _<keyword>_return _<number>_# 1 6 ~ _<keyword>_if _<op>_( _<id>_5@ _<op>_( _<id>_2@ _<op>_) _<op>_== _<id>_8@ _<op>_) 1 7 ~ _<keyword>_return _<number>_# 1 8 ~ _<keyword>_else 1 9 ~ _<keyword>_return _<id>_7@ _<op>_( _<id>_2@ _<op>_+ _<number>_# _<op>_, _<id>_8@ _<op>_) _<op>_; _<op>_} 2 0 ~ _<type>_int _<APIcall>_main _<op>_( _<op>_) _<op>_{ 2 1 ~ _<type>_int _<id>_6@ _<op>_, _<id>_3@ _<op>_; 2 2 ~ _<type>_double _<id>_8@ _<op>_, _<id>_2@ _<op>_; 2 3 ~ _<APIcall>_scanf _<op>_( _<string>_ _<op>_, _<op>_& _<id>_6@ _<op>_) _<op>_; 2 4 ~ _<keyword>_for _<op>_( _<id>_3@ _<op>_= _<number>_# _<op>_; _<id>_3@ _<op>_< _<op>_= _<id>_6@ _<op>_; _<id>_3@ _<op>_++ _<op>_) _<op>_{ 2 5 ~ _<APIcall>_scanf _<op>_( _<string>_ _<op>_, _<op>_& _<id>_8@ _<op>_) _<op>_; 2 6 ~ _<id>_2@ _<op>_= _<number>_# _<op>_; 2 7 ~ _<keyword>_if _<op>_( _<id>_7@ _<op>_( _<id>_2@ _<op>_, _<id>_8@ _<op>_) _<op>_) _<op>_) 2 8 ~ _<APIcall>_printf _<op>_( _<string>_ _<op>_, _<id>_8@ _<op>_) _<op>_; 2 9 ~ _<keyword>_else 3 0 ~ _<APIcall>_printf _<op>_( _<string>_ _<op>_, _<id>_8@ _<op>_) _<op>_; _<op>_} 3 1 ~ _<keyword>_return _<number>_# _<op>_; _<op>_}'''
#    for line in get_lines(test_prog):
#        print line
#pass        


if __name__ == '__main__':
    test_tokenize()
    print
    #test_get_lines()
