import time, numpy as np
from pycparser import parse_file
from pycparser.plyparser import ParseError

def get_ast(source, pid=''):
    name1 = int(time.time() * 10**6)
    name2 = np.random.random_integers(0, 1000)
    filename = '/tmp/%s_%d_%d.c' % (pid, name1, name2)
    with open(filename, 'w') as f:
        f.write(source)
    ast = parse_file(filename=filename, use_cpp=True,
                    cpp_path='gcc',
                    cpp_args=['-E', r'-I/home/rahul/git/pycparser/utils/fake_libc_include'])
    return ast

def show(_ast, pad=''):
    # print _ast.coord
    c = _ast.__class__.__name__
    attributes = []
    if _ast.attr_names:
        for attr_name in _ast.attr_names:
            if attr_name in ['name', 'op', 'declname', 'type', 'value']:
                attributes.append(getattr(_ast, attr_name))
            elif attr_name == 'names':
                names = getattr(_ast, attr_name)
                if type(names) == list:
                    # names = '<%s>' % ','.join(names)
                    names = ','.join(names)
                attributes.append(names)
    if attributes:
        c = '[%s:%s]' % (c, ','.join(attributes))
    else:
        c = '[%s]' % c
        
    if 'Constant:string' in c:
        c = c.replace(' ', '--SL-space--')
                
    print pad + c

    for name, child in _ast.children():
        show(child, pad+'  ')
    
    return

# def linearize_ast(_ast):
#     linearized_ast = ''
#     c = _ast.__class__.__name__

#     attributes = []
#     if _ast.attr_names:
#         for attr_name in _ast.attr_names:
#             if attr_name in ['name', 'declname', 'op', 'type', 'value']:
#                 attributes.append(getattr(_ast, attr_name))
#             elif attr_name == 'names':
#                 names = getattr(_ast, attr_name)
#                 if type(names) == list:
#                     names = ','.join(names)
#                 attributes.append(names)
#     if attributes:
#         c = '[%s:%s]' % (c, ','.join(attributes))
#     else:
#         c = '[%s]' % c

#     if 'Constant:string' in c:
#         c = c.replace(' ', '--SL-space--')

#     linearized_ast += c + ' ( '

#     for name, child in _ast.children():
#         linearized_ast += linearize_ast(child) + ' '

#     linearized_ast += ')'

#     return linearized_ast


def get_normalized_name(name, name_dict, normalize_names):
    to_ignore = ['printf', 'scanf', 'gets', 'getchar', 'puts', 'putchar', \
                 'main', 'malloc', 'calloc', 'free', 'strlen', 'strcmp', 'strcat', \
                 'floor', 'round', 'ceil', 'log', 'pow', 'sqrt', 'sin', 'cos', 'tan', 'exp', 'abs', \
                 'fgets', 'fputs', 'freopen', 'fclose', 'fprintf', 'fscanf', 'fflush', 'fgetc', 'fputc', 'feof', 'fseek']
    
    if name in to_ignore:
        return name
    elif not normalize_names:
        return '_<id>_{}@'.format(name)
    elif name not in name_dict:
        name_dict[name] = '_<id>_%d@' % len(name_dict)
    return name_dict[name]

def linearize_ast(_ast, normalize_names, name_dict={}):
    linearized_ast = ''
    c = _ast.__class__.__name__
    
    attributes = []
    if _ast.attr_names:
        for attr_name in _ast.attr_names:
            if attr_name in ['name', 'declname']:
                attributes.append(get_normalized_name(getattr(_ast, attr_name), name_dict, normalize_names))
            if attr_name in ['op', 'type', 'value']:
                attributes.append(getattr(_ast, attr_name))
            elif attr_name == 'names':
                names = getattr(_ast, attr_name)
                if type(names) == list:
                    names = ','.join(names)
                #     names = ','.join([get_normalized_name(name, name_dict, normalize_names) for name in names])
                # else:
                #     names = get_normalized_name(names, name_dict, normalize_names)
                attributes.append(names)
    if attributes:
        c = '%s:%s' % (c, ','.join(attributes))
    else:
        c = '%s' % c

    if 'Constant:string' in c:
        c = c.replace(' ', '--SL-space--')

    linearized_ast += c + ' ( '
    flag_typedefs_not_over = True
    for name, child in _ast.children():
        if flag_typedefs_not_over:
            if 'Typedef' not in child.__class__.__name__:
                flag_typedefs_not_over = False
                linearized_ast += linearize_ast(child, normalize_names, name_dict) + ' '
        else:
            linearized_ast += linearize_ast(child, normalize_names, name_dict) + ' '

    linearized_ast += ')'

    return linearized_ast
        
def get_linearized_ast(ast=None, source=None, normalize_names=False):
    assert source is not None or ast is not None
    if ast is None:
        try:
            ast = get_ast(source)
        except ParseError:
            return None
    return linearize_ast(_ast, normalize_names).replace('( )', '')

def tokenize_serialized_ast(serialized_ast):
    output = []
    for token in serialized_ast.split():
        token = token.strip()
        assert token == '(' or token == ')' or (token[0] == '[' and token[-1] == ']'), token
        output.append(token)
    return output

def get_root_class_name_with_attributes(_ast, name_dict, normalize_names):
    c = _ast.__class__.__name__
        
    attributes = []
    if _ast.attr_names:
        for attr_name in _ast.attr_names:
            if attr_name in ['name', 'declname']:
                attributes.append(get_normalized_name(getattr(_ast, attr_name), name_dict, normalize_names))
            if attr_name in ['op', 'type', 'value']:
                attributes.append(getattr(_ast, attr_name))
            elif attr_name == 'names':
                names = getattr(_ast, attr_name)
                if type(names) == list:
                    names = ','.join(names)
                #     names = ','.join([get_normalized_name(name, name_dict, normalize_names) for name in names])
                # else:
                #     names = get_normalized_name(names, name_dict, normalize_names)
                attributes.append(names)
    if attributes:
        c = '%s:%s' % (c, ','.join(attributes))
    else:
        c = '%s' % c

    if 'Constant:string' in c:
        c = c.replace(' ', '--SL-space--')

    return c

def subtree_to_list(_ast, name_dict, normalize_names, is_root=False, return_coord=True):
    subtree_list = [get_root_class_name_with_attributes(_ast, name_dict, normalize_names)]
    for name, child in _ast.children():
        if is_root:
            if 'Typedef' not in child.__class__.__name__:
                is_root = False
                subtree_list.append(get_root_class_name_with_attributes(child, name_dict, normalize_names))
        else:
            subtree_list.append(get_root_class_name_with_attributes(child, name_dict, normalize_names))
    
    if return_coord:
        return (subtree_list, str(_ast.coord).split('.c:')[-1] if _ast.coord is not None else '0:0')
    
    return subtree_list

def ast_to_subtree_lists(_ast, normalize_names, name_dict = {}):
    tree_list = [subtree_to_list(_ast, name_dict, normalize_names, is_root=True)]
    
    flag_typedefs_not_over = True
    for name, child in _ast.children():
        if flag_typedefs_not_over:
            if 'Typedef' not in child.__class__.__name__:
                flag_typedefs_not_over = False
                tree_list.extend(ast_to_subtree_lists(child, normalize_names, name_dict))
        else:
            tree_list.extend(ast_to_subtree_lists(child, normalize_names, name_dict))
    return tree_list

def get_subtree_list(ast=None, source=None, normalize_names=False, remove_leaf_subtrees=True):
    assert source is not None or ast is not None
    if ast is None:
        try:
            ast = get_ast(source)
        except ParseError:
            return None

    if remove_leaf_subtrees:
        return [(tl,coord) for tl,coord in ast_to_subtree_lists(ast, normalize_names) if len(tl)>1]
    else:
        return ast_to_subtree_lists(ast, normalize_names)


##################################################

test_prog = '''#include <stdio.h>
int main(){
   int n,k,flag=0;
   char s[n],a,e,i,o,u;
   scanf("%d\\n",&n);
   for(k=0;k<n;i++)
   {
       scanf("%c",&s[k]);
   }
   if(flag==1)
       printf("Special");
   else
       printf("Normal");
   
    return 0;
}'''

if __name__ == '__main__':
    # show(get_ast(test_prog))
    subtrees = source_to_subtree_lists(test_prog)
    for subtree, coord in subtrees:
        print coord, subtree
