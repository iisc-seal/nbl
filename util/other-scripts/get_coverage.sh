#!/bin/bash

# prefix='original'
# prefix='processed'

# In order to use parallel, we send both inputs in a single word
#input="$1"
#set $input

#prefix="$1"
#problem="$2"
#echo "source argument: $prefix"
#echo "problem: $problem"
root_dir=$(pwd)
root_dir="$root_dir/"
echo "root_dir: $root_dir"

source_dir="$1"
echo "source_dir: $source_dir"
tests_dir="../../tests"

# log_file="$root_dir/$prefix/$problem/coverage-log-file.txt"
echo $(date)
echo "Starting script"

compiled_binary="a.out"
captured_test_output="$prefix-$problem-test.out"
spaceless_actualtestout="$prefix-$problem-spaceless-test-out"

for problem in $(ls "$source_dir")
do
echo "-------- Starting problem: $problem"
cd "$root_dir/$source_dir/$problem/"
find . -name "*.c" | while read cfile; do rm "a.out" &> /dev/null; gcc "$cfile" -lm -w -fprofile-arcs -ftest-coverage -o "a.out" &> /dev/null;
echo "----- Starting program: $cfile"
if [ ! -f "a.out" ];
then
    echo "-- ERROR: $cfile did-not-compile"
else
    echo "-- COMPILED: $cfile"
    curr_dir=$(pwd)
    echo "current directory: $curr_dir";
    find "$tests_dir/$problem" -name "IN_*.txt" | while read testin; do
    # echo "--- running $testin on $cfile";
    # rm "$captured_test_output" &> /dev/null;

    # closing below statement within parantheses opens a new sub-shell and ulimit is applied only to that sub-shell.
    (ulimit -t 3; ./a.out < $testin >> /dev/null;)

    cfile_name=${cfile/"./"/""}
    cfile_name=${cfile_name/".c"/""}
    gcov "$cfile"
    mv "$cfile.gcov" "$testin-$cfile_name.gcov"
    rm "$cfile_name.gcda"

done;
fi
done;
done;

# test with no loops

# cd "$source_dir/$problem/"
# rm "a.out" &> /dev/null;
# cfile="41867008.c"
# testin="../../tests/$problem/IN_20020.txt"
# cat "$testin";
# echo "";
# gcc "$cfile" -lm -w -fprofile-arcs -ftest-coverage -o "a.out" &> /dev/null;
# if [ ! -f "a.out" ];
#     then
#         echo "-- ERROR: $cfile did-not-compile";
#     else
#         echo "-- COMPILED: $cfile";
#         (ulimit -t 5; LD_PRELOAD=/home/rahul/git/EasySandbox/EasySandbox.so "./a.out" < $testin)
#         gcov "$cfile"
#         cat "$cfile.gcov"
# fi

