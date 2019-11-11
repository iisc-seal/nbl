#!/bin/bash

# prefix='original'
# prefix='processed'

# In order to use parallel, we send both inputs in a single word
input="$1"
set $input

prefix="$1"
problem="$2"
echo "source argument: $prefix"
echo "problem: $problem"
root_dir=$(pwd)
root_dir="$root_dir/"
echo "root_dir: $root_dir"

source_dir="$prefix"
echo "source_dir: $source_dir"
tests_dir=tests

log_file="$root_dir/$prefix-$problem-log-file.txt"
echo $(date) >> "$log_file";
echo "Starting script" >> "$log_file";

compiled_binary="$prefix-$problem-a.out"
captured_test_output="$prefix-$problem-test.out"
spaceless_actualtestout="$prefix-$problem-spaceless-test-out"

#for problem in $(ls "$source_dir")
#do
    echo "-------- Starting problem: $problem" >> "$log_file";
    find "$source_dir/$problem" -name "*.c" | while read cfile; do rm "$compiled_binary" &> /dev/null; gcc "$cfile" -lm -w -o "$compiled_binary" &> /dev/null;
    echo "----- Starting program: $cfile" >> "$log_file";
    if [ ! -f "$compiled_binary" ]; 
    then 
        echo "-- ERROR: $cfile did-not-compile" >> "$log_file";
    else
        find "$tests_dir/$problem" -name "IN_*.txt" | while read testin; do 
        # echo "--- running $testin on $cfile"; 
        rm "$captured_test_output" &> /dev/null; 
        # rm "$spaceless_actualtestout" &> /dev/null;
        # closing below statement within parantheses opens a new sub-shell and ulimit is applied only to that sub-shell.
        (ulimit -t 1; LD_PRELOAD=/home/rahul/git/EasySandbox/EasySandbox.so ./"$compiled_binary" < $testin | grep -v "entering SECCOMP mode" &> "$captured_test_output";)
        #timeout -k 1s 1s ./"$compiled_binary" < $testin &> "$captured_test_output";
        if [ -e "$captured_test_output" ]; then
            actualtestout=${testin/IN/OUT}
    #            echo programout >> "$log_file";
    #            cat "$captured_test_output" >> "$log_file";
    #            echo actualtestout >> "$log_file";
    #            cat $actualtestout >> "$log_file";

            tr -d '\n' < $actualtestout | tr -d ' ' > "$spaceless_actualtestout"
            echo -e '\n' >> "$spaceless_actualtestout"
    #            cat "$spaceless_actualtestout" >> "$log_file";
        
            if diff -qwB "$actualtestout" "$captured_test_output" &> /dev/null
            then
                # if no difference than takes true branch (based on return value)
                echo "$cfile ; passed; $testin" >> "$log_file"; echo "passed-on-test $testin";
            elif diff -qB "$spaceless_actualtestout" "$captured_test_output" &> /dev/null 
            then
                # or no difference with new-line removed should-be-output (just a formatting error)
                echo "$cfile ;passed; $testin" >> "$log_file"; echo "passed $testin";
            else
                echo "$cfile ;failed; $testin" >> "$log_file"; echo "failed $testin";
            fi
        else echo "$cfile ;timed-out; $testin" >> "$log_file"; echo "timed-out $testin";
        fi
    done;
    fi
    done;
#done;
