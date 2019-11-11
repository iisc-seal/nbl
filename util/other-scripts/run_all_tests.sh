#!/bin/bash
rm args_list_for_the_run_tests_on_a_problem_script.txt &> /dev/null;

cores=$1
prefix=$2
#for prefix in "original" "processed"
#do
  #echo "$prefix"
  for problem in $(ls "$prefix")
  do
    #bash run_tests_on_a_problem.sh "$prefix" "$problem" &
    echo "$prefix" "$problem" >> args_list_for_the_run_tests_on_a_problem_script.txt
  done
# done

cat args_list_for_the_run_tests_on_a_problem_script.txt | parallel -j $cores ./run_tests_on_a_problem.sh {}
