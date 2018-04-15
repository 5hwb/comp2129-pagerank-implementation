#! /usr/bin/env bash
echo '                   Authorised by'
echo '||||||||||||||||||||||||||||||||||||||||||||||||||'
echo '||||||||||||||PERRY TESTING SOFTWARE||||||||||||||'
echo '||||||||||||||||||||||||||||||||||||||||||||||||||'
echo '              ----------------------'

# Go through all test files
for file in tests/*.in; do
    echo '======================================='
    echo 'Testing' $file '...'

    # Output results to (nameOfTestCase).res
    ((./pagerank 4) < $file) > "${file%.*}.res"

    # Do the diff operation with the (nameOfTestCase).out files,
    # showing any unexpected differences encountered
    diff "${file%.*}.res" "${file%.*}.out"

done

#./pagerank 4 < tests/sample.in
