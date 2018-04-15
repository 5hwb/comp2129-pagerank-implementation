#! /usr/bin/env bash
echo '                   Authorised by'
echo '||||||||||||||||||||||||||||||||||||||||||||||||||'
echo '||||||||||||||PERRY TESTING SOFTWARE||||||||||||||'
echo '||||||||||||||||||||||||||||||||||||||||||||||||||'
echo '              ----------------------'

echo 'Deleting all .res files...'

# Go through all test files
for file in tests/*.res; do
    echo 'Goodbye,' $file
    rm $file
done
for file in tests_small/*.res; do
    echo 'Goodbye,' $file
    rm $file
done
