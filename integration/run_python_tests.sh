#!/usr/bin/env bash


# taken from https://stackoverflow.com/a/246128
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

PYTHON_EXECUTABLE=$1

# start by calling the tests that we need to work in order to trust the result of the other python tests
if $PYTHON_EXECUTABLE $SCRIPT_DIR/tests/lemmas/assert_false.py; then
    echo "assert_false.py failed"
    exit 1
fi

if !($PYTHON_EXECUTABLE $SCRIPT_DIR/tests/lemmas/assert_true.py); then
    echo "assert_true.py failed"
    exit 1
fi

exit_code=0

for file in $(find $SCRIPT_DIR/tests/ -maxdepth 1 -type f -name "*.py"); do
    result=$($PYTHON_EXECUTABLE $file)
    retval=$?
    if [ $retval -eq 0 ]; then
        echo $file "... PASSED!"
    else
        echo $file "... FAILED! (${result})"
        exit_code=1
    fi
done

exit $exit_code