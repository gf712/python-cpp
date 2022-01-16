#!/usr/bin/env bash


# taken from https://stackoverflow.com/a/246128
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

PYTHON_EXECUTABLE=$1

for file in $(find $SCRIPT_DIR/llvm/ -maxdepth 1 -type f -name "*.py"); do
    result=$($PYTHON_EXECUTABLE $file --use-llvm)
    retval=$?
    if [ $retval -eq 0 ]; then
        echo $file "... PASSED!"
    else
        echo $file "... FAILED! (${result})"
        exit_code=1
    fi
done

exit $exit_code