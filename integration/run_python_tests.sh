#!/usr/bin/env bash


# taken from https://stackoverflow.com/a/246128
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

PYTHON_EXECUTABLE=$1
GC_FREQUENCY=${GC_FREQUENCY:-100000}

# start by calling the tests that we need to work in order to trust the result of the other python tests
if timeout 10s $PYTHON_EXECUTABLE $SCRIPT_DIR/tests/lemmas/assert_false.py --gc-frequency $GC_FREQUENCY &> /dev/null; then
    echo "assert_false.py failed"
    exit 1
fi

if !(timeout 10s $PYTHON_EXECUTABLE $SCRIPT_DIR/tests/lemmas/assert_true.py --gc-frequency $GC_FREQUENCY &> /dev/null); then
    echo "assert_true.py failed"
    exit 1
fi

exit_code=0

for file in $(find $SCRIPT_DIR/tests/ -maxdepth 1 -type f -name "*.py"); do
    result=$(timeout 10s $PYTHON_EXECUTABLE $file --gc-frequency $GC_FREQUENCY &> /dev/null)
    retval=$?
    if [ $retval -eq 0 ]; then
        echo $file "... PASSED!"
    else
        echo $file "... FAILED! (${result})"
        exit_code=1
    fi
done

# An uncaught exception must exit non-zero, and the exit-time flush of the
# buffered sys.stdout must run before the traceback is printed, so with a
# redirected stdout the script's own output comes first.
file=$SCRIPT_DIR/tests/expected_failures/print_then_raise.py
output=$(timeout 10s $PYTHON_EXECUTABLE $file --gc-frequency $GC_FREQUENCY 2>&1)
if [ $? -eq 0 ]; then
    echo $file "... FAILED! (expected a non-zero exit code)"
    exit_code=1
elif [ "$(echo "$output" | head -n 1)" != "before-raise" ]; then
    echo $file "... FAILED! (script output must precede the traceback, got: ${output})"
    exit_code=1
else
    echo $file "... PASSED!"
fi

exit $exit_code
