#!/bin/bash

if [[ $(uname) == "Darwin" ]]; then
    say -v Daniel -f extra/script.txt -o extra/script.aiff
    lame -m m extra/script.aiff extra/thisisatest.mp3
else
    echo "This script only works on Mac OS"
fi
