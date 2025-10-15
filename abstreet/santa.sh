#!/bin/bash
echo See logs in output.txt
RUST_BACKTRACE=1 ./binaries/santa 1> output.txt 2>&1
