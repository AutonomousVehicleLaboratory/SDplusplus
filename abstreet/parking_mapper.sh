#!/bin/bash
echo See logs in output.txt
RUST_BACKTRACE=1 ./binaries/parking_mapper 1> output.txt 2>&1
