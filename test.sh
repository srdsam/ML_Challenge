#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd /DIR
#replace motion.tsv with "$1" for stdin or alternate .tsv file. 
python Challenge.py < motion.tsv
