#!/bin/bash 

awk -f word_frequencies.awk $1 | sort -rn | head -$2 | cut -d' ' -f2
