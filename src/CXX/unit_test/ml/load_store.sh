#!/bin/sh

in_file=load_store.$$.in
out_file=load_store.$$.out

cat >> "$in_file" << HERE
3 6
[1.1 0.1 0.1]
[0.2 1.2 0.2]
[0.3 0.3 1.3]
[1.4 1.4 0.4]
[1.5 0.5 1.5]
[1.6 1.6 1.6]
HERE

./load_store "$in_file" "$out_file"

diff "$in_file" "$out_file" || exit 1

rm "$in_file" "$out_file"
