#!/bin/bash
for filename in $(find dashboard/output_dashboard -name '*.json'); do
    jsonvice -i $filename -o $filename -p 4
done
