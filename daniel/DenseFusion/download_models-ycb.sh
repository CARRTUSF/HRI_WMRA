#!/bin/bash
echo 'Downloading the trained checkpoints...'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bQ9H-fyZplQoNt1qRwdIUX5_3_1pj6US' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bQ9H-fyZplQoNt1qRwdIUX5_3_1pj6US" -O trained_checkpoints.zip && rm -rf /tmp/cookies.txt \
&& unzip trained_checkpoints.zip -x "__MACOSX/*" "*.DS_Store" "*.gitignore" -d trained_checkpoints \
&& mv trained_checkpoints/trained*/ycb trained_checkpoints/ycb \
&& mv trained_checkpoints/trained*/linemod trained_checkpoints/linemod \
&& rm -r trained_checkpoints/trained*/ \
&& rm trained_checkpoints.zip
