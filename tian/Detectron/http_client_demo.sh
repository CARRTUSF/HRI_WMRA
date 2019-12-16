#!/bin/bash
if [ $# -eq 0 ]
	# No args, defaults to 127.0.0.1, 665
	then 
		python3 http_client_demo.py

	# Args for ip, port
	else
		python3 http_client_demo.py $1 $2
fi
