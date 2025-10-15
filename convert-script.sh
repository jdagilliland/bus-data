#!/bin/bash

sed 's/ ([^)]*)//g;s/:/,/;s/, /,/g;s|?||g'
