#!/bin/bash

edflow -n $1 -b ../tflow/configs/base.yaml ../tflow/configs/bemoji.yaml ${@:2}
