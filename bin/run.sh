#!/bin/bash

# do args

RETURN_CODE=1
# echo $RETURN_CODE

until [ "$RETURN_CODE" -eq "0" ]; do
  $*
  RETURN_CODE=$?
  # echo $RETURN_CODE
  # echo
  # echo
  # echo
done
