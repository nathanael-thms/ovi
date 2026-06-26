#!/bin/bash
set -euo pipefail

while [[ $# -gt 0 ]]; do
  case $1 in
    --is-ovi-install)
      echo "True"
      exit 0
      shift
      ;;
    *)
      shift
      ;;
  esac
done

echo "Welcome to ovi"