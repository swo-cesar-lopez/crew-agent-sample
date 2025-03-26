#!/bin/bash
set -e

echo "Verificando el entorno..."
python check_environment.py

echo "Iniciando la aplicación..."
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1