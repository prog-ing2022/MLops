#!/bin/bash
apt install python3-pip
pip install wget
pip install scikit-learn
pip install pandas
pip install requests
python3 Load.py
python3 Standartization.py
python3 Training.py
python3 Result.py

