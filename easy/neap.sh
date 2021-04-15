rm -rf dataset/DBP15K/processed/
rm -rf dataset/SRPRS/processed/
# SRPRS
python -u neap.py --pair en_fr
python -u neap.py --pair en_de
# DBP15k
python -u neap.py --pair fr_en
python -u neap.py --pair zh_en
python -u neap.py --pair ja_en