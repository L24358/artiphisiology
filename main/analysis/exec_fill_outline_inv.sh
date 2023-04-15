mtype="VGG16"
hkey=16

python3 get_response3.py $mtype $hkey 0
python3 get_response3.py $mtype $hkey 1
python3 alys_fill_outline_inv.py $mtype $hkey