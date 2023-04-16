mtype="VGG16"

for hkey in 3 8 11 13 16 18
do
    python3 get_response4.py $mtype $hkey 0 1 1 1 2
    python3 get_response4.py $mtype $hkey 1 1 1 1 2
    python3 alys_fill_outline_inv.py $mtype $hkey 2
done