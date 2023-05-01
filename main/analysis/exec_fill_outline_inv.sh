mtype="ResNet18"

for hkey in 3
do
    python3 get_response4.py $mtype $hkey 0 1 1 1 2
    python3 get_response4.py $mtype $hkey 1 1 1 1 2
    python3 alys_fill_outline_inv.py $mtype $hkey 2
done