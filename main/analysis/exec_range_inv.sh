mtype="ResNet18"

for hkey in 3 5 7 9
do
#    python3 get_response4.py $mtype $hkey 0 1 1 1 2
    python3 get_response4.py $mtype $hkey 0 0.5 1 1 2
    python3 alys_range_inv2.py $mtype $hkey 2 0.5
done