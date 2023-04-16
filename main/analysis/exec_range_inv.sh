mtype="VGG16"

for hkey in 18
do
    python3 get_response4.py $mtype $hkey 0 1 1 1 2
    python3 get_response4.py $mtype $hkey 0 2 1 1 2
    python3 alys_range_inv2.py $mtype $hkey 2
done