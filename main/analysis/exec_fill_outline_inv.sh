mtype="AN"

for hkey in 3 8 10 15 18
do
    python3 get_response4.py $mtype $hkey 0 1 1 1 2
    python3 get_response4.py $mtype $hkey 1 1 1 1 2
    python3 alys_fill_outline_inv.py $mtype $hkey
done