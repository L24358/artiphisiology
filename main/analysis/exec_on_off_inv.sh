mtype="VGG16"

for hkey in 18
do
python3 get_response4.py $mtype $hkey 0 1 0 1 2
python3 alys_on_off_inv.py $mtype $hkey 2
done