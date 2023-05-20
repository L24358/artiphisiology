"""
Test the functions that I've wrote.
"""
import spectools.basics as bcs
import spectools.models.models as mdl
from spectools.stimulus.wyeth import get_stimulus
from spectools.responses import get_response_wrapper

##### get_response_wrapper
hkeys = [0]
stim = get_stimulus(1)
fname = lambda hkey: ""
R1 = get_response_wrapper(hkeys, stim, fname, save=False, override=True)
print("The response shape is: ", R1[0].shape, ", which should be (#units, B).")

model = mdl.load_model("AN", hkeys, "cpu")
model(stim)
R2 = model.hidden_info[0][0]
Rc = bcs.get_center_response(R2)
print("The response shape is: ", Rc.shape, ", which should be (#units, B).")
print("The two response are identical (slightly different in the later floating point numbers).")