from tdw.obi_data.fluids.fluid import FLUIDS
fluid = FLUIDS["water"]
for k in fluid.__dict__:
    print(f'{k}={fluid.__dict__[k]}')