import numpy as np
from argparse import ArgumentParser
import toml

parser = ArgumentParser()
parser.add_argument('--dz', action='store', type=float, default=0.05)
parser.add_argument('--z_min', action='store', type=float, default=0.1)
parser.add_argument('--z_max', action='store', type=float, default=0.3)
parser.add_argument('--nRv', action='store', type=int, default=1)
parser.add_argument('--Rv_min', action='store', type=float, default=8.0)
parser.add_argument('--Rv_max', action='store', type=float, default=30.0)
parser.add_argument('--voidtype', action='store', type=str, default='mix', choices=['mix', 'S', 'R'])
parser.add_argument('--delta_min', action='store', type=float, default=-1.0)
parser.add_argument('--delta_max', action='store', type=float, default=10.0)
parser.add_argument('--RIN', action='store', type=float, default=0.01)
parser.add_argument('--ROUT', action='store', type=float, default=5.0)
parser.add_argument('--NDOTS', action='store', type=int, default=40)
parser.add_argument('--nback', action='store', type=str, default='FULL', choices=['FULL', 'DES', 'EUCLID'])
args = parser.parse_args()

# TODO: quiza mejor usar mediana... para eso debo cargar el catálogo...
rvrange = np.linspace(args.Rv_min, args.Rv_max, args.nRv+1)
radii = np.column_stack([rvrange[:-1], rvrange[1:]])

nz = np.round((args.z_max-args.z_min)/args.dz).astype(int)+1
zrange = np.linspace(args.z_min, args.z_max, nz, endpoint=True)
redshift = np.column_stack([zrange[:-1], zrange[1:]])

if args.voidtype==None:
    delta = [(args.delta_min, args.delta_max)]
elif args.voidtype=='mix':
    delta = [(-1.0,0.0), (0.0,10.0)]
elif args.voidtype=='S':
    delta = [(0.0,10.0)]
elif args.voidtype=='R':
    delta = [(-1.0,0.0)]

if args.nback=='DES':
    nback = 6.0
elif args.nback=='EUCLID':
    nback = 26.0
else:
    nback = 31.0

print(f'{radii=}')
print(f'{redshift=}')
print(f'{delta=}')


i = 0
for rv in radii:
    for zs in redshift:
        for d in delta:

            config = {
                'NCORES':16,
                'NK':100,
                'BIN':'lin',
                'prof':{
                    'NDOTS':args.NDOTS,
                    'RIN':args.RIN,
                    'ROUT':args.ROUT,
                },
                'void': {
                    'Rv_min':float(rv[0]),
                    'Rv_max':float(rv[1]),
                    'z_min':float(zs[0]),
                    'z_max':float(zs[1]),
                    'delta_min':float(d[0]),
                    'delta_max':float(d[1])
                },
                'lens':{
                    'name':'MICE/voids_MICE.dat',
                },
                'source':{
                    'name':'MICE_sources_HSN_withextra.fits',
                    'nback':nback
                }
            }

            with open(f'config_{i}.toml', 'w') as f:
                toml.dump(config, f)
            i+=1
