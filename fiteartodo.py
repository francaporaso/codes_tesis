from fit_void_leastsq import *

samples = ['Rv_6-9','Rv_9-12', 'Rv_12-15', 'Rv_15-18', 'Rv_18-50']
name    = ['smallz_6-9','smallz_9-12', 'smallz_12-15', 'smallz_15-18', 'smallz_18-50']
nameS   = ['sz_S_6-9','sz_S_9-12', 'sz_S_12-15', 'sz_S_15-18', 'sz_S_18-50']
nameR   = ['sz_R_6-9','sz_R_9-12', 'sz_R_12-15', 'sz_R_15-18', 'sz_R_18-50']

nameh    = ['highz_6-9','highz_9-12', 'highz_12-15', 'highz_15-18', 'highz_18-50']
nameSh   = ['hz_S_6-9','hz_S_9-12', 'hz_S_12-15', 'hz_S_15-18', 'hz_S_18-50']
nameRh   = ['hz_R_6-9','hz_R_9-12', 'hz_R_12-15', 'hz_R_15-18', 'hz_R_18-50']

for x in samples:
    for y in name:
        fitear(x,y)
    for z in nameS:
        fitear(x,z)
    for m in nameR:
        fitear(x,m)

    for l in nameh:
        fitear(x,l)
    for s in nameSh:
        fitear(x,s)
    for q in nameRh:
        fitear(x,q)