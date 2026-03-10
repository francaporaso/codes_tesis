from astropy.table import Table

def sourcecat_load(name):
    """Can only load sourcecat as it is. For different Nback use sourcecat_load_nback"""

    S = Table.read(name, memmap=True, format='fits')
    return S