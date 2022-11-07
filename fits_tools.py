from astropy.io import fits
from scipy import spatial
import numpy as np
import os

def append_fits_field(fits_array, names, arrays, formats, outfile = None):
	
	'''
	Adds a column to a fits array
	INPUTS:
		fits_array:	Data from fits array
		names:      (str) Array of names of the columns that wants to be added
		arrays:     Arrays of the columns that are going to be added
		formats:    (str) Array of the formats of each column
	OPTIONAL INPUT:
		outfile:    (str) name of the file where the data with added columns are going to be saved
	OUTPUT:
	    hdu         Fits bin table
	'''
	
	new_cols = [] 
	for j in range(len(names)):
		col = [fits.Column(name=names[j], format=formats[j], array=arrays[j])]
		new_cols += col
	
	orig_cols = fits_array.columns
	new_table = fits.BinTableHDU.from_columns(new_cols)
	columns   = orig_cols + new_table.columns
	hdu = fits.BinTableHDU.from_columns(columns, nrows=fits_array.shape[0])
	
	for colname in orig_cols.names:
		hdu.data[colname] = fits_array[colname]
	
	for colname in new_table.columns.names:
		hdu.data[colname] = new_table.data[colname]
	
	if outfile is not None:
		hdu.writeto(outfile, overwrite = True)  
	
    
	return hdu
	
def correl_quantity(name_cat,data,label,formats,
                    cols = ['RA','DEC'],
                    upper_limit=1./3600.,
                    out_file=False,
                    uncorrel_value=-999.):
						
						
	'''
	Correlates columns array with a fits catalog considering
	X,Y columns from the fits catalog with x_i,y_i
	from the column that is going to be correlated. It can
	take many columns to be correlated. The correlation is 
	done using kDTree, considering the min distance in the
	parameter space within an upper limit value.
	INPUTS:
		name_cat:	      (str) Name of the fits catalog
		data:             (float) List of the parameters considered to 
		                        do the correlation. The length is the 
		                        number of columns that will be correlated.
		                        [[x1,y1,col1],...,[xn,yn,coln]]		                   
		label:            (str) List of the labels of the added columns
		                        ['col1',...,'coln']
		formats:          (str) List of the formats of each column		               
	OPTIONAL INPUT:       
		cols:             (str) List of the labels of columns X,Y.
		                        If not provided it assumes that the
		                        labels are 'RA' and 'DEC'
		upper_limit       (float) Upper limit considered to correlate the catalogs
		out_file          (str) Name of the output fits catalog.
		                        If not provided it overwrites the input catalog
		uncorrel_value    Value given to the elements that do not correlate
		                  within the upper_limit. If it is not given it takes
		                  -999 or '-999' if the label of the column is a string
	'''						

	if not out_file:
		out_file = name_cat

	print('#################')
	print('MATCHING CATALOGS')
	
	cat = fits.open(name_cat)[1].data
	x   = cat[cols[0]]
	y   = cat[cols[1]]

	new_cols = []

	for col in range(len(data)):
		
		if 'A' in formats[col]: #A es 1 caracter
			correl = np.chararray(len(x),itemsize=int(label[col][1:]))
			correl.fill(str(uncorrel_value))			
		else:	
			correl = np.zeros(len(x))
			correl.fill(uncorrel_value)
				
		tree = spatial.cKDTree(np.array([x,y]).T)
		dist,ind = tree.query(np.array([data[col][0],data[col][1]]).T)
	
		mdist = dist < upper_limit
		
		correl[ind[mdist]] = data[col][2][mdist]
		new_cols.append(correl)
		
		
	hdu = append_fits_field(cat,
	                        label,
	                        new_cols,
	                        formats,
	                        outfile = out_file)
	return hdu
	

def correl_quantity_stilts(name_cat,data,label,formats,
                    cols = ['RA','DEC'],
                    upper_limit=1.,
                    out_file=False,
                    uncorrel_value=-999.):

	'''
	Correlates columns array with a fits catalog considering
	sky coordinates. It can
	take many columns to be correlated. The correlation is 
	done using stilts function tmatch2.
	INPUTS:
		name_cat:	      (str) Name of the fits catalog
		data:             (float) List of the parameters considered to 
		                        do the correlation. The length is the 
		                        number of columns that will be correlated.
		                        [[ra1,dec1,col1],...,[ran,decn,coln]]		                   
		label:            (str) List of the labels of the added columns
		                        ['col1',...,'coln']
		formats:          (str) List of the formats of each column		               
	OPTIONAL INPUT:       
		cols:             (str) List of the labels of columns of the
		                        sky coordinates of the fits catalog.
		                        If not provided it assumes that the
		                        labels are 'RA' and 'DEC'
		upper_limit       (float) Upper limit considered to correlate the catalogs
		                          If not provided it consideres 1 arcsecond.
		out_file          (str) Name of the output fits catalog.
		                        If not provided it overwrites the input catalog
		uncorrel_value    Value given to the elements that do not correlate
		                  within the upper_limit. If it is not given it takes
		                  -999 or '-999' if the label of the column is a string
	'''				


	if not out_file:
		out_file = name_cat

	print('#################')
	print('MATCHING CATALOGS')
	
	cat = fits.open(name_cat)[1].data
	header = fits.open(name_cat)[1].header
	x   = cat[cols[0]]
	y   = cat[cols[1]]

	try:
		cat.INDEX
		fits.writeto(out_file, data=cat, \
                     header=header, \
                     overwrite=True)
	except:
		cat = append_fits_field(cat,
	                            ['INDEX'],
	                            [np.arange(len(x))],
	                            ['J'],
	                            outfile = out_file)
	      
	cat = cat.data


	new_cols = []

	for col in range(len(data)):
		
		correl = np.zeros(len(x))
		correl.fill(uncorrel_value)

		tbhdu = fits.BinTableHDU.from_columns(
				[fits.Column(name='INDEX2', format='J', array=np.arange(len(data[col][0]))),
				fits.Column(name='RA', format='D', array=data[col][0]),
				fits.Column(name='DEC', format='D', array=data[col][1]),
				fits.Column(name=label[col], format=formats[0], array=data[col][2])])
						
		tbhdu.writeto('temp.fits',overwrite=True)

				
		command_str ='stilts.sh tmatch2 in1="'+out_file+\
			'" in2="temp.fits" matcher=sky values1="'+cols[0]+\
			' '+cols[1]+'" values2="RA DEC" params="'\
			+str(upper_limit)+'" out=temp2.fits'
		
		
		os.system(command_str)
		
		ind1 = fits.open('temp2.fits')[1].data.INDEX
		ind2 = fits.open('temp2.fits')[1].data.INDEX2
		
		correl[ind1] = data[col][2][ind2]
		new_cols.append(correl)
		
	os.system('rm temp*')
		
	hdu = append_fits_field(cat,
	                        label,
	                        new_cols,
	                        formats,
	                        outfile = out_file)

	return hdu
