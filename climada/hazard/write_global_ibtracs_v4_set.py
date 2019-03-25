#!/usr/bin/env python
# -*- coding: utf-8 -*- 

### read and re-write ibtracs data to be used in windfield-routine
### runs with python27
#################################################################
'''
script written by Tobias Geiger, Nov 2018
Potsdam Institute for Climate Impact Research
for questions and comments please contact:
geiger@pik-potsdam.de
'''
#################################################################


from __future__ import division
from pandas import DataFrame
import pandas as pd
import numpy as np
from datetime import datetime
from netCDF4 import Dataset

#################### FUNCTIONS ###############################

def basin_choice(basin):
	### US East coast
	if basin=='US':
		#pocibasin=1015 # peduzzi
		pocibasin=1010 # ibtracs 1980-2013
		provider='usa'
		provider_alt=''
		hemisphere='N'

	### NA - North Atlantic
	# include east pacific?
	if basin=='NA':
		#pocibasin=1015 # peduzzi
		pocibasin=1010 # ibtracs 1980-2013
		provider='usa'
		provider_alt=''
		hemisphere='N'

	### SA - South Atlantic
	if basin=='SA':
		#pocibasin=1015 # peduzzi
		pocibasin=1010 # ibtracs 1980-2013
		provider='usa'
		provider_alt=''
		hemisphere='S'

	### NI - North Indian
	if basin=='NI':
		#pocibasin=1008 # peduzzi
		pocibasin=1005 # ibtracs 1980-2013
		provider='usa'
		provider_alt='newdelhi'
		hemisphere='N'

	### SI - South Indian
	if basin=='SI':
		#pocibasin=1008 # peduzzi
		pocibasin=1005 # ibtracs 1980-2013
		provider='usa'
		provider_alt='bom'
		hemisphere='S'

	### SP - South Pacific
	if basin=='SP':
		#pocibasin=1005 # peduzzi
		pocibasin=1004 # ibtracs 1980-2013
		provider='usa'
		provider_alt=''
		hemisphere='S'

	### WP - West Pacific
	if basin=='WP':
		#pocibasin=1005 # peduzzi
		pocibasin=1005 # ibtracs 1980-2013
		provider='usa'
		provider_alt='tokyo'
		provider_alt2='cma'
		hemisphere='N'

	### EP - East Pacific
	if basin=='EP':
		#pocibasin=1010 # peduzzi
		pocibasin=1010 # ibtracs 1980-2013 1010
		provider='usa'
		provider_alt=''
		hemisphere='N'
		
	if (basin != 'WP'):
		provider_alt2=''
	return pocibasin, hemisphere,provider,provider_alt, provider_alt2


##### create custom Nan error

class NANError(Exception):
	pass

def set_sing(lons):
	if np.all(np.abs(lons) > 5000):
		raise NANError('alternative provider chosen')

def set_mult(lons, ps, vs):
	if np.all(np.abs(lons) > 5000):
		raise NANError('alternative provider chosen')
	elif (np.any(np.abs(lons) > 5000)) & ((np.all(np.abs(ps) > 5000)) & (np.all(np.abs(vs) > 5000))):
		raise NANError('alternative provider chosen')

########################## major programme #####################################

#### advanced windfield input of all ibtracs

### parameters

# local
fund_path1='Documents/PIK/data'
fund_path2='Documents/PIK/worldwide_storm'
# cluster
#fund_path1='data/expact/tobias/cluster/data'

# track time interval
intfact=3. 

minyear=1950    # default 1950
maxyear=2017	# defualt 2017
maxprintyear=int(maxyear)
maxyear=maxyear+1
mayxear=np.int(maxyear)
print(minyear, '-', maxyear)

# load ibtracs v4 data as a single nc file
# last 3 years
#path=fund_path2+'/all_ibtracs/ibtracs/ibtracs_v40_beta/IBTrACS.last3years.hotel1.nc'
# all files
#path=fund_path2+'/all_ibtracs/ibtracs/ibtracs_v40_beta/IBTrACS.ALL.hotel1.nc'
path='/Users/aznarsig/Desktop/IBTrACS.ALL.v04r00.nc'

ncfile=Dataset(path)
# read all storm ids
stnames=ncfile.variables['sid'][:]
stnames=[''.join(list(el)) for el in stnames]
stnames=zip(np.arange(len(stnames)),stnames)
#print stnames

counter_df=pd.DataFrame()  # keep track of successful saved events per year

# loop over years
for ino, iyea in enumerate(range(minyear,maxyear)):
	yea=str(iyea) 
	event_counter=0
	
	### identify all storms in specific year
	isot_yd_ls=[]; isot_m_ls=[]; isot_d_ls=[]; ind_ls=[]; stm_ls=[]; bas_ls=[]; ind_full_ls=[]

	stnames_yr=[ (el1,el2) for el1,el2 in stnames if el2[:4]== yea]
	#print stnames_yr #236N23285 katrina  # 296N14283 sandy #  293N13260 # 258N16300 jeanne 2004

	for stno,stname in stnames_yr:
		print(stname)
		
		# read genesis basin as first entree of basin values, genesis_basin not included in ibtracs v40
		basin=ncfile.variables['basin'][stno,0,:].data[0] +ncfile.variables['basin'][stno,0,:].data[1]
	
		# read time stamp from data
		isot=ncfile.variables['iso_time'][stno,:,:]
		valdim=isot.mask[isot.mask==False].shape[0]//isot.shape[1]
		isot=isot.data[:valdim]
		isot_len=isot.shape[0]
		fmt = '%Y-%m-%d %H:%M:%S'
		iso_ls=[]
		for ent in range(isot_len): 
			isot_i = datetime.strptime(''.join(isot[ent]), fmt)
			iso_ls.append(isot_i.strftime('%Y%m%d%H'))
		isotime=np.array(iso_ls).astype('int')
		ind_3h = np.where(isotime % 100 % 3 == 0)[0]  # take only all 3 hour time steps, neglect intermediate reported values
		isotime=isotime[ind_3h]
		#print isotime

		if stname[7]=='N':
			placeholder='0'
		else:
			placeholder='1'
		stmid=stname
		stmid=int(stname[:7]+placeholder+stname[8:])
		#print stmid
		
		# preparation for grid
		pocibasin, hemisphere, provider, provider_alt,provider_alt2 =basin_choice(basin=basin)
		
		print(basin, provider, provider_alt, provider_alt2)
		
		try:
			provid=provider
			latsall=ncfile.variables[provider+'_lat'][stno,:].data[:valdim][ind_3h]
			lonsall=ncfile.variables[provider+'_lon'][stno,:].data[:valdim][ind_3h]
			# 1-min sustained wind
			windall=ncfile.variables[provider+'_wind'][stno,:].data[:valdim][ind_3h]
			presall=ncfile.variables[provider+'_pres'][stno,:].data[:valdim][ind_3h]
			#print presall
			set_mult(lonsall,presall,windall)
		except(NANError) as e:
			#print e.message
			if provider_alt != '':
				try:
					print('provider {0} used for basin {1}'.format(provider_alt, basin))
					provid=provider_alt
					latsall=ncfile.variables[provider_alt+'_lat'][stno,:].data[:valdim][ind_3h]
					lonsall=ncfile.variables[provider_alt+'_lon'][stno,:].data[:valdim][ind_3h]
					# 1-min sustained wind
					windall=ncfile.variables[provider_alt+'_wind'][stno,:].data[:valdim][ind_3h]
					presall=ncfile.variables[provider_alt+'_pres'][stno,:].data[:valdim][ind_3h]
					#print presall
					set_mult(lonsall,presall,windall)
				except(NANError) as e:
					#print e.message
					if provider_alt2 != '':
						try:
							print('provider {0} used for basin {1}'.format(provider_alt2, basin))
							provid=provider_alt2
							latsall=ncfile.variables[provider_alt2+'_lat'][stno,:].data[:valdim][ind_3h]
							lonsall=ncfile.variables[provider_alt2+'_lon'][stno,:].data[:valdim][ind_3h]
							# 1-min sustained wind
							windall=ncfile.variables[provider_alt2+'_wind'][stno,:].data[:valdim][ind_3h]
							presall=ncfile.variables[provider_alt2+'_pres'][stno,:].data[:valdim][ind_3h]
							#print presall
							set_mult(lonsall,presall,windall)
						except(NANError):
							print('data WMO used for basin {1}'.format(provider_alt2, basin))
							latsall=ncfile.variables['lat'][stno,:].data[:valdim][ind_3h]
							lonsall=ncfile.variables['lon'][stno,:].data[:valdim][ind_3h]
							# X min-sustained winds as provided by WMO chosen provider 
							# not very crucial as few storms and pressure is used preferably	
							# hurdat/atcf = North Atlantic - U.S. Miami (NOAA NHC) - 1-minute winds
							# tokyo = RSMC Tokyo (JMA) - 10-minute
							# newdelhi = RSMC New Delhi (IMD) - 3-minute
							# reunion = RSMC La Reunion (MFLR) - 10 minute
							# bom = Australian TCWCs (TCWC Perth, Darwin, Brisbane) - 10-minute
							# nadi = RSMC Nadi (FMS) - 10 minute
							# wellington = TCWC Wellington (NZMS) - 10-minute
							windall=ncfile.variables['wmo_wind'][stno,:].data[:valdim][ind_3h]
							presall=ncfile.variables['wmo_pres'][stno,:].data[:valdim][ind_3h]
					else:
						print('wmo data used for basin {1}'.format(provider_alt2, basin))
						provid='ibtracs and WMO'
						latsall=ncfile.variables['lat'][stno,:].data[:valdim][ind_3h]
						lonsall=ncfile.variables['lon'][stno,:].data[:valdim][ind_3h]
						# X min-sustained winds as provided by WMO chosen provider 
						# not very crucial as few storms and pressure is used preferably
						windall=ncfile.variables['wmo_wind'][stno,:].data[:valdim][ind_3h]
						presall=ncfile.variables['wmo_pres'][stno,:].data[:valdim][ind_3h]
			else:
				print('wmo data used for basin {1}'.format(provider_alt, basin))
				provid='ibtracs and WMO'
				latsall=ncfile.variables['lat'][stno,:].data[:valdim][ind_3h]
				lonsall=ncfile.variables['lon'][stno,:].data[:valdim][ind_3h]
				# X min-sustained winds as provided by WMO chosen provider 
				# not very crucial as few storms and pressure is used preferably
				windall=ncfile.variables['wmo_wind'][stno,:].data[:valdim][ind_3h]
				presall=ncfile.variables['wmo_pres'][stno,:].data[:valdim][ind_3h]
		# drop indices where lat/lon are Nan (1e36)
		keepind=np.where(np.abs(lonsall) <  1000)[0]
		#print keepind
		lonsall=(lonsall*10).astype('int')
		latsall=(latsall*10).astype('int')
		try:
			rmaxall=ncfile.variables[provider+'_rmw'][stno,:].data[:valdim][ind_3h]
			poci=ncfile.variables[provider+'_poci'][stno,:].data[:valdim][ind_3h]
			set_sing(rmaxall)
		except (NANError):
			if (provider_alt != '') & (basin != 'WP'):
				print('alternative provider needed')
				try:
					if basin != 'NI':
						rmaxall=ncfile.variables[provider_alt+'_rmw'][stno,:].data[:valdim][ind_3h]
					poci=ncfile.variables[provider_alt+'_poci'][stno,:].data[:valdim][ind_3h]
					set_sing(poci)
				except (NANError):
					if provider_alt2 != '':
						print('yet another alternative provider needed')
						rmaxall=ncfile.variables[provider_alt2+'_rmw'][stno,:].data[:valdim][ind_3h]
						poci=ncfile.variables[provider_alt2+'_poci'][stno,:].data[:valdim][ind_3h]
					else:
						rmaxall=np.zeros(len(latsall))
						poci=np.zeros(len(latsall))
			else:
				rmaxall=np.zeros(len(latsall))
				poci=np.zeros(len(latsall))
		
#		print 'provider', provid
#		print 'lats'
#		print latsall
#		print 'lons'
#		print lonsall
#		print 'pres'
#		print presall
#		print 'wind'
#		print windall
#		print 'poci'
#		print poci
#		print 'rmaxall'
#		print rmaxall
	
		# drop indices where lat/lon are Nan (1e36)
		lonsall=lonsall[keepind]
		latsall=latsall[keepind]
		windall=windall[keepind]
		presall=presall[keepind]
		rmaxall=rmaxall[keepind]
		poci=poci[keepind]
		isotime=isotime[keepind]
		
		if len(lonsall) ==0:
			print(stname, 'dropped, no data in lonsall')
			continue
		# check for identical length and zero values
		if len(latsall)==len(lonsall)==len(presall)==len(windall):
			pass
		else:
			print('unequal length of input')
			continue
		
		# define rmaxmin such that rmax is not interpolated between rmaxmin und zero
		rmaxcut=rmaxall[rmaxall > 0]
		if len(rmaxcut) >0: 
			rmaxmin=np.min(rmaxcut)
		else:
			rmaxmin=0
		#print rmaxmin
			
#		print 'lats'
#		print latsall
#		print 'lons'
#		print lonsall
#		print 'pres'
#		print presall
#		print 'wind'
#		print windall
#		print 'poci'
#		print poci

		# windmap information
		res=0.1
		# original data used flag
		marker=True

		df_ibtracs=pd.DataFrame({'ibtracsID':[stname]*len(lonsall),'res':[res]*len(lonsall),'cgps_lon':lonsall/10.,'cgps_lat':latsall/10.,'penv':poci,'pcen':presall,'rmax':rmaxall,'vmax':windall,'tint':[intfact]*len(lonsall),   'original_data':[marker]*len(lonsall),'data_provider':[provid]*len(lonsall),'isotime':isotime, 'gen_basin':[basin]*len(lonsall)})
		
		### model effects
		# 'H80' requires vmax and penv and pcen as input
		# 'H08' requires only pressure as input
		# 'H08_v' requires only vmax as input
		# 'H10' so far same as 'H08' but without gradient winds

		# decide whether to use 'H08_v' if only v is given, never, or always
		# delete missing points or full events where no pressure and vmax values are available
		df_ibtracs['model']=np.nan
		df_ibtracs.loc[df_ibtracs.index[(df_ibtracs['pcen'] > 850 ) | (df_ibtracs['vmax'] > 5)],'model']='H08'
		del_rows = df_ibtracs.loc[df_ibtracs['model'].isna()].shape[0]
		ib_shape = df_ibtracs.shape[0]
		df_ibtracs = df_ibtracs.loc[~df_ibtracs['model'].isna()].reset_index(drop=True)
		if del_rows == ib_shape:
			print('no pressure or vmax value given, continue with next storm!')
			continue
		elif del_rows > 0:
			print('no pressure or vmax value given, deleted ', del_rows, ' rows!')
			#print df_ibtracs

		# work on poci and rmax
		# fill missing environmental pressure values
		df_ibtracs.loc[df_ibtracs.index[df_ibtracs['penv'] < 0],'penv'] = np.nan
		df_ibtracs['penv']=df_ibtracs['penv'].ffill(limit=4).bfill(limit=4)
		# ensure that central pressure is always lower than environmental pressure
		# CAUTION: One might additionally need to ensure that the previous central pressure point 'prepcen' is also lower or equal to penv
		df_ibtracs.loc[df_ibtracs.index[(df_ibtracs['penv'] < 990 )] ,'penv'] = pocibasin
		df_ibtracs.loc[df_ibtracs.index[ (df_ibtracs['penv'] < df_ibtracs['pcen']) ],'penv'] = df_ibtracs.loc[df_ibtracs.index[ (df_ibtracs['penv'] < df_ibtracs['pcen']) ],'pcen']
		# add previous pressure point to determine pressure gradient
		df_ibtracs['prepcen'] = df_ibtracs['pcen'].shift(1).bfill(limit=1)
		# work with missing rmax values
		df_ibtracs.loc[df_ibtracs.index[df_ibtracs['rmax'] < 0],'rmax'] = np.nan
		df_ibtracs['rmax']=df_ibtracs['rmax'].ffill(limit=1).bfill(limit=1).fillna(0)
		# set next lat/lon points to determine translational windspeed
		df_ibtracs['ngps_lon'] = df_ibtracs['cgps_lon'].shift(-1).ffill(limit=1)
		df_ibtracs['ngps_lat'] = df_ibtracs['cgps_lat'].shift(-1).ffill(limit=1)
		# msize : raster size to model windfield on
		df_ibtracs['msize']= 201
		df_ibtracs.loc[df_ibtracs.index[np.abs(df_ibtracs['cgps_lat']) < 60.0 ],'msize']= 101

		# save output
		df_ibtracs.to_csv(fund_path2 + '/all_ibtracs/windfield_input/global_v4/ibtracs_global_intp-None_{0}.csv'.format(stname),index=None)
		event_counter= event_counter+1
		print(stname, " completed!")

	temp_df=pd.DataFrame({'year':[iyea],'freq':event_counter})
	counter_df=counter_df.append(temp_df)
counter_df.to_csv(fund_path2 + '/all_ibtracs/windfield_input/global_v4/annual_event_frequency_{0}-{1}.csv'.format(minyear,maxprintyear),index=None)

