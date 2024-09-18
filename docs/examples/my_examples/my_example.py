
''' PlasticParcels example for the Italian coast '''

import pdb
import os
from datetime import datetime, timedelta
import xarray as xr
import numpy as np

import plasticparcels as pp
from parcels import FieldSet,AdvectionRK4

output_file = '/mnt/c/Users/martino.rial/Escritorio/resultados_parcels/my_example.zarr'
settings_file = '/mnt/c/Users/martino.rial/Escritorio/git/plasticparcels/docs/examples/my_examples/my_example_settings.json'


def create_fieldset_from_netcdf(settings):
    nc_files_path = settings['ocean']['directory']
    ocean_mesh = settings['ocean']['ocean_mesh']
    nc_files = [os.path.join(nc_files_path,arquivo) for arquivo in os.listdir(nc_files_path) if arquivo.endswith('.nc4')]
    nc_files = sorted(nc_files)
    variables = {'U': 'u',
                 'V': 'v'}
                 #'absolute_salinity': 'salt',
                 #'conservative_temperature': 'temp'}
    # dimensions = {'time': 'time',
    #               'lon': 'lon',
    #               'lat': 'lat'}
    dimensions = {"U": {"lon": "lon",
                        "lat": "lat",
                        "time": "time"},
                  "V": {"lon": "lon",
                        "lat": "lat",
                        "time": "time"}}
    # create  (lat,lon) netcdf file for ocean mesh 
    filenames = {'U': {'lon': ocean_mesh, 'lat': ocean_mesh, 'data': nc_files},
                 'V': {'lon': ocean_mesh, 'lat': ocean_mesh, 'data': nc_files}}
    # Add grid type
    # This would allow to use a certain grid type --> CHECK IF A-GRID OR C-GRID
    #fieldset = FieldSet.from_netcdf(nc_files, variables, dimensions, allow_time_extrapolation=True)
    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, allow_time_extrapolation=True)

    fieldset.add_constant('use_mixing', settings['use_mixing'])
    fieldset.add_constant('use_biofouling', settings['use_biofouling'])
    fieldset.add_constant('use_stokes', settings['use_stokes'])
    fieldset.add_constant('use_wind', settings['use_wind'])
    fieldset.add_constant('G', 9.81)  # Gravitational constant [m s-2]
    fieldset.add_constant('use_3D', settings['use_3D'])
    return fieldset

def caudal_reader(settings):
    ulla = settings['sources']['river']
    total_days = settings['simulation']['runtime']
    total_days = total_days.days
    with open(ulla, 'r') as ulla_file:
        ulla_dat = ulla_file.read()
    lines = ulla_dat.split('\n')
    times = []   # tempos
    ratios = []   # ratios
    ler_datos = False
    for line in lines:
        if "<BeginTimeSerie>" in line:
            ler_datos = True
        elif "<EndTimeSerie>" in line:
            ler_datos = False
        elif ler_datos:
            elements = line.split()
            if len(elements) == 2:
                times.append(int(elements[0]))
                ratios.append(float(elements[1]))
    arr_ratios = np.array(ratios)
    arr_ratios = arr_ratios[:total_days]

    intervals = 1/arr_ratios
    npart = int(round(sum(arr_ratios*3600*24)))

    emission_times = np.zeros(npart)
    count=0; j=0; day=1
    for i in range(npart):
        emission_times[i] = count
        count = count + intervals[j]
        if int(count) > day*24*3600:
            j = j+1
            day = day+1
            if j == len(intervals):
                break

    return emission_times, npart

def particle_dictionary(settings):
    emission_times, npart = caudal_reader(settings)
    emission_times = emission_times.tolist()
    lats = npart*[settings['sources']['latitude']]
    lons = npart*[settings['sources']['longitude']]
    pdict = {"emission_times": emission_times,
             "lats": lats,
             "lons": lons}
    return pdict

# Load the model settings
settings = pp.utils.load_settings(settings_file)

# Create the simulation settings
settings['simulation'] = {
    'startdate': datetime.strptime('2024-09-01-00:00:00', '%Y-%m-%d-%H:%M:%S'), # Start date of simulation
    'runtime': timedelta(days=2),        # Runtime of simulation
    'outputdt': timedelta(hours=1),      # Timestep of output
    'dt': timedelta(minutes=30),          # Timestep of advection
}

# Overwrite some settings
settings['use_3D'] = False
settings['use_biofouling'] = False
settings['use_stokes'] = False
settings['use_wind'] = False

# Plastic type settings
settings['plastictype'] = {
    'wind_coefficient' : 0.01,  # Percentage of wind to apply to particles
    'plastic_diameter' : 0.001, # Plastic particle diameter (m)
    'plastic_density' : 1030.,  # Plastic particle density (kg/m^3)
}

# Create the fieldset
fieldset = create_fieldset_from_netcdf(settings)

# Create the particleset
emission_times, total_particles = caudal_reader(settings)  # Interval between particles for each day (in seconds)
release_locs = particle_dictionary(settings)
pset = pp.constructors.create_particleset(fieldset, settings, release_locs)

# Create the applicable kernels to the plastic particles
#kernels = pp.constructors.create_kernel(fieldset)
kernels = [AdvectionRK4]

runtime = settings['simulation']['runtime']
dt = settings['simulation']['dt']
outputdt = settings['simulation']['outputdt']

# Create the particle file where output will be stored
pfile = pp.ParticleFile(output_file, pset, settings=settings, outputdt=outputdt)

pdb.set_trace()
# Execute the simulation
pset.execute(kernels, runtime=runtime, dt=dt, output_file=pfile)

