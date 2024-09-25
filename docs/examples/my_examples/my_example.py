
''' PlasticParcels example for the Italian coast '''

import pdb
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


import plasticparcels as pp
from parcels import FieldSet

output_file = '/mnt/c/Users/martino.rial/Escritorio/resultados_parcels/my_example.zarr'
settings_file = '/mnt/c/Users/martino.rial/Escritorio/git/plasticparcels/docs/examples/my_examples/my_example_settings.json'


def create_fieldset_from_netcdf(settings):
    """Constructor method to create a Parcels.Fieldset with all fields necessary for a plasticparcels simulation

    Parameters
    ----------
    settings :
        A dictionary of model settings used to create the fieldset

    Returns
    -------
    fieldset
        A parcels.FieldSet object
    """

    nc_files_path = settings['ocean']['directory']
    ocean_mesh = settings['ocean']['ocean_mesh']
    bathymetry_mesh = settings['ocean']['bathymetry_mesh']
    nc_files = [os.path.join(nc_files_path, arquivo) for arquivo in os.listdir(nc_files_path) if arquivo.endswith('.nc4')]
    nc_files = sorted(nc_files)

    variables = {'U': 'u',
                 'V': 'v',
                 'W': 'w',
                 'absolute_salinity': 'salt',
                 'conservative_temperature': 'temp',
                 'bathymetry': 'bathymetry'}

    dimensions = {'U': {'lon': 'lon', 'lat': 'lat', 'depth': 'depth', 'time': 'time'},
                  'V': {'lon': 'lon', 'lat': 'lat', 'depth': 'depth', 'time': 'time'},
                  'W': {'lon': 'lon', 'lat': 'lat', 'depth': 'depth', 'time': 'time'},
                  'absolute_salinity': {'lon': 'lon', 'lat': 'lat', 'depth': 'depth', 'time': 'time'},
                  'conservative_temperature': {'lon': 'lon', 'lat': 'lat', 'depth': 'depth', 'time': 'time'},
                  'bathymetry': {'lon': 'lon', 'lat': 'lat', 'depth': 'depth'}}

    filenames = {'U': {'lon': ocean_mesh, 'lat': ocean_mesh, 'depth': ocean_mesh, 'data': nc_files},
                 'V': {'lon': ocean_mesh, 'lat': ocean_mesh, 'depth': ocean_mesh, 'data': nc_files},
                 'W': {'lon': ocean_mesh, 'lat': ocean_mesh, 'depth': ocean_mesh, 'data': nc_files},
                 'absolute_salinity': {'lon': ocean_mesh, 'lat': ocean_mesh, 'depth': ocean_mesh, 'data': nc_files},
                 'conservative_temperature': {'lon': ocean_mesh, 'lat': ocean_mesh, 'depth': ocean_mesh, 'data': nc_files},
                 'bathymetry': {'lon': ocean_mesh, 'lat': ocean_mesh, 'data': bathymetry_mesh}}

    if not settings['use_3D']:
        variables.pop('W', None)
        dimensions.pop('W', None)
        filenames.pop('W', None)
        for key, value in dimensions.items():
            value.pop('depth', None)
        for key, value in filenames.items():
            value.pop('depth', None)

    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, allow_time_extrapolation=True)

    fieldset.add_constant('use_mixing', settings['use_mixing'])
    fieldset.add_constant('use_biofouling', settings['use_biofouling'])
    fieldset.add_constant('use_stokes', settings['use_stokes'])
    fieldset.add_constant('use_wind', settings['use_wind'])
    fieldset.add_constant('G', 9.81)  # Gravitational constant [m s-2]
    fieldset.add_constant('use_3D', settings['use_3D'])
    fieldset.add_constant('z_start', 0.5)
    return fieldset


def select_period(start_date, end_date, data):
    """ Select data within a certain period from CSV with dates column """
    data = data[(data['Data'] >= start_date)  # "Data" means "date" in Galician
                & (data['Data'] <= end_date)].copy()
    return data


def generate_emissions_from_source(settings):
    """ Method to read a CSV file containing daily river flowrate
    Distrubute particles along the study period according to this flowrate

    Parameters
    ----------
    settings:
        A dictionary of model settings containing simluation parameters

    Returns
    -------
    emission_times:
        A list containing each particle's emission time (seconds since start)
    """

    flowrate_file = settings['sources']['river']
    npart = settings['sources']['npart']
    runtime = settings['simulation']['runtime']
    start = settings['simulation']['startdate']
    end = start + runtime
    start_date = start.strftime('%Y-%m-%d-%H:%M:%S')
    end_date = end.strftime('%Y-%m-%d-%H:%M:%S')

    flowrate = pd.read_csv(flowrate_file)
    period = select_period(start_date, end_date, flowrate)
    period.loc[:, 'Valor'] = period['Valor'].replace(to_replace=-9999.0, value=np.nan)
    period.loc[:, 'Valor'] = np.abs(period['Valor'])
    period.loc[:, 'Valor'] = period['Valor'].interpolate()
    caudal_diario = period['Valor'].tolist()
    caudal_diario = np.array(caudal_diario)
    total = np.sum(caudal_diario)

    relative_caudal = caudal_diario / total

    daily_rates = npart * relative_caudal
    rates = daily_rates / (3600 * 24)
    intervals = 1 / rates

    emission_times = np.zeros(npart)
    count = 0
    j = 0
    day = 1
    for i in range(npart):
        emission_times[i] = count
        count = count + intervals[j]
        if int(count) > day * 24 * 3600:
            j = j + 1
            day = day + 1
            if j == len(intervals):
                break

    return emission_times


def generate_uniform_emissions(settings):
    """ Method to uniformely distribute a certain number of particles along the
        study period and calculate the emssion time for each one

    Parameters
    ----------
    settings:
        A dictionary of model settings containing simluation parameters

    Returns
    -------
    emission_times:
        A list containing each particle's emission time (seconds since start)
    """

    npart = settings['sources']['npart']
    runtime = settings['simulation']['runtime']
    runtime_seconds = runtime.days*3600*24
    emission_times = np.linspace(0, runtime_seconds, npart, endpoint=False)

    return emission_times


def particle_dictionary(settings):
    """ Creates a dictionary containing the necessary data for each particle
    in order to create the particle set

    Parameters
    ----------
    settings:
        A dictionary of model settings used to create the fieldset

    Returns
    -------
    pdict:
        dictionary containing lat, lon and emission time for each particle
    """
    npart = settings['sources']['npart']
    emission_times = generate_uniform_emissions(settings)
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
    'startdate': datetime.strptime('2024-09-01-00:00:00', '%Y-%m-%d-%H:%M:%S'),  # Start date of simulation
    'runtime': timedelta(days=2),        # Runtime of simulation
    'outputdt': timedelta(hours=1),      # Timestep of output
    'dt': timedelta(minutes=30),          # Timestep of advection
}

# Overwrite some settings
settings['use_3D'] = False
settings['use_wind'] = False
settings['use_biofouling'] = False
settings['use_stokes'] = False

# Plastic type settings
settings['plastictype'] = {
    'wind_coefficient': 0.01,  # Percentage of wind to apply to particles
    'plastic_diameter': 0.001,  # Plastic particle diameter (m)
    'plastic_density': 1030.}  # Plastic particle density (kg/m^3)

# Create the fieldset
fieldset = create_fieldset_from_netcdf(settings)

# Create the particleset
release_locs = particle_dictionary(settings)
pset = pp.constructors.create_particleset(fieldset, settings, release_locs)

# Create the applicable kernels to the plastic particles
kernels = pp.constructors.create_kernel(fieldset)

runtime = settings['simulation']['runtime']
dt = settings['simulation']['dt']
outputdt = settings['simulation']['outputdt']

# Create the particle file where output will be stored
pfile = pp.ParticleFile(output_file, pset, settings=settings, outputdt=outputdt)

pdb.set_trace()
pset.execute(kernels, runtime=runtime, dt=dt, output_file=pfile)
