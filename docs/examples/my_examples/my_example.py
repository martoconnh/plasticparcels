
''' PlasticParcels example for the Italian coast '''

import pdb
from datetime import datetime, timedelta

import plasticparcels as pp


output_file = '/mnt/c/Users/martino.rial/Escritorio/resultados_parcels/my_example.zarr'
settings_file = '/mnt/c/Users/martino.rial/Escritorio/git/plasticparcels/docs/examples/my_examples/my_example_settings.json'


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
    emission_times = pp.constructors.generate_uniform_emissions(settings)
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
fieldset = pp.constructors.create_fieldset_from_netcdf(settings)

# Create the particleset
release_locs = particle_dictionary(settings)
pset = pp.constructors.create_particleset(fieldset, settings, release_locs)

# Create the applicable kernels to the plastic particles
kernels = pp.constructors.create_kernel(fieldset)

runtime = settings['simulation']['runtime']
dt = settings['simulation']['dt']
outputdt = settings['simulation']['outputdt']

# Create the particle file where output will be stored
pfile = pp.particlefile.ParticleFile(output_file, pset, settings=settings, outputdt=outputdt)

pdb.set_trace()
pset.execute(kernels, runtime=runtime, dt=dt, output_file=pfile)
