import os
import glob
import numpy as np

import pandas as pd
from parcels import FieldSet, Field, ParticleSet, JITParticle, Variable, AdvectionRK4, AdvectionRK4_3D
from parcels.tools.converters import Geographic, GeographicPolar

from plasticparcels.kernels import PolyTEOS10_bsq, StokesDrift, WindageDrift, SettlingVelocity, Biofouling, VerticalMixing, periodicBC, checkErrorThroughSurface, deleteParticle, checkThroughBathymetry
from plasticparcels.utils import select_files, select_period


def create_hydrodynamic_fieldset_from_netcdf(settings):
    """Constructor method to create a Parcels.Fieldset with a hydrodynamic field

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
    nc_files = glob.glob(nc_files_path)
    nc_files = sorted(nc_files)

    variables = settings['ocean']['variables']
    dimensions = settings['ocean']['dimensions']

    filenames = {'U': {'lon': ocean_mesh, 'lat': ocean_mesh, 'depth': ocean_mesh, 'data': nc_files},
                 'V': {'lon': ocean_mesh, 'lat': ocean_mesh, 'depth': ocean_mesh, 'data': nc_files},
                 'W': {'lon': ocean_mesh, 'lat': ocean_mesh, 'depth': ocean_mesh, 'data': nc_files},
                 'absolute_salinity': {'lon': ocean_mesh, 'lat': ocean_mesh, 'depth': ocean_mesh, 'data': nc_files},
                 'conservative_temperature': {'lon': ocean_mesh, 'lat': ocean_mesh, 'depth': ocean_mesh, 'data': nc_files},
                 'bathymetry': {'lon': ocean_mesh, 'lat': ocean_mesh, 'data': bathymetry_mesh}}

    if not settings['use_3D']:
        print('Using 2D model')
        variables.pop('W', None)
        dimensions.pop('W', None)
        filenames.pop('W', None)
        for key, value in dimensions.items():
            value.pop('depth', None)
        for key, value in filenames.items():
            value.pop('depth', None)
    else:
        print('Using 3D model')

    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, allow_time_extrapolation=True)

    fieldset.add_constant('use_mixing', settings['use_mixing'])
    fieldset.add_constant('use_biofouling', settings['use_biofouling'])
    fieldset.add_constant('use_stokes', settings['use_stokes'])
    fieldset.add_constant('use_wind', settings['use_wind'])
    fieldset.add_constant('G', 9.81)  # Gravitational constant [m s-2]
    fieldset.add_constant('use_3D', settings['use_3D'])
    fieldset.add_constant('z_start', 0.5)
    return fieldset


def create_fieldset_from_netcdf(settings):
    """ A constructor method to create a Parcels.Fieldset with all fields necessary for a plasticparcels simulation

    Parameters
    ----------
    settings :
        A dictionary of model settings used to create the fieldset

    Returns
    -------
    fieldset
        A parcels.FieldSet object
    """
    fieldset = create_hydrodynamic_fieldset_from_netcdf(settings)
    if fieldset.use_wind:
        print('Including wind data into fieldset')
        wind_files_path = settings['wind']['directory']
        windfiles = glob.glob(wind_files_path)
        windfiles = sorted(windfiles)

        variables = settings['wind']['variables']
        dimensions = settings['wind']['dimensions']

        filenames_wind = {'Wind_U': windfiles,
                          'Wind_V': windfiles}

        fieldset_wind = FieldSet.from_netcdf(filenames_wind, variables, dimensions)
        fieldset_wind.Wind_U.units = GeographicPolar()
        fieldset_wind.Wind_V.units = Geographic()

        fieldset.add_field(fieldset_wind.Wind_U)
        fieldset.add_field(fieldset_wind.Wind_V)

    return fieldset


def create_hydrodynamic_fieldset(settings):
    """ A constructor method to create a Parcels.Fieldset from hydrodynamic model data

    Parameters
    ----------
    settings :
        A dictionary of settings used to create the fieldset

    Returns
    -------
    fieldset
        A parcels.FieldSet object
    """

    # Location of hydrodynamic data
    dirread_model = os.path.join(settings['ocean']['directory'], settings['ocean']['filename_style'])

    # Start date and runtime of the simulation
    startdate = settings['simulation']['startdate']
    runtime = int(np.ceil(settings['simulation']['runtime'].total_seconds()/86400.))  # convert to days

    # Mesh masks
    ocean_mesh = os.path.join(settings['ocean']['directory'], settings['ocean']['ocean_mesh'])  # mesh_mask

    # Usan un ficheiro para cada compoñente da velocidade + temperatura + salinidade

    # Setup input for fieldset creation
    ufiles = select_files(dirread_model, 'U_%4i*.nc', startdate, runtime, dt_margin=3)
    vfiles = select_files(dirread_model, 'V_%4i*.nc', startdate, runtime, dt_margin=3)
    wfiles = select_files(dirread_model, 'W_%4i*.nc', startdate, runtime, dt_margin=3)
    tfiles = select_files(dirread_model, 'T_%4i*.nc', startdate, runtime, dt_margin=3)
    sfiles = select_files(dirread_model, 'S_%4i*.nc', startdate, runtime, dt_margin=3)

    filenames = {'U': {'lon': ocean_mesh, 'lat': ocean_mesh, 'depth': wfiles[0], 'data': ufiles},
                 'V': {'lon': ocean_mesh, 'lat': ocean_mesh, 'depth': wfiles[0], 'data': vfiles},
                 'W': {'lon': ocean_mesh, 'lat': ocean_mesh, 'depth': wfiles[0], 'data': wfiles},
                 'conservative_temperature': {'lon': ocean_mesh, 'lat': ocean_mesh, 'depth': wfiles[0], 'data': tfiles},
                 'absolute_salinity': {'lon': ocean_mesh, 'lat': ocean_mesh, 'depth': wfiles[0], 'data': sfiles}}

    variables = settings['ocean']['variables']
    dimensions = settings['ocean']['dimensions']
    indices = settings['ocean']['indices']

    if not settings['use_3D']:
        indices['depth'] = range(0, 2)

    # USA FieldSet.from_nemo --> CAMBIAR POR FieldSet.from_netcdf ????

    # Load the fieldset
    fieldset = FieldSet.from_nemo(filenames, variables, dimensions,
                                  indices=indices, allow_time_extrapolation=settings['allow_time_extrapolation'])

    # Create flags for custom particle behaviour
    fieldset.add_constant('use_mixing', settings['use_mixing'])
    fieldset.add_constant('use_biofouling', settings['use_biofouling'])
    fieldset.add_constant('use_stokes', settings['use_stokes'])
    fieldset.add_constant('use_wind', settings['use_wind'])
    fieldset.add_constant('G', 9.81)  # Gravitational constant [m s-1]
    fieldset.add_constant('use_3D', settings['use_3D'])

    # Add in bathymetry
    fieldset.add_constant('z_start', 0.5)
    bathymetry_variables = settings['ocean']['bathymetry_variables']
    bathymetry_dimensions = settings['ocean']['bathymetry_dimensions']
    bathymetry_mesh = os.path.join(settings['ocean']['directory'], settings['ocean']['bathymetry_mesh'])
    bathymetry_field = Field.from_netcdf(bathymetry_mesh, bathymetry_variables, bathymetry_dimensions)
    fieldset.add_field(bathymetry_field)

    # If vertical mixing is turned on, add in the KPP-Profile
    if fieldset.use_mixing:
        dirread_model = os.path.join(settings['ocean']['directory'], settings['ocean']['filename_style'])
        kzfiles = select_files(dirread_model, 'KZ_%4i*.nc', startdate, runtime, dt_margin=3)
        mixing_filenames = {'lon': ocean_mesh, 'lat': ocean_mesh, 'depth': wfiles[0], 'data': kzfiles}
        mixing_variables = settings['ocean']['vertical_mixing_variables']
        mixing_dimensions = settings['ocean']['vertical_mixing_dimensions']
        mixing_fieldset = FieldSet.from_nemo(mixing_filenames, mixing_variables, mixing_dimensions)
        fieldset.add_field(mixing_fieldset.mixing_kz)  # phytoplankton primary productivity

    return fieldset


def create_fieldset(settings):
    """ A constructor method to create a Parcels.Fieldset with all fields necessary for a plasticparcels simulation

    Parameters
    ----------
    settings :
        A dictionary of model settings used to create the fieldset

    Returns
    -------
    fieldset
        A parcels.FieldSet object
    """

    # First create the hydrodynamic fieldset
    fieldset = create_hydrodynamic_fieldset(settings)

    # Now add the other fields
    # Start date and runtime of the simulation
    startdate = settings['simulation']['startdate']
    runtime = int(np.ceil(settings['simulation']['runtime'].total_seconds()/86400.))  # convert to days

    if fieldset.use_biofouling:
        # MOi glossary: https://www.mercator-ocean.eu/wp-content/uploads/2021/11/Glossary.pdf
        # and https://catalogue.marine.copernicus.eu/documents/PUM/CMEMS-GLO-PUM-001-028.pdf

        # Add BGC constants to current fieldset
        for key in settings['bgc']['constants']:
            fieldset.add_constant(key, settings['bgc']['constants'][key])

        # Create a fieldset with BGC data
        dirread_bgc = os.path.join(settings['bgc']['directory'], settings['bgc']['filename_style'])
        bgc_mesh = os.path.join(settings['bgc']['directory'], settings['bgc']['bgc_mesh'])  # mesh_mask_4th

        dirread_model = os.path.join(settings['ocean']['directory'], settings['ocean']['filename_style'])
        wfiles = select_files(dirread_model, 'W_%4i*.nc', startdate, runtime, dt_margin=3)

        ppfiles = select_files(dirread_bgc, 'nppv_%4i*.nc', startdate, runtime, dt_margin=8)
        phy1files = select_files(dirread_bgc, 'phy_%4i*.nc', startdate, runtime, dt_margin=8)
        phy2files = select_files(dirread_bgc, 'phy2_%4i*.nc', startdate, runtime, dt_margin=8)

        filenames_bio = {'pp_phyto': {'lon': bgc_mesh, 'lat': bgc_mesh, 'depth': wfiles[0], 'data': ppfiles},
                         'bio_nanophy': {'lon': bgc_mesh, 'lat': bgc_mesh, 'depth': wfiles[0], 'data': phy1files},
                         'bio_diatom': {'lon': bgc_mesh, 'lat': bgc_mesh, 'depth': wfiles[0], 'data': phy2files}}

        variables_bio = settings['bgc']['variables']
        dimensions_bio = settings['bgc']['dimensions']

        # Create the BGC fieldset
        bio_fieldset = FieldSet.from_nemo(filenames_bio, variables_bio, dimensions_bio)

        # Add the fields to the main fieldset
        fieldset.add_field(bio_fieldset.pp_phyto)  # phytoplankton primary productivity
        fieldset.add_field(bio_fieldset.bio_nanophy)  # nanopyhtoplankton concentration [mmol C m-3]
        fieldset.add_field(bio_fieldset.bio_diatom)  # diatom concentration [mmol C m-3]

    if fieldset.use_stokes:
        dirread_Stokes = os.path.join(settings['stokes']['directory'], settings['stokes']['filename_style'])
        wavesfiles = select_files(dirread_Stokes, '%4i*.nc', startdate, runtime, dt_margin=32)

        filenames_Stokes = {'Stokes_U': wavesfiles,
                            'Stokes_V': wavesfiles,
                            'wave_Tp': wavesfiles}

        variables_Stokes = settings['stokes']['variables']
        dimensions_Stokes = settings['stokes']['dimensions']

        fieldset_Stokes = FieldSet.from_netcdf(filenames_Stokes, variables_Stokes, dimensions_Stokes, mesh='spherical')
        fieldset_Stokes.Stokes_U.units = GeographicPolar()
        fieldset_Stokes.Stokes_V.units = Geographic()
        fieldset_Stokes.add_periodic_halo(zonal=True)

        fieldset.add_field(fieldset_Stokes.Stokes_U)
        fieldset.add_field(fieldset_Stokes.Stokes_V)
        fieldset.add_field(fieldset_Stokes.wave_Tp)

    if fieldset.use_wind:
        dirread_wind = os.path.join(settings['wind']['directory'], settings['wind']['filename_style'])
        windfiles = select_files(dirread_wind, '%4i*.nc', startdate, runtime, dt_margin=32)

        filenames_wind = {'Wind_U': windfiles,
                          'Wind_V': windfiles}

        variables_wind = settings['wind']['variables']
        dimensions_wind = settings['wind']['dimensions']

        fieldset_wind = FieldSet.from_netcdf(filenames_wind, variables_wind, dimensions_wind, mesh='spherical')
        fieldset_wind.Wind_U.units = GeographicPolar()
        fieldset_wind.Wind_V.units = Geographic()
        fieldset_wind.add_periodic_halo(zonal=True)

        fieldset.add_field(fieldset_wind.Wind_U)
        fieldset.add_field(fieldset_wind.Wind_V)

    # Apply unbeaching currents when Stokes/Wind can push particles into land cells
    if fieldset.use_stokes or fieldset.use_wind > 0:
        unbeachfiles = settings['unbeaching']['filename']
        filenames_unbeach = {'unbeach_U': unbeachfiles,
                             'unbeach_V': unbeachfiles}

        variables_unbeach = settings['unbeaching']['variables']

        dimensions_unbeach = settings['unbeaching']['dimensions']

        fieldset_unbeach = FieldSet.from_netcdf(filenames_unbeach, variables_unbeach, dimensions_unbeach, mesh='spherical')
        fieldset_unbeach.unbeach_U.units = GeographicPolar()
        fieldset_unbeach.unbeach_V.units = Geographic()

        fieldset.add_field(fieldset_unbeach.unbeach_U)
        fieldset.add_field(fieldset_unbeach.unbeach_V)

    fieldset.add_constant('verbose_delete', settings['verbose_delete'])

    return fieldset


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


def create_particleset(fieldset, settings, release_locations):
    """ A constructor method to create a Parcels.ParticleSet for a plasticparcels simulation

    Parameters
    ----------
    fieldset :
        A Parcels.FieldSet object
    settings :
        A dictionary of model settings, simulation settings, and release settings and plastic-type settings
    release_locations :
        A dictionary of release locations for particles

    Returns
    -------
    particleset
        A parcels.ParticleSet object
    """

    # Set the longitude, latitude, and plastic amount per particle
    lons = np.array(release_locations['lons'])
    lats = np.array(release_locations['lats'])
    emission_times = release_locations['emission_times']
    if 'plastic_amount' in release_locations.keys():
        plastic_amounts = release_locations['plastic_amount']
    else:
        plastic_amounts = np.full_like(lons, np.nan)

    # Set particle properties
    plastic_densities = np.full(lons.shape, settings['plastictype']['plastic_density'])
    plastic_diameters = np.full(lons.shape, settings['plastictype']['plastic_diameter'])
    wind_coefficients = np.full(lons.shape, settings['plastictype']['wind_coefficient'])

    PlasticParticle = JITParticle
    variables = [Variable('plastic_diameter', dtype=np.float32, initial=np.nan, to_write=False),
                 Variable('plastic_density', dtype=np.float32, initial=np.nan, to_write=False),
                 Variable('wind_coefficient', dtype=np.float32, initial=0., to_write=False),
                 Variable('settling_velocity', dtype=np.float64, initial=0., to_write=False),
                 Variable('seawater_density', dtype=np.float32, initial=np.nan, to_write=False),
                 Variable('absolute_salinity', dtype=np.float64, initial=np.nan, to_write=False),
                 Variable('algae_amount', dtype=np.float64, initial=0., to_write=False),
                 Variable('plastic_amount', dtype=np.float32, initial=0., to_write=True)]
    # Add beaching variable to each particle

    for variable in variables:
        setattr(PlasticParticle, variable.name, variable)

    pset = ParticleSet.from_list(fieldset,
                                 PlasticParticle,
                                 lon=lons,
                                 lat=lats,
                                 time=emission_times,
                                 plastic_diameter=plastic_diameters,
                                 plastic_density=plastic_densities,
                                 wind_coefficient=wind_coefficients,
                                 plastic_amount=plastic_amounts)

    return pset


def create_particleset_from_map(fieldset, settings):
    """ A constructor method to create a Parcels.ParticleSet for a plasticparcels simulation from one of the available initialisation maps

    Parameters
    ----------
    fieldset :
        A Parcels.FieldSet object
    settings :
        A dictionary of model settings, simulation settings, and release settings and plastic-type settings

    Returns
    -------
    particleset
        A parcels.ParticleSet object
    """

    # Load release type information
    release_type = settings['release']['initialisation_type']

    release_quantity_names = {
        'coastal': 'MPW_Cell',
        'rivers': 'Emissions',
        'fisheries': 'fishing_hours',
        'global_concentrations': 'Concentration'
    }
    release_quantity_name = release_quantity_names[release_type]

    particle_locations = pd.read_csv(settings['release_maps'][release_type])

    # Select specific continent/region/subregion/country/economic status if applicable:
    if 'continent' in settings['release'].keys():
        particle_locations = particle_locations[particle_locations['Continent'] == settings['release']['continent']]
    if 'region' in settings['release'].keys():
        particle_locations = particle_locations[particle_locations['Region'] == settings['release']['region']]
    if 'subregion' in settings['release'].keys():
        particle_locations = particle_locations[particle_locations['Subregion'] == settings['release']['subregion']]
    if 'country' in settings['release'].keys():
        particle_locations = particle_locations[particle_locations['Country'] == settings['release']['country']]
    if 'economicstatus' in settings['release'].keys():
        particle_locations = particle_locations[particle_locations['Economic status'] == settings['release']['economicstatus']]
    if 'concentration_type' in settings['release'].keys():
        particle_locations = particle_locations[particle_locations['ConcentrationType'] == settings['release']['concentration_type']]

    particle_locations = particle_locations.groupby(['Longitude', 'Latitude'])[release_quantity_name].agg('sum').reset_index()
    particle_locations = particle_locations[particle_locations[release_quantity_name] > 0]

    release_locations = {'lons': particle_locations['Longitude'],
                         'lats': particle_locations['Latitude'],
                         'plastic_amount': particle_locations[release_quantity_name]}

    pset = create_particleset(fieldset, settings, release_locations)

    return pset


def create_kernel(fieldset):
    """ A constructor method to create a list of kernels for a plasticparcels simulation

    Parameters
    ----------
    fieldset :
        A parcels.FieldSet object containing a range of constants to turn on/off different kernel behaviours

    Returns
    -------
    kernels :
        A list of kernels used in the execution of the particle set
    """
    kernels = []

    kernels.append(PolyTEOS10_bsq)  # To set the seawater_density variable  # TODO do we need this always? Or only for some kernels?

    if fieldset.use_3D:
        kernels.append(AdvectionRK4_3D)
    else:
        kernels.append(AdvectionRK4)

    if not fieldset.use_biofouling and fieldset.use_3D:
        kernels.append(SettlingVelocity)
    elif fieldset.use_biofouling and fieldset.use_3D:  # Must be in 3D to use biofouling mode
        kernels.append(Biofouling)

    if fieldset.use_stokes:
        kernels.append(StokesDrift)

    if fieldset.use_wind:
        # print('Windage added')
        kernels.append(WindageDrift)

    if fieldset.use_mixing:
        kernels.append(VerticalMixing)

    # Add the unbeaching kernel to the beginning
    # if fieldset.use_stokes or fieldset.use_wind:
    #     kernels.append(unbeaching)

    if fieldset.use_3D:
        kernels.append(checkThroughBathymetry)
        kernels.append(checkErrorThroughSurface)

    # Add statuscode kernels
    kernels.append(periodicBC)
    kernels.append(deleteParticle)

    return kernels
