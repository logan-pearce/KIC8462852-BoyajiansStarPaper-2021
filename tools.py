import numpy as np

# A small function for performing Monte Carlo simulations:
def MonteCarloIt(thing, N = 10000):
    ''' Generate a random sample of size = N from a 
    Gaussian centered at thing[0] with std thing[1]
    
    Args:
        thing (tuple, flt): tuple of (value,uncertainty).  Can be either astropy units object \
            or float
        N (int): number of samples
    Returns:
        array: N random samples from a Gaussian.

    Written by Logan Pearce, 2020
    '''
    try:
        out = np.random.normal(thing[0].value,thing[1].value,N)
    except:
        out = np.random.normal(thing[0],thing[1],N)

    return out

def to_si(mas,mas_yr,d):
    '''Convert from mas -> km and mas/yr -> km/s
        Input: 
         mas (array) [mas]: separation in mas
         mas_yr (array) [mas/yr]: velocity in mas/yr
         d (float) [pc]: distance to system in parsecs
        Returns:
         km (array) [km]: separation in km
         km_s (array) [km/s]: velocity in km/s
    '''
    import astropy.units as u
    km = ((mas*u.mas.to(u.arcsec)*d)*u.AU).to(u.km)
    km_s = ((mas_yr*u.mas.to(u.arcsec)*d)*u.AU).to(u.km)
    km_s = (km_s.value)*(u.km/u.yr).to(u.km/u.s)
    return km.value,km_s

def circular_velocity(au,m):
    """ Given separation in AU and total system mass, return the velocity of a test particle on a circular orbit
        around a central body at that mass """
    import astropy.constants as c
    import astropy.units as u
    import numpy as np
    m = m*u.Msun
    au = au*u.AU
    v = np.sqrt( c.G * m.to(u.kg) / (au.to(u.m)) )
    return v.to(u.km/u.s)

def physical_separation(d,theta):
    """
    Returns separation between two objects in the plane of the sky in AU given distance and parallax
    Distance and parallax must be astropy unit objects.
    Args:
        d (float): distance
        theta (float): parallax
    Return:
        separation in AU
    Written by: Logan Pearce, 2017
    """
    from astropy import units as u
    d = d.to(u.pc)
    theta = theta.to(u.arcsec)
    a = (d)*(theta)
    return a.to(u.au, equivalencies=u.dimensionless_angles())

def distance(parallax,parallax_error):
    '''Computes distance from Gaia parallaxes using the Bayesian method of Bailer-Jones 2015.
    Input: parallax [mas], parallax error [mas]
    Output: distance [pc], 1-sigma uncertainty in distance [pc]
    '''
    import numpy as np
    # Compute most probable distance:
    L=1350 #parsecs
    # Convert to arcsec:
    parallax, parallax_error = parallax/1000., parallax_error/1000.
    # establish the coefficients of the mode-finding polynomial:
    coeff = np.array([(1./L),(-2),((parallax)/((parallax_error)**2)),-(1./((parallax_error)**2))])
    # use numpy to find the roots:
    g = np.roots(coeff)
    # Find the number of real roots:
    reals = np.isreal(g)
    realsum = np.sum(reals)
    # If there is one real root, that root is the  mode:
    if realsum == 1:
        gd = np.real(g[np.where(reals)[0]])
    # If all roots are real:
    elif realsum == 3:
        if parallax >= 0:
            # Take the smallest root:
            gd = np.min(g)
        elif parallax < 0:
            # Take the positive root (there should be only one):
            gd = g[np.where(g>0)[0]]
    
    # Compute error on distance from FWHM of probability distribution:
    from scipy.optimize import brentq
    rmax = 1e6
    rmode = gd[0]
    M = (rmode**2*np.exp(-rmode/L)/parallax_error)*np.exp((-1./(2*(parallax_error)**2))*(parallax-(1./rmode))**2)
    lo = brentq(lambda x: 2*np.log(x)-(x/L)-(((parallax-(1./x))**2)/(2*parallax_error**2)) \
               +np.log(2)-np.log(M)-np.log(parallax_error), 0.001, rmode)
    hi = brentq(lambda x: 2*np.log(x)-(x/L)-(((parallax-(1./x))**2)/(2*parallax_error**2)) \
               +np.log(2)-np.log(M)-np.log(parallax_error), rmode, rmax)
    fwhm = hi-lo
    # Compute 1-sigma from FWHM:
    sigma = fwhm/2.355
            
    return gd[0],sigma
