#!/usr/bin/env python

''' 
This is a python traslation of chooch C code writen by Gwyndaf Evans.
https://github.com/fadnywg/chooch

Translated by Martin Savko (savko@synchrotron-soleil.fr) to python to understand 
methods of integration around singularity a bit better.

. All credits to Gwyndaf and co-authors.'''

import numpy as np
from scipy.integrate import quad as integrate 
import scipy.signal
import mucal
import pylab
import time

def get_cross_section(element, energy_keV):
    energy, xsec, fluo = mucal.mucal(element, energy_keV)
    return xsec[0]

def get_fpp(element, energy_keV):
    return 143.10935e-10 * energy_keV * 1000.0 * get_cross_section(element, energy_keV)

def get_splinor(element, energy_keV):
    return get_cross_section(element, energy_keV)

def savgol_win(fEdge, dE, fEres=0.00014):
    
    fMonoRes = fEres * fEdge
    
    print 'fMonoRes', fMonoRes
    print 'dE', dE
    
    nSavGolWin = int(fMonoRes/dE)
    print 'nSavGolWin', nSavGolWin
    
    if nSavGolWin % 2 == 0:
        nSavGolWin += 1
    print 'nSavGolWin', nSavGolWin
    nSavGolWin = min([29, nSavGolWin])
    nSavGolWin = max([11, nSavGolWin])
    print 'nSavGolWin', nSavGolWin
    print("dE = %f Resol = %f" % (dE, fMonoRes))
    print("Savitsky-Golay window value = %d" % nSavGolWin)
    return nSavGolWin

def normalize(fXraw, fYraw, below_edge_boundary, above_edge_boundary): 
    below_edge_linear_fit = np.poly1d(np.polyfit(fXraw[:below_edge_boundary], fYraw[:below_edge_boundary], 1))
    
    above_edge_linear_fit = np.poly1d(np.polyfit(fXraw[above_edge_boundary:], fYraw[above_edge_boundary:], 1))
    
    fYfitb = below_edge_linear_fit(fXraw)
    fYfita = above_edge_linear_fit(fXraw)
    
    fYnorm = (fYraw - fYfitb) / (fYfita - fYfitb)
    return fYnorm

def get_theory_fpp(element, energy_keV):
    if type(energy_keV) in [float, int]:
        fpp = get_fpp(element, energy_keV)
    else:
        fpps = []
        for item in energy_keV:
            fpps.append(get_fpp(element, item))
        fpp = np.array(fpps)
    return fpp

def get_fYfpp(fXraw, fYnorm, element, below_edge_boundary, above_edge_boundary):
    fYtheory = get_theory_fpp(element, fXraw/1.e3)
    
    below_edge_theory_quadratic_fit = np.poly1d(np.polyfit(fXraw[:below_edge_boundary], fYtheory[:below_edge_boundary], 2))
    
    above_edge_theory_quadratic_fit = np.poly1d(np.polyfit(fXraw[above_edge_boundary:], fYtheory[above_edge_boundary:], 2))
    
    fYfitb = below_edge_theory_quadratic_fit(fXraw)
    fYfita = above_edge_theory_quadratic_fit(fXraw)
    
    fYfpp = fYnorm * (fYfita - fYfitb) + fYfitb
    return fYfpp, fYtheory

def get_edge_boundaries(fXraw, fYraw, element):
    peak = fXraw[np.argmax(fYraw)]
    below = fXraw[fXraw < peak-7.].max()
    above = fXraw[fXraw > peak+7.].min()
    below_edge_boundary = list(fXraw).index(below)
    above_edge_boundary = list(fXraw).index(above)
    fEdge = mucal.k_edge[mucal.element.index(element)+1] * 1.e3
    return below_edge_boundary, above_edge_boundary, fEdge

def integrand_extrapolate(E, E0, element):
    return E*get_fpp(element, E/1.e3)/(E0**2 - E**2)

def integrand_intrapolate(E, E0, get_fpp_intrapolate):
    return E*get_fpp_intrapolate(E)/(E0**2 - E**2)

def singularity(E, E0, get_fpp_intrapolate):
    return -1.*get_fpp_intrapolate(E)/(E0+E)
  
def get_from_energy_extrapolate_low_to_first(energy, element, energy_extrapolate_low, energy_measured_first_point, subintervals):
    #points = np.linspace(energy_extrapolate_low, energy_measured_first_point, 3)
    #boundaries = zip(points[:-1], points[1:])
    #return sum([integrate(integrand_extrapolate, l, h, args=(energy, element), limit=subintervals)[0] for l, h in boundaries]) * 2./np.pi
    return integrate(integrand_extrapolate, energy_extrapolate_low, energy_measured_first_point, args=(energy, element), limit=subintervals)[0] * 2./np.pi

def get_from_first_to_singularity((energy, a), energy_measured_first_point, get_fpp_intrapolate, subintervals):
    #points = np.linspace(energy_measured_first_point, a, 3)
    #boundaries = zip(points[:-1], points[1:])
    #return sum([integrate(integrand_intrapolate, l, h, args=(energy, get_fpp_intrapolate), limit=subintervals)[0] for l, h in boundaries]) * 2./np.pi    
    return integrate(integrand_intrapolate, energy_measured_first_point, a, args=(energy, get_fpp_intrapolate), limit=subintervals)[0] * 2./np.pi
    
def get_singularity_integral((energy, a, b), get_fpp_intrapolate, subintervals):
    return integrate(singularity, a, b, args=(energy, get_fpp_intrapolate), limit=subintervals)[0]

def get_from_singularity_to_last((energy, b), energy_measured_last_point, get_fpp_intrapolate, subintervals):
    #points = np.linspace(b, energy_measured_last_point, 3)
    #boundaries = zip(points[:-1], points[1:])
    #return sum([integrate(integrand_intrapolate, l, h, args=(energy, get_fpp_intrapolate), limit=subintervals)[0] for l, h in boundaries])* 2./np.pi
    return integrate(integrand_intrapolate, b, energy_measured_last_point, args=(energy, get_fpp_intrapolate), limit=subintervals)[0] * 2./np.pi                          

def get_from_last_to_energy_extrapolate_high(energy, element, energy_measured_last_point, energy_extrapolate_high, subintervals):
    #points = np.linspace(energy_measured_last_point, energy_extrapolate_high, 3)
    #boundaries = zip(points[:-1], points[1:])
    #return sum([integrate(integrand_extrapolate, l, h, args=(energy, element), limit=subintervals)[0] for l, h in boundaries]) * 2./np.pi
    return integrate(integrand_extrapolate, energy_measured_last_point, energy_extrapolate_high, args=(energy, element), limit=subintervals)[0] * 2./np.pi
    
def calculate_integral(X, energy_extrapolate_low, energy_extrapolate_high, energy_measured_first_point, energy_measured_last_point, element, get_fpp_intrapolate, subintervals=50):
    
    #X = [energy, fpp, fppd1, fppd2, fppd3, a, b, d1, d2]
    #Exrapolate to low energy 
    ltf = time.time()
    from_energy_extrapolate_low_to_first = np.apply_along_axis(get_from_energy_extrapolate_low_to_first, 1, X[:,[0,]], element, energy_extrapolate_low, energy_measured_first_point, subintervals)
    print 'from low to first took %.3f' % (time.time() - ltf)
    
    #From first data point up to singularity energy-dE
    fts = time.time()
    from_first_to_singularity = np.apply_along_axis(get_from_first_to_singularity, 1, X[:,[0, 5]], energy_measured_first_point, get_fpp_intrapolate, subintervals)
    print 'from first to singularity took %.3f' % (time.time() - fts)
    
    #Singularity
    s = time.time()
    at_singularity = np.apply_along_axis(get_singularity_integral, 1, X[:, [0, 5, 6]], get_fpp_intrapolate, subintervals)
    at_singularity += -(np.log(np.abs(X[:,8])) - np.log(np.abs(X[:,7])))
    at_singularity += -X[:,2] * (X[:,6] - X[:,5])
    at_singularity += -X[:,3] * (X[:,8]**2 - X[:,7]**2)/4.
    at_singularity += -X[:,4] * (X[:,8]**3 - X[:,7]**3)/18.
    at_singularity /= np.pi
    print 'singularity took %.3f' % (time.time() - s)
    
    #From singularity energy+dE up to last data point
    stl = time.time()
    from_sigularity_to_last = np.apply_along_axis(get_from_singularity_to_last, 1, X[:,[0, 6]], energy_measured_last_point, get_fpp_intrapolate, subintervals)
    print 'from singularity to last took %.3f' % (time.time() - stl)
    #Extrapolate to high energy
    lth = time.time()
    from_last_to_energy_extrapolate_high = np.apply_along_axis(get_from_last_to_energy_extrapolate_high, 1, X[:,[0,]], element, energy_measured_last_point, energy_extrapolate_high, subintervals)
    print 'last to high took %.3f' % (time.time() - lth)
    
    fYfp = from_energy_extrapolate_low_to_first + \
           from_first_to_singularity + \
           at_singularity + \
           from_sigularity_to_last + \
           from_last_to_energy_extrapolate_high
       
    return fYfp
    
def calculate_integral_for_a_single_point(i, E0, fXraw, fElo, fEhi, fYfpps, fYderiv1, fYderiv2, fYderiv3, element, get_fpp_intrapolate, dE=0.01, subintervals=50):
    
    a = E0-dE
    b = E0+dE
    d1 = a-E0
    d2 = b-E0
            
    #Exrapolate to low energy 
    I_low = integrate(integrand_extrapolate, fElo, fXraw[0], args=(E0, element), limit=subintervals)[0]
    from_fElo_to_first = I_low * 2./np.pi

    #From first data point up to singularity E0-dE
    from_first_to_singularity = integrate(integrand_intrapolate, fXraw[0], E0-dE, args=(E0, get_fpp_intrapolate), limit=subintervals)[0] * 2./np.pi
    
    #Singularity
    at_singularity = integrate(singularity, a, b, args=(E0, get_fpp_intrapolate), limit=subintervals)[0]
    at_singularity += -(np.log(np.abs(d2)) - np.log(np.abs(d1)))
    at_singularity += -fYderiv1[i]*(b-a)
    at_singularity += -fYderiv2[i]*(d2**2 - d1**2)/4.
    at_singularity += -fYderiv3[i]*(d2**3 - d1**3)/18.
    at_singularity /= np.pi
    
    #From singularity E0+dE up to last data point
    from_sigularity_to_last = integrate(integrand_intrapolate, E0+dE, fXraw[-1], args=(E0, get_fpp_intrapolate), limit=subintervals)[0] * 2./np.pi

    #Extrapolate to high energy
    from_last_to_fEhi = integrate(integrand_extrapolate, fXraw[-1], fEhi, args=(E0, element), limit=subintervals)[0] * 2./np.pi
    #from_last_to_fEhi = 0.
    fYfp = from_fElo_to_first + \
           from_first_to_singularity + \
           at_singularity + \
           from_sigularity_to_last + \
           from_last_to_fEhi
       
    return fYfp

def load_spectrum(sFilename):
    spectrum = np.loadtxt(sFilename, skiprows=2)
    fXraw = spectrum[:, 0]
    fYraw = spectrum[:, 1]
    return fXraw, fYraw

def chooch(sFilename, element, edge):
    start = time.time()
    #load specrum
    fXraw, fYraw = load_spectrum(sFilename)
    
    #pylab.figure()
    #pylab.title('Raw spectrum')
    #pylab.plot(fXraw, fYraw)
    #pylab.xlabel('energy [eV]')
    #pylab.ylabel('raw scpetrum [a.u.]')
    #pylab.show()
    #check input for common errors
    
    dE = np.min(fXraw[1:] - fXraw[:-1])
    
    #determine edge
    below_edge_boundary, above_edge_boundary, fEdge = get_edge_boundaries(fXraw, fYraw, element)
    print 'below_edge_boundary', below_edge_boundary
    print 'above_edge_boundary', above_edge_boundary
    print 'fEdge', fEdge
    print 'dE', dE
    
    #Normalize data
    fYnorm = normalize(fXraw, fYraw, below_edge_boundary, above_edge_boundary)
    
    #pylab.figure()
    #pylab.title('Normalized spectrum')
    #pylab.plot(fXraw, fYnorm)
    #pylab.xlabel('energy [eV]')
    #pylab.ylabel('normalized fluorescence [a.u.]')
    #pylab.show()
    
    #Convert spectrum to f''
    fYfpp, fYtheory = get_fYfpp(fXraw, fYnorm, element, below_edge_boundary, above_edge_boundary)
    #pylab.figure()
    #pylab.title('Spectrum converted to f"')
    #pylab.plot(fXraw, fYfpp, label='observation')
    #pylab.plot(fXraw, fYtheory, label='theory')
    #pylab.xlabel('energy [eV]')
    #pylab.ylabel('f'' [e]')
    #pylab.legend()
    #pylab.show()
    
    #determine Savitzky-Golay window
    nSavGolWin = savgol_win(fEdge, dE)
    print 'nSavGolWin', nSavGolWin
    
    fYfpps = scipy.signal.savgol_filter(fYfpp, nSavGolWin, 4, deriv=0)
    fYderiv1 = scipy.signal.savgol_filter(fYfpp, nSavGolWin, 4, deriv=1)
    fYderiv2 = scipy.signal.savgol_filter(fYfpp, nSavGolWin, 4, deriv=2)
    fYderiv3 = scipy.signal.savgol_filter(fYfpp, nSavGolWin, 4, deriv=3)
    
    #pylab.figure()
    #pylab.title('Savitzky-Golay filtered spectrum and its first three derivatives')
    #pylab.plot(fYfpps, label='fYfpps')
    #pylab.plot(fYderiv1, label='fYderiv1')
    #pylab.plot(fYderiv2, label='fYderiv2')
    #pylab.plot(fYderiv3, label='fYderiv3')
    #pylab.legend()
    #pylab.show()
    
    get_fpp_intrapolate = scipy.interpolate.interp1d(fXraw, fYfpps, kind='linear', fill_value='extrapolate')
    #Perform Kramer-Kroning transform
    dE = 0.01
    fElo = fEdge/1.e3
    fEhi = fEdge*50.

    #fYfp = scipy.integrate.quad(kramers_kronig_integrand, fElo, fXraw[0])
    a = fXraw - dE
    b = fXraw + dE
    d1 = a - fXraw
    d2 = b - fXraw
    
    X = np.vstack([fXraw, fYfpps, fYderiv1, fYderiv2, fYderiv3, a, b, d1, d2]).T
    
    print 'time until integration %.3f' % (time.time() - start)
    
    start = time.time()
    fYfpx = calculate_integral(X, fElo, fEhi, fXraw[0], fXraw[-1], element, get_fpp_intrapolate, subintervals=35)
    end = time.time()
    print 'integration1 took %.3f seconds' % (end-start)
    
    #start = time.time()
    #fYfp = []
    #for i, energy in enumerate(fXraw):
        #integral = calculate_integral_for_a_single_point(i, energy, fXraw, fElo, fEhi, fYfpps, fYderiv1, fYderiv2, fYderiv3, element, get_fpp_intrapolate, dE=0.01)
        #fYfp.append(integral)
    #end = time.time()
    #print 'integration2 took %.3f seconds' % (end-start)
    pylab.figure()
    pylab.title("f'' and f'")
    #pylab.plot(fXraw, fYfp, label='fYfp')
    pylab.plot(fXraw, fYfpx, label='fYfpx')
    pylab.plot(fXraw, fYfpps, label='fYfpp')
    #pylab.plot(fXraw, get_fpp_intrapolate(fXraw), label='spline')
    pylab.xlabel('energy [eV]')
    pylab.ylabel("f' and f'' [electrons]")
    pylab.legend()
    pylab.show()
    
if __name__ == '__main__':
    import optparse
    usage='''Usage: chooch.py -e <element> <filename>\n
             Try chooch -h to show all options\n'''
             
    parser = optparse.OptionParser(usage=usage)
    
    parser.add_option('-s', '--spectrum', type=str, help='Spectrum in .raw format')
    parser.add_option('-e', '--element', default=None, type=str, help='Atomic element')
    parser.add_option('-a', '--edge', default=None, type=str, help='Absorption edge entered but will be auto-determined anyway')
    parser.add_option('-r', default=None, type=float, help='Energy resolution')
    parser.add_option('-k', action='store_true', help='Input data will be converted from keV to eV')
    parser.add_option('-x', action='store_true', help='display graphics window')
    parser.add_option('-o', default=None, type=str, help='Output file name')
    parser.add_option('-i', action='store_true', help='plot in window')
    parser.add_option('-p', default=None, type=str, help='PS output file')
    parser.add_option('-g', default=None, type=str, help='PNG output file')
    parser.add_option('-v', default=None, type=int, help='Verbosity level'  )
    parser.add_option('-1', default=None, type=float, help='Below edge fit lower limit')
    parser.add_option('-2', default=None, type=float, help='Below edge fit upper limit')
    parser.add_option('-3', default=None, type=float, help='Above edge fit lower limit')
    parser.add_option('-4', default=None, type=float, help='Above edge fit upper limit')
    parser.add_option('-d', action='store_true', help='Dump data file for pooch')
    parser.add_option('-z', action='store_true', help='Output splinor file for raddose')
    parser.add_option('-f', default=None, type=str, help='return anom. scattering factors for RemE')
    
    options, args = parser.parse_args()
    
    chooch(options.spectrum, options.element, options.edge)
