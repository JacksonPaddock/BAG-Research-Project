# -*- coding: utf-8 -*-
# Jackson Paddock

# ASSUMPTIONS:
# (1) 90nm channel length, 500nm finger width.
# (2) LVT devices
# (3) All NMOS devices share a well, all PMOS devices share a well
# (4) 300K
# (5) TT process corner

import pprint

import numpy as np
import scipy.optimize as sciopt

import sys
sys.path.append("./BAG_framework/")
sys.path.append("./bag_testbenches_ec/")

from math import isnan
from bag.util.search import BinaryIterator
from verification_ec.mos.query import MOSDBDiscrete
from scipy import signal
from bag.data.lti import LTICircuit, get_w_3db, get_stability_margins



def get_db(spec_file, intent, interp_method='spline', sim_env='TT'):
    # initialize transistor database from simulation data
    mos_db = MOSDBDiscrete([spec_file], interp_method=interp_method)
    # set process corners
    mos_db.env_list = [sim_env]
    # set layout parameters
    mos_db.set_dsn_params(intent=intent)
    return mos_db

def design_seriesN_reg_eqn(db_n, sim_env,
        ibias, vdd, vref, vb_n,
        cload, rload, rsource,
        vg_res, psrr_min, pm_min, err_max,
        linereg_max, loadreg_max, delta_v_lnr, delta_i_ldr):
    """
    Designs an LDO with an op amp abstracted as a voltage-controlled voltage
    source and source resistance. Equation-based.
    Inputs:
        db_n:         Database for NMOS device characterization data.
        sim_env:      Simulation corner.
        ibias:        Float. Bias current source.
        vdd:          Float. Supply voltage.
        vref:         Float. Reference voltage.
        vb_n:         Float. Back-gate/body voltage of NMOS.
        cload:        Float. Output load capacitance.
        rload:        Float. Output load resistance.
        rsource:      Float. Supply resistance.
        vg_res:       Float. Step resolution when sweeping transistor gate voltage.
        psrr_min:     Float. Minimum PSRR.
        pm_min:       Float. Minimum phase margin.
        err_max       Float. Maximum percent static error at output.
        linereg_max   Float. Maximum percent change in output voltage given
                             change in input voltage.
        loadreg_max   Float. Maximum percent change in output voltage given
                             change in output current.
        delta_v_lnr   Float. Given change in input voltage to calculate line reg.
        delta_i_ldr   Float. Given change in output current to calculate load reg.

    Raises:
        ValueError: If unable to meet the specification requirements.
    Returns:
        A dictionary with the following key:value pairings:
        Ao:     Float. DC gain of op amp.
        w1:     Float. Lowest op amp pole frequency.
        w2:     Float. Second lowest op amp pole frequency.
        psrr:   Float. Expected PSRR.
        pm:     Float. Expected phase margin.
        linereg:Float. Expected maximum percent line regulation for given input.
        loadreg:Float. Expected maximum percent load regulation for given input.
    """
    if rload == 0:
        raise ValueError("Output is shorted to ground.")

    # Adjust for total current and voltage affected by it.
    i_total = ibias + vref/rload
    vds = vdd - i_total*rsource - vref

    # Create range for sweep of gate voltage.
    vg_vec = np.arange(vref, vdd, vg_res)

    # Find the closest operating point and corresponding small signal parameters.
    best_i_total_est = float('inf')
    gm = 0
    ro = 0
    
    for vg in vg_vec:
        params = db_n.query(vgs=vg-vref, vds=vds, vbs=vb_n-vref)
        i_total_est = params['ibias']
        if abs(i_total - best_i_total_est) > abs(i_total - i_total_est):
            best_i_total_est = i_total_est
            gm = params['gm']
            ro = 1 / params['gds']

    print("Estimated gm and ro: {}, {}\n".format(gm, ro))
    #print(db_n.query(vds=vds,vbs=vb_n-vref))

    # Find location of output pole with determined parameters.
    wn = (ro + rsource + rload + gm*ro*rload)/(rload*cload*(ro + rsource))
    # Find lowest possible unity gain frequency, not dependent on other poles.
    if psrr_min > wn*(ro + rsource + gm*ro*rsource):
        wo_min = wn*np.sqrt((psrr_min/(wn*cload*(ro + rsource)))**2 - 1)
    else:
        wo_min = 1
    # Sweep op amp lower pole magnitude and determine other parameters.
    stop = 8 #TODO: Define upper limit on pole frequencies.
    designs = []
    for w1 in np.logspace(1,stop):
        # Choose third pole to interfere less with second pole. TODO: Better way?
        w2 = 100*max(wn, w1)
        # Find minimum op amp gain required.
        # With 3 poles, wo is the solution to a cubic equation:
        a = 1
        b = -(wn + w1 + w2) * np.tan(np.pi/180*(180 - pm_min))
        c = -(wn*w1 + wn*w2 + w1*w2)
        d = wn*w1*w2*np.tan(np.pi/180*(180 - pm_min))
        wo_max = float('inf')
        for root in np.roots([a,b,c,d]):
            if wo_max > root and root >= 0:
                wo_max = root
        if wo_min > wo_max:
            print("Bounds cannot be met for specific op amp poles. Trying next iteration.")
            continue
        

        # TODO: Check if op amp design is plausible through different file?
        
        # TODO: Check definition of Ao. What equation is it derived from? why does it not set psrr correctly?
        for wo in np.logspace(np.log10(wo_min), np.log10(wo_max)):
            Ao = wn*(ro + rsource)*cload*(1 + (wo/wn)**2)*(1 + (wo/w1)**2)*(1 + (wo/w2)**2)/(gm*ro)
            if Ao < vg/err_max - 1:
                continue
            # Variables for transfer function and output resistance equations.
            a2 = ro + rsource + rload + gm*ro*rload
            b2 = gm*ro*rload*Ao + a2
            c2 = rload*cload*(ro + rsource)
            d2 = w1*w2
            e2 = 1/w1 + 1/w2
            f2 = 1/(w1*w1)+1/(w2*w2)
            if a2 > c2*d2*e2 and b2/(a2 - c2*d2*e2) == 1 + a2*e2/c2:
                asymptote = True
                H_max = float('inf')
                rout_max = float('inf')
            else:
                asymptote = False
                # Find maximum magnitude of transfer function in [0, wo].
                H = lambda x: (b2 - a2) / np.sqrt((b2 + x*x*(c2*e2 - a2/d2))**2 + (x*(a2*e2 + c2*(1 - x*x/d2)))**2)
                H_max = max(H(0),H(wo))
                a3 = 3*c**2
                b3 = - 4*c2*d2*(c2 + 2*a2*e2) + 2*(c2*d2*e2)**2 + 2*a2**2
                c3 = 2*c2*d2**2*e2*(b2 + a2) + (a2*d2*e2)**2 + (c2*d2)**2 - 2*a2*d2*b2
                deriv_roots2 = np.sqrt(np.roots([a3, b3, c3]))
                for root in deriv_roots2:
                    if np.isreal(root): #TODO: Should root only be considered if less than wo?
                        H_max = max(H_max, H(root))
                # Find maximum magnitude of output impedance in [0, wo].
                rout = lambda x: (ro + rsource)/(gm*ro*Ao)*H(x)*np.sqrt((1+(x/w1)**2)*(1+(x/w2)**2))
                rout_max = max(rout(0),rout(wo))
                a4 = 4*c2**2/d2**4
                b4 = (c2**2*(3*e2*e2 + 5*f2) - 12*a2*c2*e2/d2 + 2*(a2/d2)**2)/d2**2
                c4 = 3*f2*(c2*e2)**2 - 10*a2*c2*e2*f2/d2 + 3*f2*(a2/d2)**2 + 12*(c2/d2)**2
                d4 = 2*c2*e2*f2*(a2 + b2) + f2*(a2*e2)**2 + f2*c2**2 + 4*(c2*e2)**2 - 2*a2*(b2*f2 + 8*c2*e2)/d2 + (4*a2**2 - 2*b2**2)/d2**2
                e4 = 4*c2*e2*(a2 + b2) + 2*(a2*e2)**2 + 2*c2**2 - 4*a2*b2/d2 - b2**2*f2
                deriv_roots3 = np.sqrt(np.roots([a4, b4, c4, d4, e4]))
                for root in deriv_roots3:
                    if np.isreal(root): #TODO: Should root only be considered if less than wo?
                        rout_max = np.real(max(rout_max, rout(root)))
            # Check if parameters fit given bounds.
            fits_linereg, fits_loadreg = False, False
            linereg = H_max*delta_v_lnr / vref
            loadreg = rout_max*delta_i_ldr / vref
            #print("line reg, load reg: {}, {}".format(linereg,loadreg))
            if not asymptote:
                if linereg_max >= linereg:
                    fits_linereg = True
                    #print("Line Regulation bound met.")
                if loadreg_max >= loadreg:
                    fits_loadreg = True
                    #print("Load Regulation bound met.")
            A = Ao / np.sqrt((1 + wo*wo/(w1*w1))*(1 + wo*wo/(w2*w2)))
            psrr = gm*ro*A
            if fits_linereg and fits_loadreg and psrr > psrr_min:
                pm = 180 - 180/np.pi*(np.arctan(wo/wn) + np.arctan(wo/w1) + np.arctan(wo/w2))
                designs += [(Ao, w1, w2, psrr, pm, linereg, loadreg, wo, vg, i_total)]
                #print("All bounds met.")
            else:
                #print("Not all bounds met.\n")
                pass
                
    if designs == []:
        print("FAIL. No solutions found.")
        return
    else:
        print("Number of solutions found: {}".format(len(designs)))
        final_op_amp = designs[len(designs)//2]
        final_design = dict(
            ibias=best_i_total_est - vref/rload,
            Ao=final_op_amp[0],
            w1=final_op_amp[1],
            w2=final_op_amp[2],
            psrr=final_op_amp[3],
            pm=final_op_amp[4],
            linereg=final_op_amp[5],
            loadreg=final_op_amp[6],
            wo=final_op_amp[7],
            vg=final_op_amp[8],
            i_total=final_op_amp[9])
        return final_design


def design_seriesN_reg_lti(db_n, sim_env,
        ibias, vdd, vref, vb_n,
        cload, rload, rsource,
        vg_res, psrr_min, pm_min,
        linereg_max, loadreg_max, delta_v_lnr, delta_i_ldr):
    """
    Designs an LDO with an op amp abstracted as a voltage-controlled voltage
    source and source resistance. Equation-based.
    Inputs:
        db_n:         Database for NMOS device characterization data.
        sim_env:      Simulation corner.
        vg_res:       Float. Step resolution when sweeping transistor gate voltage.
        vdd:          Float. Supply voltage.
        vref:         Float. Reference voltage.
        vb_n:         Float. Back-gate/body voltage of NMOS.
        cload:        Float. Output load capacitance.
        rload:        Float. Output load resistance.
        rsource:      Float. Supply resistance.
        psrr_min:     Float. Minimum PSRR.
        pm_min:       Float. Minimum phase margin.
        ibias_margin  Float. Maximum acceptable percent error for bias current.
        Ao_margin     Float. Maximum acceptable percent error for op amp DC gain.
        linereg_max   Float. Maximum percent change in output voltage given
                             change in input voltage.
        loadreg_max   Float. Maximum percent change in output voltage given
                             change in output current.
        delta_v_lnr   Float. Given change in input voltage to calculate line reg.
        delta_i_ldr   Float. Given change in output current to calculate load reg.

    Raises:
        ValueError: If unable to meet the specification requirements.
    Returns:
        A dictionary with the following key:value pairings:
        ibias:  Float. DC bias current.
        Ao:     Float. DC gain of op amp.
        w1:     Float. Lowest op amp pole frequency.
        w2:     Float. Second lowest op amp pole frequency.
        psrr:   Float. Expected PSRR.
        pm:     Float. Expected phase margin.
        linereg:Float. Expected maximum percent line regulation for given input.
        loadreg:Float. Expected maximum percent load regulation for given input.
    """
    return




def run_main():
    interp_method = 'spline'
    sim_env = 'tt'
    nmos_spec = 'specs_mos_char/nch_w0d5_90n.yaml'
    intent = 'lvt'

    nch_db = get_db(nmos_spec, intent, interp_method=interp_method,
                    sim_env=sim_env)

    specs = dict(
        db_n=nch_db,
        sim_env=sim_env,
	ibias=1e-6,
	vdd=1.0,
	vref=0.5,
	vb_n=0,
	cload=20e-14,
	rload=1e6,
	rsource=100,
        vg_res=0.001,
	psrr_min=1,
        pm_min=45,
        err_max=0.1,
	linereg_max=0.02,
	loadreg_max=0.05,
	delta_v_lnr=1e-3,
	delta_i_ldr=1e-9
        )

    amp_specs = design_seriesN_reg_eqn(**specs)
    # amp_specs = design_seriesN_reg_lti(**specs)
    pprint.pprint(amp_specs)
    print('done')

if __name__ == '__main__':
    run_main()
