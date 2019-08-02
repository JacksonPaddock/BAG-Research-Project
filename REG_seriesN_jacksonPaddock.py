# -*- coding: utf-8 -*-
# Jackson Paddock

# ASSUMPTIONS:
# (1) 100nm channel length, 500nm finger width.
# (2) LVT devices
# (3) All NMOS devices share a well, all PMOS devices share a well
# (4) 300K
# (5) TT process corner

import pprint

import numpy as np
import scipy.optimize as sciopt

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
        vg_res, vdd, vref, vb_n,
        cload, rload, rsource=0,
        psrr_min, pm_min, ibias_margin, Ao_margin,
        linereg_max, loadreg_max,
        delta_v_lnr, delta_i_ldr):
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
    # Get sweep values (Vg, Vd)
    vth = db_n.query()['vth'] #DEBUGGING: Figure out how this will actually work (syntax)
    vg_min = vref + vth
    vg_max = vdd + vth
    vg_vec = np.arange(vg_min, vg_max, vg_res)

    nf_n_vec = np.arange(1, 20, 1)  # DEBUGGING: Is there a non-brute force way of setting this?

    # Find the best operating point
    best_ibias = float('inf')
    best_gm    = 0
    best_ro    = 0


    for vg in vg_vec:
        init_params = db_n.query(vgs=vg-vref, vds=vdd-vref, vbs=vb_n-0)
        ibias_est = init_params['ibias']
        margin = ibias_margin + 1
        while margin > ibias_margin:
            if ibias_est == 0:
                raise ValueError("Estimate of bias current is 0.")
            params = db_n.query(vgs=vg-vref, vds=vdd-vref-rsource*ibias_est, vbs=vb_n-0)
            ibias_new = params['ibias']
            new_margin = abs(ibias_est - ibias_new)/ibias_est
            if new_margin > margin:
                raise ValueError("Current accuracy margin is increasing.")
            margin = new_margin
            ibias_est = ibias_new

        # Finding current and small signal parameters at gate voltage
        n_op_info = db_n.query(vgs=vg-vref, vds=high_vds, vbs=vb_n-0)
        ibias = n_op_info['ibias']
        gm    = n_op_info['gm']
        gds   = n_op_info['gds']
        ro    = 1/abs(gds)

        if gm*ro > best_gm*best_ro and :
            best_vg    = vg
            best_ibias = ibias
            best_gm    = gm
            best_ro    = ro
            print("Updating best gmro: (ibias: {}), (gm: {}), (ro: {}), (gmro: {})\n".format(best_ibias,best_gm,best_ro,best_gm*best_ro))
    gm = best_gm
    ro = best_ro
    print("Chosen bias current: {}\n\n".format(best_ibias))

    # Find location of output pole with determined parameters.
    wn = (ro + rsource + rload + gm*ro*rload)/(rload*cload*(ro + rsource))
    # Find lowest possible unity gain frequency, not dependent on other poles.
    if psrr_min > wn*(ro + rsource + gm*ro*rsource):
        wo_min = wn*np.sqrt((psrr_min/(wn*cload*(ro + rsource)))**2 - 1)
    else:
        wo_min = 0
    # Sweep op amp lower pole magnitude and determine other parameters.
    stop = 6 #TODO: Define upper limit on pole frequencies.
    designs = []
    for w1 in np.logspace(1,stop):
        # Choose third pole to interfere less with second pole.
        w2 = 100*max(wn, w1)
        # Find minimum op amp gain required.
        # With 3 poles, wo is the solution to a very messy cubic equation:
        a = - np.tan(np.pi/180 * (180 - pm_min)) * (wn + w1 + w2)
        b = - (wn*w1 + wn*w2 + w1*w2)
        c = - np.tan(np.pi/180 * (180 - pm_min)) * wn*w1*w2
        d = np.cbrt((9*a*b - 2*a**3 - 27*c + 3*np.sqrt(3*(4*a**3*c - (a*b)**2 - 18*a*b*c + 4*b**3 + 27*c*c))) / 2)
        wo_max = (d - (3*b - a*a)/d - a) / 3
        if wo_min > wo_max:
            print("Bounds cannot be met for op amp poles. Trying next iteration.")
            continue
        Ao_max = wn * cload * (ro + rsource) / (gm*ro) * np.sqrt((1 + wo*wo/(wn*wn))(1 + wo_max**2/(w1*w1))(1 + wo_max**2/(w2*w2)))
        Ao_min = psrr_min / (gm*ro) * np.sqrt((1 + wo_min**2/(w1*w1))(1 + wo_min**2/(w2*w2)))
        Ao = (Ao_max + Ao_min) / 2
        # TODO: Check if op amp design is plausible through different file?
        """ TODO: Sweep through Ao to decide rather than using midpoint?
                  Or seep wo with bounds above to make equation simpler?
                  For (wo, Ao) > 0, relationship is monotonic and cubic.
                  H and Rout in terms of wo or Ao is even uglier than code below...
        """

        # Find minimum op amp gain in [0, wo], which is at wo.
        A = Ao / np.sqrt((1 + wo*wo/(w1*w1))*(1 + wo*wo/(w2*w2)))
        # Variables for transfer function and output resistance equations.
        a1 = ro + rsource + rload + gm*ro*rload
        b1 = gm*ro*rload*Ao + a1
        c1 = rload*cload*(ro + rsource)
        d1 = w1*w2
        e1 = 1/w1 + 1/w2
        f1 = 1/(w1*w1)+1/(w2*w2)
        if a1 > c1*d1*e1 and b1/(a1 - c1*d1*e1) == 1 + a1*e1/c1:
            asymptote = True
            H_max = float('inf')
            rout_max = float('inf')
        else:
            asymptote = False
            # Find maximum magnitude of transfer function in [0, wo].
            H = lambda x: (b1 - a1) / np.sqrt((b1 + x*x*(c1*e1 - a1/d1))**2 + (x*(a1*e1 + c1*(1 - x*x/d1)))**2)
            H_max = max(H(0),H(wo))
            a2 = 3*c**2
            b2 = 2*c1*d1*(c1 + 2*a1*e1) - (c1*d1*e1)**2 - a1**2
            c2 = 2*c1*d1**2*e1*(b1 + a1) + (a1*d1*e1)**2 + (c1*d1)**2 - 2*a1*d1*b1
            deriv_roots2 = np.sqrt(np.roots([a2, b2, c2]))
            for root in deriv_roots2:
                if np.isreal(root):
                    H_max = max(H_max, H(root))
            # Find maximum magnitude of output impedance in [0, wo].
            rout = lambda x: (ro + rsource)/(gm*ro*Ao)*H(x)*np.sqrt((1+(x/w1)**2)(1+(x/w2)**2))
            rout_max = max(rout(0),rout(wo))
            a3 = 4*c1**2/d1**4
            b3 = c1**2*(3*e1*e1 + 5*f1) + 12*a1*c1*e1/d1 - 2(a1/d1)**2)/d1**2
            c3 = 3*(f1*(c1*e1)**2 - 4*a1*c1*e1*f1/d1 + f1*(a1/d1)**2 + 2*(c1/d1)**2)
            d3 = 2*c1*e1*f1*(a1 + b1) + f1*(a1*e1) + f1*c1**2 + 4*(c1*e1)**2 - 2*a1*(b1*f1 + 8*c1*e1)/d1 + (4*a1**2 - 2*b1**2)/d1**2
            e3 = 4*c1*e1*(a1 + b1) + 2*(a1*e1)**2 + 2*c1**2 - 4*a1*b1/d1 - b1**2*f1
            deriv_roots3 = np.sqrt(np.roots([a3, b3, c3, d3, e3]))
            for root in deriv_roots3:
                if np.isreal(root):
                    rout_max = max(rout_max, rout(root))
        # Check if parameters fit given bounds.
        fits_linereg, fits_loadreg = False, False
        linereg = H_max*delta_v_lnr / vref
        loadreg = rout_max*delta_i_ldr / vref
        if not asymptote:
            if linereg_max >= linereg:
                fits_linereg = True
                print("Line Regulation bound met.")
            if loadreg_max >= loadreg:
                fits_loadreg = True
                print("Load Regulation bound met.")
        if fits_linereg and fits_loadreg:
            psrr = gm*ro*A
            pm = 180 - 180/np.pi*(np.arctan(wo/wn) + np.arctan(wo/w1) + np.arctan(wo/w2))
            designs += [(Ao, w1, w2, psrr, pm, linereg, loadreg)]
            print("All bounds met.")
        else:
            print("Not all bounds met.")

    if designs == []:
        raise ValueError("FAIL. No solutions found.")
    else:
        final_op_amp = designs[len(designs)//2]
        final_design = dict(
            ibias=best_ibias
            Ao=final_op_amp[0]
            w1=final_op_amp[1]
            w2=final_op_amp[2]
            psrr=final_op_amp[3]
            pm=final_op_amp[4]
            linereg=final_op_amp[5]
            loadreg=final_op_amp[6])
        return final_design


def design_seriesN_reg_lti(db_n, sim_env,
        vg_res, vdd, vref, vb_n,
        cload, rload, rsource=0,
        psrr_min, pm_min, ibias_margin, Ao_margin,
        linereg_max, loadreg_max,
        delta_v_lnr, delta_i_ldr):
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
    sim_env = 'TT'
    nmos_spec = 'specs_mos_char/nch_w0d5_100nm.yaml'
    pmos_spec = 'specs_mos_char/pch_w0d5_100nm.yaml'
    intent = 'lvt'

    nch_db = get_db(nmos_spec, intent, interp_method=interp_method,
                    sim_env=sim_env)
    pch_db = get_db(pmos_spec, intent, interp_method=interp_method,
                    sim_env=sim_env)

    specs = dict(
        db_n=nch_db,
        db_p=pch_db,
        sim_env=sim_env,
        vg_res=0.01,
        rf_res=100,
        vdd=1.0,
        cpd=5e-15,
        cload=20e-15,
        rdc_min=1e3,
        fbw_min=5e9,
        pm_min=45,
        vb_n=0,
        vb_p=0
        )

    # amp_specs = design_seriesN_reg_eqn(**specs)
    amp_specs = design_seriesN_reg_lti(**specs)
    pprint.pprint(amp_specs)
    print('done')

if __name__ == '__main__':
    run_main()
