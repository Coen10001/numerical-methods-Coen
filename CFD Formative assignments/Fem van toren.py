# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:15:46 2026

@author: gijsb
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sympy import *

var("t x_s z_s phi_s phi_t")      
# x_s substructure surge, z_s heave, phi_s roll, phi_t tower angle (to vertical)
x_s = Function("x_s")(t)
z_s = Function("z_s")(t)
phi_s = Function("phi_s")(t)
phi_t = Function("phi_t")(t)

# Define the kinematic relations here

# contants needed
var("x_st z_st L_t")
# dx_st hotizontal distance substructure centre to tower base
# dz_st vertica distance, L_t tower length

# positions of nacelle
x_t = x_s + x_st*cos(phi_s) - z_st*sin(phi_s) - L_t*sin(phi_t)
z_t = z_s + x_st*sin(phi_s) + z_st*cos(phi_s) + L_t*cos(phi_t)

# for energy relations later
phi_st = phi_s - phi_t

# Compute/define the velocities here
x_s_dot = diff(x_s, t)
z_s_dot = diff(z_s, t)
phi_s_dot = diff(phi_s, t)
x_t_dot = diff(x_t, t)
z_t_dot = diff(z_t, t)
phi_t_dot = diff(phi_t, t) # will not be sued for point masses
phi_st_dot = diff(phi_st, t) # velocity will not be used

var("rho_s B_s H_s W_s m_t")
# substructure density, breadth, height, width; turbine mass

# Define the kinetic energy here (T)
m_s = rho_s*B_s*H_s*W_s
J_s = 1/12*m_s*(B_s**2 + H_s**2)
T_s = 1/2*m_s*(x_s_dot**2 + z_s_dot**2) + 1/2*J_s*(phi_s_dot**2)
T_t = 1/2*m_t*(x_t_dot**2 + z_t_dot**2) + 1/2*0*(phi_t_dot**2) # point mass so J_t = 0
T = T_s+T_t

var("rho_w g kr_st k_h")
# water density, rotational spring stiffness
# Define the potential energy here (V)

k_s = rho_w*g*B_s*W_s # Approximate hydrostatic restoring stiffness (linearized buoyancy)

draft_s = (B_s*W_s*H_s*rho_s)/(B_s*W_s*rho_w)
KB_s = draft_s/2 # COB, due to constant shape
nabla_s = B_s*W_s*draft_s # Submerged volume, taken in neutral position
J_sub = 1/12*W_s*B_s**3
BM_s = J_sub/nabla_s
KG_s = H_s/2 # COG, due to uniform weight
GM_s = KB_s + BM_s - KG_s
kr_s = rho_w*g*nabla_s*GM_s # Nm/rad

V_s = m_s*g*z_s + 1/2*k_h*x_s**2 + 1/2*k_s*z_s**2 + 1/2*kr_s*phi_s**2
V_t = m_t*g*z_t + 1/2*kr_st*phi_st**2 # need relative angle for this spring
V = V_s + V_t

F_wave = Function("F_wave")(t)
M_wave = Function("M_wave")(t)
F_wind = Function("F_wind")(t)

# Define the generalized forces here (Qi)
q = [x_s, z_s, phi_s, phi_t] 
Q = [] 
for qi in q: 
    Q_i = ( 
        F_wind*diff(x_t, qi) + 
        F_wave*diff(z_s, qi) + 
        M_wave*diff(phi_s, qi) 
        ) 
    Q.append(Q_i)
    
# Define your Lagrangian here (L)
L = T - V

# Compute the EOM here
EOM_x_s = diff( diff(L, x_s_dot), t) - diff(L, x_s) - Q[0]
EOM_z_s = diff( diff(L, z_s_dot), t) - diff(L, z_s) - Q[1]
EOM_phi_s = diff( diff(L, phi_s_dot), t) - diff(L, phi_s) - Q[2]
EOM_phi_t = diff( diff(L, phi_t_dot), t) - diff(L, phi_t) - Q[3]

# dictionaries for substitution
var("x_s_0 x_s_epsilon")
tmp1_x_s = symbols("tmp1_x_s")
psi_x_s = Function("psi_x_s")(t) # perturbation function
tmp2_x_s = symbols("tmp2_x_s")

var("z_s_0 z_s_epsilon")
psi_z_s = Function("psi_z_s")(t) # perturbation function
tmp1_z_s = symbols("tmp1_z_s")
tmp2_z_s = symbols("tmp2_z_s")

var("phi_s_0 phi_s_epsilon")
psi_phi_s = Function("psi_phi_s")(t) # perturbation function
tmp1_phi_s = symbols("tmp1_phi_s")
tmp2_phi_s = symbols("tmp2_phi_s")

var("phi_t_0 phi_t_epsilon")
psi_phi_t = Function("psi_phi_t")(t) # perturbation function
tmp1_phi_t = symbols("tmp1_phi_t")
tmp2_phi_t = symbols("tmp2_phi_t")

subs1_dict = {x_s: tmp1_x_s, z_s: tmp1_z_s, phi_s: tmp1_phi_s, phi_t: tmp1_phi_t}
subs2_dict = {tmp1_x_s: x_s_0 + x_s_epsilon*psi_x_s,
              tmp1_z_s: z_s_0 + z_s_epsilon*psi_z_s,
              tmp1_phi_s: phi_s_0 + phi_s_epsilon*psi_phi_s,
              tmp1_phi_t: phi_t_0 + phi_t_epsilon*psi_phi_t}
subs3_dict = {diff(x_s, (t, 2)): tmp2_x_s, x_s: tmp1_x_s,
              diff(z_s, (t, 2)): tmp2_z_s, z_s: tmp1_z_s,
              diff(phi_s, (t, 2)): tmp2_phi_s, phi_s: tmp1_phi_s,
              diff(phi_t, (t, 2)): tmp2_phi_t, phi_t: tmp1_phi_t}
subs4_dict = {tmp2_x_s: diff(x_s_0 + x_s_epsilon*psi_x_s, (t, 2)), tmp1_x_s: x_s_0 + x_s_epsilon*psi_x_s,
              tmp2_z_s: diff(z_s_0 + z_s_epsilon*psi_z_s, (t, 2)), tmp1_z_s: z_s_0 + z_s_epsilon*psi_z_s,
              tmp2_phi_s: diff(phi_s_0 + phi_s_epsilon*psi_phi_s, (t, 2)), tmp1_phi_s: phi_s_0 + phi_s_epsilon*psi_phi_s,
              tmp2_phi_t: diff(phi_t_0 + phi_t_epsilon*psi_phi_t, (t, 2)), tmp1_phi_t: phi_t_0 + phi_t_epsilon*psi_phi_t}
epsilons_dict = {x_s_epsilon: 1, z_s_epsilon: 1, phi_s_epsilon: 1, phi_t_epsilon: 1}


startpos_dict = {x_s_0: 0, z_s_0: 0, phi_s_0: 0, phi_t_0: 0}

# x_s
EOM_psi_x_s = EOM_x_s.evalf(subs=subs1_dict)
EOM_psi_x_s = EOM_psi_x_s.evalf(subs=subs2_dict)
EOM_lin_x_s = series(EOM_psi_x_s, x_s_epsilon, n=2)

EOM_psi2_x_s = EOM_x_s.evalf(subs=subs3_dict)
EOM_psi2_x_s = EOM_psi2_x_s.evalf(subs=subs4_dict)
EOM_lin_x_s = series(EOM_psi2_x_s, x_s_epsilon, n=2)

EOM_lin_x_s = EOM_lin_x_s.removeO().evalf(subs=epsilons_dict)
EOM_lin_x_s_simplified = EOM_lin_x_s.evalf(subs=startpos_dict) # makes symbolic calcs easier
EOM_lin_x_s_iso = solve(EOM_lin_x_s_simplified, diff(psi_x_s, (t, 2)))

x_s_dotdot = EOM_lin_x_s_iso[0].evalf(subs=startpos_dict)

# z_s
EOM_psi_z_s = EOM_z_s.evalf(subs=subs1_dict)
EOM_psi_z_s = EOM_psi_z_s.evalf(subs=subs2_dict)
EOM_lin_z_s = series(EOM_psi_z_s, z_s_epsilon, n=2)

EOM_psi2_z_s = EOM_z_s.evalf(subs=subs3_dict)
EOM_psi2_z_s = EOM_psi2_z_s.evalf(subs=subs4_dict)
EOM_lin_z_s = series(EOM_psi2_z_s, z_s_epsilon, n=2)

EOM_lin_z_s = EOM_lin_z_s.removeO().evalf(subs=epsilons_dict)
EOM_lin_z_s_simplified = EOM_lin_z_s.evalf(subs=startpos_dict) # makes symbolic calcs easier
EOM_lin_z_s_iso = solve(EOM_lin_z_s_simplified, diff(psi_z_s, (t, 2)))

z_s_dotdot = EOM_lin_z_s_iso[0].evalf(subs=startpos_dict)

# phi_s

EOM_psi_phi_s = EOM_phi_s.evalf(subs=subs1_dict)
EOM_psi_phi_s = EOM_psi_phi_s.evalf(subs=subs2_dict)
EOM_lin_phi_s = series(EOM_psi_phi_s, phi_s_epsilon, n=2)

EOM_psi2_phi_s = EOM_phi_s.evalf(subs=subs3_dict)
EOM_psi2_phi_s = EOM_psi2_phi_s.evalf(subs=subs4_dict)
EOM_lin_phi_s = series(EOM_psi2_phi_s, phi_s_epsilon, n=2)

EOM_lin_phi_s = EOM_lin_phi_s.removeO().evalf(subs=epsilons_dict)
EOM_lin_phi_s_simplified = EOM_lin_phi_s.evalf(subs=startpos_dict) # makes symbolic calcs easier
EOM_lin_phi_s_iso = solve(EOM_lin_phi_s_simplified, diff(psi_phi_s, (t, 2)))

phi_s_dotdot = EOM_lin_phi_s_iso[0].evalf(subs=startpos_dict)

# phi_t

EOM_psi_phi_t = EOM_phi_t.evalf(subs=subs1_dict)
EOM_psi_phi_t = EOM_psi_phi_t.evalf(subs=subs2_dict)
EOM_lin_phi_t = series(EOM_psi_phi_t, phi_t_epsilon, n=2)

EOM_psi2_phi_t = EOM_phi_t.evalf(subs=subs3_dict)
EOM_psi2_phi_t = EOM_psi2_phi_t.evalf(subs=subs4_dict)
EOM_lin_phi_t = series(EOM_psi2_phi_t, phi_t_epsilon, n=2)

EOM_lin_phi_t = EOM_lin_phi_t.removeO().evalf(subs=epsilons_dict)
EOM_lin_phi_t_simplified = EOM_lin_phi_t.evalf(subs=startpos_dict) # makes symbolic calcs easier
EOM_lin_phi_t_iso = solve(EOM_lin_phi_t_simplified, diff(psi_phi_t, (t, 2)))

phi_t_dotdot = EOM_lin_phi_t_iso[0].evalf(subs=startpos_dict)

var("acc1 acc2 acc3 acc4 vel1 vel2 vel3 vel4")

dict_values = { Derivative(psi_x_s, (t,2)): acc1,
                Derivative(psi_z_s, (t,2)): acc2,
                Derivative(psi_phi_s, (t,2)): acc3,
                Derivative(psi_phi_t, (t,2)): acc4,
                Derivative(psi_x_s, t): vel1,
                Derivative(psi_z_s, t): vel2,
                Derivative(psi_phi_s, t): vel3,
                Derivative(psi_phi_t, t): vel4}

EOM_1 = EOM_lin_x_s_simplified.evalf(subs=dict_values)
EOM_2 = EOM_lin_z_s_simplified.evalf(subs=dict_values)
EOM_3 = EOM_lin_phi_s_simplified.evalf(subs=dict_values)
EOM_4 = EOM_lin_phi_t_simplified.evalf(subs=dict_values)

MTRX = linear_eq_to_matrix([EOM_1, EOM_2, EOM_3, EOM_4],
                           [acc1, acc2, acc3, acc4])
# Note: The results per line are the same af from for example EOM_lin_phi_t_iso

M = MTRX[0]
F = MTRX[1]

print( M )

print( F )

# Filling in values

#Floater (has to come from other code):
L = 50 #[m]
B = 50 #[m]
H = 10 #[m]
rho_ship = 500 #[kg/m3]
k_anchor = 10e6 #[N/m]

#tower:
L_tower = 81.5 #[m]
M_COM = 350e3 #[kg]
k_r = 15e9 #[Nm/rad]
x_r = 1 #x-position of rotational spring 'r' w.r.t. rigid body centre
z_r = 1 #z-position of rotational spring 'r' w.r.t. rigid body centre
rho_water = 1025 #[kg/m3]
g_const = 9.81 #[m/s^2]

def F_h(t):
    return (1.25*cos(0.1*t) + 0.5*cos(0.5*t - 0.75) + 0.1*cos(1.3*t + 2.28))*10e-6

# Connecting above values to existing parameters
dict_values = {B_s: L, H_s: H, W_s: B, L_t: L_tower,
                m_t: M_COM, z_st: z_r, x_st: x_r,
               rho_s: rho_ship, rho_w: rho_water, g: g_const,
               k_h: k_anchor, kr_st: k_r, F_wind: 0, M_wave: 0}

# Creating filled Mass matrix with given values
M_filled = M.evalf(subs=dict_values)
display(M_filled)

# Creating filled force matrix with given values
F_filled = F.evalf(subs=dict_values)
display(F_filled)


#pre-check to see if all values have been filled
#print(M.free_symbols)
#print(F.free_symbols)

# Initial conditions
q0 = np.zeros(8)

def qdot(t,q):

    vt = q[int(len(q)/2):]

    F_x_s = F_h(t)

    dict_values2 ={psi_x_s: q[0], psi_z_s: q[1], psi_phi_s: q[2], psi_phi_t: q[3],
                   vel1: q[4], vel2: q[5], vel3: q[6], vel4: q[7],
                   F_wave: F_x_s}
    Mass_matr =  M_filled.evalf(subs=dict_values2)
    Force_vect = F_filled.evalf(subs=dict_values2)
    at = Mass_matr.inv()*Force_vect

    return list(vt) + list(np.transpose(at)[0])


#solve ivp
sol = solve_ivp(fun=qdot,t_span=[0,100],y0=q0); #solve from 0 to 100s


# Plot
x_s = sol.y[0]
phi_s = sol.y[2]
phi_t = sol.y[3]
move_nacelle = x_s + x_r*np.cos(phi_s) - z_r*np.sin(phi_s) - L_tower*np.sin(phi_t) - x_r

plt.plot(sol.t,move_nacelle, label="Nacelle movement at 3 frequencies")
plt.xlabel("Time [s]")
plt.ylabel("Excursion from initial position [m]")
plt.title("Nacelle from Lagrangian equations")
plt.legend();