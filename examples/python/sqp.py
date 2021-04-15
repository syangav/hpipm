###################################################################################################
#                                                                                                 #
# This file is part of HPIPM.                                                                     #
#                                                                                                 #
# HPIPM -- High-Performance Interior Point Method.                                                #
# Copyright (C) 2019 by Gianluca Frison.                                                          #
# Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              #
# All rights reserved.                                                                            #
#                                                                                                 #
# The 2-Clause BSD License                                                                        #
#                                                                                                 #
# Redistribution and use in source and binary forms, with or without                              #
# modification, are permitted provided that the following conditions are met:                     #
#                                                                                                 #
# 1. Redistributions of source code must retain the above copyright notice, this                  #
#    list of conditions and the following disclaimer.                                             #
# 2. Redistributions in binary form must reproduce the above copyright notice,                    #
#    this list of conditions and the following disclaimer in the documentation                    #
#    and/or other materials provided with the distribution.                                       #
#                                                                                                 #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND                 #
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED                   #
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                          #
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR                 #
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES                  #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;                    #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND                     #
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT                      #
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS                   #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    #
#                                                                                                 #
# Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             #
#                                                                                                 #
###################################################################################################
from hpipm_python import *
from hpipm_python.common import *
import numpy as np
import time
import sys
import os



# check that env.sh has been run
env_run = os.getenv('ENV_RUN')
if env_run!='true':
	print('ERROR: env.sh has not been sourced! Before executing this example, run:')
	print('source env.sh')
	sys.exit(1)

travis_run = os.getenv('TRAVIS_RUN')

## solving optimal control problem with SQP

total_time = 10

# dim
N = 50
nx = 2
nu = 1

nbx = nx

dim = hpipm_ocp_qp_dim(N)

dim.set('nx', nx, 0, N) # number of states
dim.set('nu', nu, 0, N-1) # number of inputs
dim.set('nbx', nbx, 0) # number of state bounds

status = 0
debug = 1
delta_t = total_time/N

Jx = np.array([1, 0, 0, 1]).reshape(nbx,nx)

def odefunction(x,u):
	# input x of shape (nx,1)
	# input u of shape (nu,1)
	# output xdot=f(x,u) of shape (nx,1)
	x1 = x[0,0]
	x2 = x[1,0]
	u_scalar = u[0,0]
	f = np.array([(1-x2*x2)*x1-x2+u_scalar, x1 ]).reshape(nx,1)
	return f*delta_t + x

def qp_solution(x,u,pi):
	# input x should contain x_0, x_1, ..., x_N for the k-th interation of SQP
	# x_i of shape (nx,1)
	# input u should contain u_0, u_1, ..., u_N-1 for the k-th interation of SQP
	# u_i of shape (nu,1)
	# input pi should contain pi_0, ..., pi_N-1 for the k-th iteration of SQP
	# pi_i of shape (nx,1)
	qp = hpipm_ocp_qp(dim)

	[dim_x, n_x] = x.shape
	assert dim_x == nx
	assert n_x == N+1

	for k in range(N):
		x_k = x[:,k].reshape(nx,1)
		A_k = np.identity(nx) + delta_t*np.array([1-x_k[1,0]*x_k[1,0],-1-2*x_k[0,0]*x_k[1,0],1,0]).reshape(nx,nx)
		qp.set('A',A_k,k)
		x_k_1 = x[:,k+1].reshape(nx,1)
		u_k = u[:,k].reshape(nu,1)
		b_k = odefunction(x_k,u_k) - x_k_1
		qp.set('b',b_k,k)
		pi_k = pi[:,k].reshape(nx,1)
		# Q_k = np.identity(nx)+delta_t*np.array([0,-2*x_k[1,0]*pi_k[0,0],-2*x_k[1,0]*pi_k[0,0],-2*x_k[0,0]*pi_k[0,0]]).reshape(nx,nx)
		Q_k = np.identity(nx)
		qp.set('Q', Q_k, k)
		q_k = x_k
		qp.set('q', q_k, k)
		r_k = u_k
		qp.set('r', r_k, k)

	x_N = x[:,N].reshape(nx,1)
	# no need for A_N
	# A_N = np.identity(nx) + delta_t*np.array([1-x_N[1,0]*x_N[1,0],-1-2*x_N[0,0]*x_N[1,0],1,0]).reshape(nx,nx)
	# qp.set('A',A_N, N)
	Q_N = np.identity(nx)
	qp.set('Q', Q_N, N)
	q_N = x_N # last element 
	qp.set('q', q_N, N)

	B = delta_t*np.array([1, 0]).reshape(nx,nu)
	qp.set('B', B, 0, N-1)
	S = np.array([0, 0]).reshape(nu,nx)
	qp.set('S', S, 0, N-1)
	R = np.array([2]).reshape(nu,nu)
	qp.set('R', R, 0, N-1)

	x0 = np.array([0, 1]).reshape(nx,1) - x[:,0].reshape(nx,1)

	qp.set('Jx', Jx, 0)
	qp.set('lx', x0, 0)
	qp.set('ux', x0, 0)

	# qp sol
	qp_sol = hpipm_ocp_qp_sol(dim)

	# set up solver arg
	mode = 'speed'
	# create and set default arg based on mode
	arg = hpipm_ocp_qp_solver_arg(dim, mode)

	# create and set default arg based on mode
	arg.set('mu0', 1e4)
	arg.set('iter_max', 30)
	arg.set('tol_stat', 1e-4)
	arg.set('tol_eq', 1e-5)
	arg.set('tol_ineq', 1e-5)
	arg.set('tol_comp', 1e-5)
	arg.set('reg_prim', 1e-12)

	# set up solver
	solver = hpipm_ocp_qp_solver(dim, arg)

	# solve qp
	start_time = time.time()
	solver.solve(qp, qp_sol)
	end_time = time.time()

	delta_u = qp_sol.get('u', 0, N-1)
	delta_x = qp_sol.get('x', 0, N)
	new_pi = qp_sol.get('pi', 0, N-1)

	# qp_sol.print_C_struct()
	if debug:
		print('solve time {:e}'.format(end_time-start_time))
		print('delta u =')
		for i in range(N):
			print(delta_u[i])

		print('delta x =')
		for i in range(N+1):
			print(delta_x[i])

		print('new pi =')
		for i in range(N):
			print(new_pi[i])

	status = solver.get('status')
	if debug: 
		if status==0:
			print('\nsuccess!\n')
		else:
			print('\nSolution failed, solver returned status {0:1d}\n'.format(status))

	return delta_x, delta_u, new_pi

x = np.random.rand(nx,N+1)
u = np.random.rand(nu,N)
pi = np.random.rand(nx,N+1)

max_iter = 5
start_time = time.time()
for i in range(max_iter):
	delta_x, delta_u, new_pi = qp_solution(x,u,pi)
	delta_x = np.array(delta_x).reshape(N+1,nx).transpose()
	delta_u = np.array(delta_u).reshape(N,nu).transpose()
	new_pi = np.array(new_pi).reshape(N,nx).transpose()
	x += delta_x
	u += delta_u
	pi = new_pi
end_time = time.time()
print('solve time {:e}'.format(end_time-start_time))


print("optimal x")
print(x)

tgrid = [total_time/N*k for k in range(N+1)]
u = u.transpose()
u = np.insert(u, 0, np.nan, axis=0)
print("optimal u")
print(u)
import matplotlib.pyplot as plt
plt.figure(1)
plt.clf()
plt.plot(tgrid, x[0,:], '--')
plt.plot(tgrid, x[1,:], '-')
plt.step(tgrid, u, '-.')
plt.xlabel('t')
plt.legend(['x1','x2','u'])
plt.grid()
plt.show()

sys.exit(int(status))
