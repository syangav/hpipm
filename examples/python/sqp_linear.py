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
debug = 0

# define flags
codegen_data = 1; # export qp data in the file ocp_qp_data.c for use from C examples

# dim
N = 5
nx = 2
nu = 1
nbx = nx

dim = hpipm_ocp_qp_dim(N)
dim.set('nx', nx, 0, N) # number of states
dim.set('nu', nu, 0, N-1) # number of inputs
dim.set('nbx', nbx, 0) # number of state bounds
if codegen_data:
	dim.codegen('gululu.c', 'w')

Q = np.array([1, 0, 0, 1]).reshape(nx,nx)
R = np.array([1]).reshape(nu,nu)

def get_cost(x,u):
	cost = 0.0
	for k in range(N):
		x_k = x[:,k].reshape(nx,1)
		u_k = u[:,k].reshape(nu,1)
		cost += np.transpose(x_k).dot(Q).dot(x_k)
		cost += np.transpose(u_k).dot(R).dot(u_k)
	x_N = x[:,N]
	cost += np.transpose(x_N).dot(Q).dot(x_N)
	return cost

def qp_solution(x,u,pi):
	A = np.zeros((2,2))
	A[0][0] = 1.0
	A[0][1] = 1.0
	A[1][1] = 1.0
	B = np.array([0, 1]).reshape(nx,nu)
	qp = hpipm_ocp_qp(dim)

	for k in range(N):
		x_k = x[:,k].reshape(nx,1)
		x_k_1 = x[:,k+1].reshape(nx,1)
		u_k = u[:,k].reshape(nu,1)
		b_k = A.dot(x_k) + B.dot(u_k) - x_k_1
		qp.set('b', b_k, k)
		q_k = x_k.reshape(nx,1)
		qp.set('q', q_k, k)

	q_N = x[:,N].reshape(nx,1) # last element 
	qp.set('q', q_N, N)
	S = np.array([0, 0]).reshape(nu,nx)
	r = np.array([0]).reshape(nu,nu)

	Jx = np.array([1, 0, 0, 1]).reshape(nbx,nx)
	x0 = np.array([1, 1]).reshape(nx,1) - x[:,0].reshape(nx,1)
	# qp

	qp.set('A', A, 0, N-1)
	qp.set('B', B, 0, N-1)

	qp.set('Q', Q, 0, N)
	qp.set('S', S, 0, N-1)
	qp.set('R', R, 0, N-1)
	qp.set('r', r, 0, N-1)
	qp.set('Jx', Jx, 0)
	qp.set('lx', x0, 0)
	qp.set('ux', x0, 0)
	if codegen_data:
		qp.codegen('gululu.c', 'a')

	# qp sol
	qp_sol = hpipm_ocp_qp_sol(dim)


	# set up solver arg
	#mode = 'speed_abs'
	mode = 'speed'
	#mode = 'balance'
	#mode = 'robust'
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
	if codegen_data:
		arg.codegen('ocp_qp_data.c', 'a')
	# set up solver
	solver = hpipm_ocp_qp_solver(dim, arg)


	# solve qp
	solver.solve(qp, qp_sol)

	delta_u = qp_sol.get('u', 0, N-1)
	delta_x = qp_sol.get('x', 0, N)
	new_pi = qp_sol.get('pi', 0, N-1)

	if debug:
		print('delta_u =')
		for i in range(N):
			print(delta_u[i])
		print('delta_x =')
		for i in range(N+1):
			print(delta_x[i])


	delta_u = np.array(delta_u).reshape(N,nu).transpose()
	delta_x = np.array(delta_x).reshape(N+1,nx).transpose()
	new_pi = np.array(new_pi).reshape(N,nx).transpose()

	status = solver.get('status')

	# if status==0:
	# 	print('success!')
	# else:
	# 	print('Solution failed, solver returned status {0:1d}\n'.format(status))

	return delta_x, delta_u, new_pi

x = np.zeros((nx,N+1))
u = np.zeros((nu,N))

# x = np.random.rand(nx,N+1)
# u = np.random.rand(nu,N)

pi = np.ones((nx,N)) # does not matter 
max_iter = 10
for i in range(max_iter):
	delta_x, delta_u, new_pi = qp_solution(x,u,pi)
	x += delta_x
	u += delta_u
	pi = new_pi
	cost = get_cost(x,u)
	print("cost of iteration ", i, " is: ", cost)


print(x)
print(u)
print('new pi =')
print(new_pi)

# sys.exit(int(status))
