/**************************************************************************************************
*                                                                                                 *
* This file is part of HPIPM.                                                                     *
*                                                                                                 *
* HPIPM -- High Performance Interior Point Method.                                                *
* Copyright (C) 2017 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* HPMPC is free software; you can redistribute it and/or                                          *
* modify it under the terms of the GNU Lesser General Public                                      *
* License as published by the Free Software Foundation; either                                    *
* version 2.1 of the License, or (at your option) any later version.                              *
*                                                                                                 *
* HPMPC is distributed in the hope that it will be useful,                                        *
* but WITHOUT ANY WARRANTY; without even the implied warranty of                                  *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                                            *
* See the GNU Lesser General Public License for more details.                                     *
*                                                                                                 *
* You should have received a copy of the GNU Lesser General Public                                *
* License along with HPMPC; if not, write to the Free Software                                    *
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA                  *
*                                                                                                 *
* Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/



#if defined(RUNTIME_CHECKS)
#include <stdlib.h>
#include <stdio.h>
#endif

#include <blasfeo_target.h>
#include <blasfeo_common.h>
#include <blasfeo_s_aux.h>

#include "../include/hpipm_s_ocp_qp_dim.h"
#include "../include/hpipm_s_ocp_qp.h"
#include "../include/hpipm_s_ocp_qp_sol.h"



#define CREATE_STRVEC s_create_strvec
#define CVT_STRVEC2VEC s_cvt_strvec2vec
#define CVT_VEC2STRVEC s_cvt_vec2strvec
#define OCP_QP s_ocp_qp
#define OCP_QP_DIM s_ocp_qp_dim
#define OCP_QP_SOL s_ocp_qp_sol
#define REAL float
#define STRVEC s_strvec
#define SIZE_STRVEC s_size_strvec
#define VECCP_LIBSTR sveccp_libstr

#define CREATE_OCP_QP_SOL s_create_ocp_qp_sol
#define MEMSIZE_OCP_QP_SOL s_memsize_ocp_qp_sol
#define CVT_OCP_QP_SOL_TO_COLMAJ s_cvt_ocp_qp_sol_to_colmaj
#define CVT_COLMAJ_TO_OCP_QP_SOL s_cvt_colmaj_to_ocp_qp_sol
#define CVT_OCP_QP_SOL_TO_ROWMAJ s_cvt_ocp_qp_sol_to_rowmaj
#define CVT_OCP_QP_SOL_TO_LIBSTR s_cvt_ocp_qp_sol_to_libstr



#include "x_ocp_qp_sol.c"

