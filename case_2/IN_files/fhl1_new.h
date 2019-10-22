/*
** svn $Id: estuary_test.h 939 2019-01-28 07:02:47Z arango $
*******************************************************************************
** Copyright (c) 2002-2019 The ROMS/TOMS Group                               **
**   Licensed under a MIT/X style license                                    **
**   See License_ROMS.txt                                                    **
*******************************************************************************
**
** Options for Estuary with Sediment Transport Test.
**
** Application flag:   ESTUARY_TEST
** Input script:       roms_estuary_test.in
**                     sediment_estuary_test.in
** Modified by ASR 2019 for Beta_plume case 1
*/

#define UV_LOGDRAG
#define UV_ADV
#define UV_COR


#define DJ_GRADPS
#define TS_MPDATA
#undef NONLIN_EOS
#define SALINITY
#define MASKING
#define SOLVE3D
#define SPLINES_VDIFF /* THIS IS NEW 2019 I THINK IT SUBSTITUTES SPLINES */
#define SPLINES_VVISC /* THIS IS NEW 2019 I THINK IT SUBSTITUTES SPLINES */
#undef T_PASSIVE
#undef FLOATS

#undef RADIATION_2D

#define ANA_SMFLUX
#define ANA_SRFLUX
#define ANA_SSFLUX
#define ANA_STFLUX
#define ANA_BSFLUX
#define ANA_BTFLUX
#define ANA_VMIX /* We added this to remove vertical mixing */
#define ANA_FSOBC
#define ANA_M2OBC
#define ANA_SPFLUX
#define ANA_BPFLUX


#define GLS_MIXING
#if defined GLS_MIXING 
# undef CANUTO_A
# define N2S2_HORAVG
# define RI_SPLINES
# define KANTHA_CLAYSON
#endif



