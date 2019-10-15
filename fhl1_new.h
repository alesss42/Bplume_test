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
#define SPLINES_VDIFF
#define SPLINES_VVISC
#undef T_PASSIVE
#define FLOATS

#undef RADIATION_2D

#define ANA_SMFLUX
#define ANA_SRFLUX
#define ANA_SSFLUX
#define ANA_STFLUX
#define ANA_BSFLUX
#define ANA_BTFLUX
#define ANA_FSOBC
#define ANA_M2OBC
#define ANA_SPFLUX
#define ANA_BPFLUX

#undef SPONGE
#ifdef SPONGE
# define MIX_S_TS
# define MIX_S_UV
# undef VISC_GRID
# undef DIFF_GRID
# define UV_VIS2
# define TS_DIF2
#endif


#define GLS_MIXING
#if defined GLS_MIXING || defined MY25_MIXING
# undef CANUTO_A
# define N2S2_HORAVG
# define RI_SPLINES
# define KANTHA_CLAYSON
#endif

#define SEDIMENT
#ifdef SEDIMENT
# define SUSPLOAD
# define SED_DENS
#endif


#define DIAGNOSTICS_UV
#define DIAGNOSTICS_TS
#define AVERAGES
#undef AVERAGES_AKV
#undef AVERAGES_AKS

#undef  SG_BBL
#ifdef SG_BBL
# undef  SG_CALC_ZNOT
# undef  SG_LOGINT
#endif

#undef  MB_BBL
#ifdef MB_BBL
# undef  MB_CALC_ZNOT
# undef  MB_Z0BIO
# undef  MB_Z0BL
# undef  MB_Z0RIP
#endif

#undef SSW_BBL
#ifdef SSW_BBL
# define SSW_CALC_ZNOT
# undef  SSW_LOGINT
#endif




