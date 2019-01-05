#xc=FX10
xc=IMPI
#xc=MVAPICH2
#xc=INTEL
FALANX_TOP=$(HOME)/local

CPP = gcc -E

xcOK = 0

xcCUDA = KEPLER
#xcCUDA = VOLTA  # not work...
#xcCUDA = PASCAL # not work...
xcCUDA = 0

xcPROF = 0
xcMIC = 0

xcTLOG = 0

CUDA_DEFS0 =
DEFS0 = -DDEBUG -DOFMO_PROF # -DDEBUG_SORT
DEFS0 += $(TARGET_DEFS)


DEFS = $(DEFS0)
DEFS += -DSORT_CSP
#DEFS += -DSORT_CSP_SCHWARZ
#DEFS += -DFMT_ORG
#DEFS += -DPARA_SUB
#DEFS += -DMINIMUM_SCF=2

CUDA_DEFS  = $(CUDA_DEFS0)
CUDA_DEFS += -DGPU_DLB
CUDA_DEFS += -DDLB_KL
CUDA_DEFS += -DSORT_IJ_SCHWARZ
#CUDA_DEFS += -DSORT_KL_SCHWARZ
#CUDA_DEFS += -DUSE_ATOMIC
#CUDA_DEFS += -DUSE_INSTANT_SCHWARZ
#CUDA_DEFS += -DADD_FULL_NAO


USERINC = -I$(HOME)/local/include
USERLIB = -L$(HOME)/local/lib

# -------------
xcOK = 0
xcMPI = 0
xcOPENMP = 1

ifeq ($(xc), OpenMPI)
    MPICC = mpicc
    xcMPI = 1
    xc = INTEL
endif

ifeq ($(xc), IMPI)
    MPICC = mpiicc
    xcMPI = 1
    xc = INTEL
endif

ifeq ($(xc), MVAPICH2)
    MPICC = mpicc
    xcMPI = 1
    xc = INTEL
endif

ifeq ($(xc), INTEL)
    CC = icc
    FC = ifort
    COPT = -O3 -no-prec-div -no-prec-sqrt -fno-alias
    COPT0 = -O0
    CFLAGS = -g -Wall
    C99FLAGS = -std=c99
    FFLAGS = -warn -g -O3 -no-prec-div -no-prec-sqrt -extend-source 132
ifeq ($(xcMIC), 1)
    CFLAGS += -mmic
    FFLAGS += -mmic
else
    CFLAGS += -xHOST
    #CFLAGS += -xCORE-AVX512 # SKL
    #CFLAGS += -xCORE-AVX2   # HSW/BDW
    #CFLAGS += -xCORE-AVX-I   # IVB
endif
    OPENMP = -qopenmp
    PROF = -pg
    LIBS = -mkl -limf # -lifcore#-lefence
    xcOK = 1
endif

ifeq ($(xc), GNU)
    CC = gcc
    FC = gfortran
    CFLAGS = -g #-m64
    COPT = -O3
    COPT0 = -O0
    C99FLAGS = -std=c99
    FFLAGS = -O3 -ffixed-line-length-132 -ffast-math
    OPENMP = -fopenmp
    PROF = -pg
    ACML = /home/umeda/opt/acml4.4.0/gfortran64_mp/
    LIBS = -L$(ACML)/lib -lacml_mp -lacml_mv -lgfortran -lm
    LIBS = $(ACML)/lib/libacml_mp.a -lgfortran -lm
    xcOK = 1
endif

ifeq ($(xc), FX10)
    CC = mpifccpx
    MPICC = mpifccpx
    FC = frtpx
    OPENMP = -Kopenmp
    COPT = -Kfast
    COPT0 = -O0
    CFLAGS = -DFJ_MAIN -Xa -noansi
    FFLAGS = -Kfast -Fwide
    LIBS = -SSL2BLAMP
    xcOK = 1
endif

ifeq ($(xc), ASC)
    CC = mpicc
    MPICC = mpicc
    FC = gfortran
    CFLAGS = -O0 -g -fopenmp -std=c99 -Wall
    CFLAGS0 = -O0 -g -fopenmp -std=c99 -Wall
    FLAGS0 = -O0 -g -fopenmp -std=c99 -Wall
    FFLAGS = -fopenmp -ffixed-line-length-132
    LIBS = -L/usr/lib64/atlas -llapack -lcblas -lm -lz
    xcOK = 1
    xcMPI = 1
endif

ifeq ($(xc), AGC)
    CC = mpicc
    MPICC = mpicc
    FC = ifort
    OMPI_F77 = ifort
    OMPI_FC = ifort
    OMPI_CC = icc
    export OMPI_CC OMPI_FC OMPI_F77
    CFLAGS = -g -O3 -xhost -openmp -std=c99 -mkl -Wall
    CFLAGS0 = -g -O0 -g -openmp -std=c99 -mkl -Wall
    FFLAGS = -fopenmp -ffixed-line-length-132 -mkl
    FLAGS0 = -g -O0 -fopenmp -std=c99 -mkl -Wall
    LIBS = -mkl -lm -lz
    xcOK = 1
    xcMPI = 1
endif

# C compiler with nvcc
ifeq ($(xcCUDA), 0)
  xcCUDA=
endif
ifdef xcCUDA
    CC_CC = gcc
    CC_CFLAGS = -g -O3
    CC_OPENMP = -fopenmp
endif

ifeq ($(xcMPI), 0)
#	@echo requires MPI
	xcOK = 1
endif

ifeq ($(xcOK), 0)
	@echo invalid xc value $(xc).
	exit 1
endif

# -------------
ifeq ($(xcMPI), 1)
CC = $(MPICC)
MPI_DEFS = -DUSE_MPI
#MPI_INCDIR = $(patsubst %/bin/mpicc,%/include, $(shell which mpicc))
MPIB = $(shell basename $(MPICC))
MPI_INCDIR = $(shell which $(MPICC))
MPI_INCDIR := $(patsubst %/bin/$(MPIB),%/include, $(MPI_INCDIR))
MPI_INCDIR := $(patsubst %/bin64/$(MPIB),%/include64, $(MPI_INCDIR))
endif
LD = $(CC)

#DEFS = -DOFMO_PROF -DDEBUG_MODE

INCDIR = -I../basis -I../master -I.. -I../scf -I../integ -I../matope -I../mserv \
	 -I../common -I../worker -I../include
INCDIR += ${USERINC}
LIBS   += ${USERLIB}

ifeq ($(xcMPI), 1)
  DEFS += $(MPI_DEFS)
endif
ifeq ($(xcOPENMP), 1)
  CFLAGS += $(OPENMP)
endif
ifeq ($(xcPROF), 1)
  CFLAGS += ${PROF}
endif

ifeq ($(xcTLOG), 1)
  CFLAGS += -DUSE_TLOG
  LIBS += -ltlog
endif
ifeq ($(xcTLOG), 2)
  CFLAGS += -DUSE_TLOG -DUSE_MPE
  LIBS += -lmpilog
endif

CFLAGS += $(DEFS) ${INCDIR}
CFLAGS += -I$(FALANX_TOP)/include

CFLAGS0 := $(CFLAGS) $(COPT0)
#CFLAGS  := $(CFLAGS) $(COPT)
#CFLAGS0 = $(CFLAGS) $(COPT0)
CFLAGS  += $(COPT)

LIBS += -lz
LIBS_FALANX = $(shell pkg-config --silence-errors --libs kyotocabinet)
LIBS_FALANX += $(FALANX_TOP)/lib/libfalanx_ds.a


#  CUDA
# -------------
ifdef xcCUDA

DEFS1 = -DUSE_CUDA
# ARCH
DEFS1 += -D$(xcCUDA)
ifeq ($(xcCUDA), KEPLER)
ARCHFLAGS = -gencode=arch=compute_35,code=sm_35
DEFS1 += -DCUDA_ARCH=350
endif
ifeq ($(xcCUDA), PASCAL)
ARCHFLAGS = -gencode=arch=compute_60,code=sm_60
DEFS1 += -DCUDA_ARCH=600
endif
ifeq ($(xcCUDA), VOLTA)
ARCHFLAGS = -gencode=arch=compute_70,code=sm_70
DEFS1 += -DCUDA_ARCH=700
endif

# FMT
ifeq ($(xcCUDA), FERMI)
#xcCUDA_FMT_M = 3s
#xcCUDA_FMT_M_NEXP = 8
xcCUDA_FMT_M = 1
xcCUDA_FMT_M_NEXP = 6
else
xcCUDA_FMT_M = 1
xcCUDA_FMT_M_NEXP = 6
#xcCUDA_FMT_M_K1 = 3 # special setting for Kepler, m=1
endif
ifdef xcCUDA_FMT_M
ifeq ($(xcCUDA_FMT_M), 3s)
DEFS1 += -DCUDA_FMT_M_SM
xcCUDA_FMT_M = 3
endif
DEFS1 += -DCUDA_FMT_M=$(xcCUDA_FMT_M)
DEFS1 += -DCUDA_FMT_M_NEXP=$(xcCUDA_FMT_M_NEXP)
ifdef xcCUDA_FMT_M_K1
DEFS1 += -DCUDA_FMT_M_K1
endif
endif

ifeq ($(xcPROF), 1)
  PROF_FLAGS = -pg -g -G
endif

CC_CFLAGS += $(DEFS) $(DEFS1)
CUDA_DEFS += $(DEFS1)
ifeq ($(xcMPI), 1)
  CC_CFLAGS += $(MPI_DEFS) -I$(MPI_INCDIR)
endif
ifeq ($(xcOPENMP), 1)
  CC_CFLAGS += ${CC_OPENMP}
endif
CFLAGS  += $(DEFS1)
CFLAGS0 += $(DEFS1)
NVCC_LD := nvcc $(ARCHFLAGS) --compiler-bindir="$(CC_CC)"
NVCC := nvcc $(ARCHFLAGS) --compiler-bindir="$(CC_CC)"
#MPICC = $(CC)
#NVCC_CFLAGS := $(MPI_DEFS) -O3 --compiler-options="$(CC_CFLAGS)" -Xptxas=-v $(PROF_FLAGS) $(CUDA_DEFS)
NVCC_CFLAGS := $(MPI_DEFS) -O3 --compiler-options="$(CC_CFLAGS)" -Xptxas=-v -lineinfo $(PROF_FLAGS) $(CUDA_DEFS)
NVCC_LIBS = --compiler-options="$(LIBS)"
ifdef CUDALIBPATH
CUDALIBS = -L$(CUDALIBPATH) -lcudart
else
CUDALIBS = -lcudart
endif

endif # CUDA
CFLAGS += $(C99FLAGS)
CFLAGS0 += $(C99FLAGS)
# -------------

ifeq ($(LD), ld)
  LD = ${MPICC}
endif
LDFLAGS = ${CFLAGS}
SHELL = /bin/sh


export

VPATH = ./basis:./matope:./integ:./scf:./master:./worker:./mserv:./common

OBJS_TOP0 = common/ofmo-string.o common/ofmo-compress.o common/ofmo-input.o \
	    common/ofmo-data.o common/ofmo-init.o common/ofmo-misc.o \
	    common/ofmo-prof.o common/ofmo-debug.o common/ofmo-fragment-basis.o \
	    common/ofmo-fragment.o \
	    common/log_mpe.o

OBJS_BASIS = basis/ofmo-basis.o basis/ofmo-basis-database.o

OBJS_MAT = matope/ofmo-mat.o

OBJS_SCF = scf/ofmo-diis.o scf/ofmo-scf.o

OBJS_INTEG = integ/ofmo-cutoff-core-dddd.o \
	     integ/ofmo-cutoff-core-ssss-dpdp.o \
    integ/ofmo-cutoff.o integ/ofmo-ifc2c-core.o integ/ofmo-ifc2c.o \
    integ/ofmo-ifc3c-os.o integ/ofmo-ifc4c-os.o integ/ofmo-integ.o \
    integ/ofmo-oneint-core.o integ/ofmo-oneint.o \
    integ/ofmo-twoint-buffer.o integ/ofmo-twoint-core-dddd.o \
    integ/ofmo-twoint-core-dddp.o integ/ofmo-twoint-core-ddds.o \
    integ/ofmo-twoint-core-ddss-ddpp.o integ/ofmo-twoint-core-dpds-dpdp.o \
    integ/ofmo-twoint-core-dsds-dppp.o integ/ofmo-twoint-core-ssss-dspp.o \
    integ/ofmo-twoint-direct.o integ/ofmo-twoint.o integ/fmt.o \
    integ/fmt-m.o integ/fmt-method1.o integ/fmt-method2.o integ/fmt-method3.o \
    integ/ofmo-index.o integ/ofmo-os-xxxx.o integ/ofmo-oneint-gen.o \
    integ/ofmo-rys-xxxx.o integ/ofmo-root.o \
    integ/ofmo-rys-psss.o integ/ofmo-rys-psps.o integ/ofmo-rys-ppss.o \
    integ/ofmo-rys-ppps.o integ/ofmo-rys-pppp.o integ/ofmo-rys-dsss.o \
    integ/ofmo-rys-dsps.o integ/ofmo-rys-dspp.o integ/ofmo-rys-dsds.o \
    integ/ofmo-rys-dpss.o integ/ofmo-rys-dpps.o integ/ofmo-rys-dppp.o \
    integ/ofmo-rys-dpds.o integ/ofmo-rys-dpdp.o integ/ofmo-rys-ddss.o \
    integ/ofmo-rys-ddps.o integ/ofmo-rys-ddpp.o integ/ofmo-rys-ddds.o \
    integ/ofmo-rys-dddp.o integ/ofmo-rys-dddd.o \
    integ/ofmo-os-psss.o integ/ofmo-os-psps.o integ/ofmo-os-ppss.o \
    integ/ofmo-os-ppps.o integ/ofmo-os-pppp.o integ/ofmo-os-dsss.o \
    integ/ofmo-os-dsps.o integ/ofmo-os-dspp.o integ/ofmo-os-dsds.o \
    integ/ofmo-os-dpss.o integ/ofmo-os-dpps.o integ/ofmo-os-dppp.o \
    integ/ofmo-os-dpds.o integ/ofmo-os-dpdp.o integ/ofmo-os-ddss.o \
    integ/ofmo-os-ddps.o integ/ofmo-os-ddpp.o integ/ofmo-os-ddds.o \
    integ/ofmo-os-dddp.o integ/ofmo-os-dddd.o \
    integ/ofmo-ifc4c-rys.o integ/ofmo-ifc3c-rys.o integ/ofmo-ifc4c.o

#OBJS_INTEG = integ/ofmo-cutoff-core-dddd.o \
#	     integ/ofmo-cutoff-core-ssss-dpdp.o \
#	integ/ofmo-cutoff.o integ/ofmo-ifc2c-core.o integ/ofmo-ifc2c.o \
#	integ/ofmo-ifc3c-os.o integ/ofmo-ifc4c-os.o integ/ofmo-integ.o \
#	integ/ofmo-oneint-core.o integ/ofmo-oneint.o \
#	integ/ofmo-twoint-buffer.o integ/ofmo-twoint-core-dddd.o \
#	integ/ofmo-twoint-core-dddp.o integ/ofmo-twoint-core-ddds.o \
#	integ/ofmo-twoint-core-ddss-ddpp.o \
#	integ/ofmo-twoint-core-dpds-dpdp.o \
#	integ/ofmo-twoint-core-dsds-dppp.o \
#	integ/ofmo-twoint-core-ssss-dspp.o \
#	integ/ofmo-twoint-direct.o integ/ofmo-twoint.o integ/fmt.o \
#	integ/ofmo-core-ssss.o integ/fmt4.o integ/fmt4-gen.o \
#	integ/ofmo-core-psss.o integ/ofmo-core-psps.o \
#	integ/ofmo-core-ppss.o integ/ofmo-core-ppps.o \
#	integ/ofmo-core-pppp.o integ/ofmo-core-dsss.o integ/f-cons.o \
#	integ/ofmo-core-dsps.o integ/ofmo-core-dspp.o \
#	integ/ofmo-core-dsds.o \
#	integ/ofmo-core-dpss.o integ/ofmo-core-dpps.o \
#	integ/ofmo-core-dppp.o \
#	integ/ofmo-core-dpds.o integ/ofmo-core-dpdp.o \
#	integ/ofmo-core-ddss.o \
#	integ/ofmo-core-ddps.o integ/ofmo-core-ddpp.o \
#	integ/ofmo-core-ddds.o \
#	integ/ofmo-core-dddp.o integ/ofmo-core-dddd.o \
#	integ/ofmo-twoint-buffer-f.o integ/ofmo-ifc4c-f.o \
#	integ/ofmo-ifc3c-f.o integ/ofmo-ifc2c-f.o \
#	integ/ofmo-ifc2c-core-f.o \
#	integ/ofmo-oneint-core-f.o integ/ofmo-oneint-f.o

ifdef xcCUDA
 OBJS_CUDA = cuda/cuda.o cuda/cuda-drv.o cuda/cuda-fmt-drv.o
endif


target_dirs=basis matope integ scf common

OBJS_FALANX	= falanx/ofmo-falanx-main.o	\
			falanx/ofmo-monomer-data.o	\
			master/ofmo-data-struct.o	\
			worker/ofmo-inter-frag.o	\
			worker/ofmo-calc-frag.o		\
			worker/ofmo-counter.o		\
			worker/ofmo-projection.o	\
			worker/ofmo-init-dens.o		\
			falanx/ofmo-storage.o		\
			falanx/ofmo-mserv-cont.o	\
			falanx/ofmo-datastore.o		\
			falanx/ofmo-task-util.o		\
			falanx/ofmo-worker-task.o	\
			${OBJS_TOP0}	\
			${OBJS_BASIS}	\
			${OBJS_MAT}		\
			${OBJS_SCF}		\
			${OBJS_INTEG}	\
			${OBJS_CUDA}
PROG_FALANX = ofmo-falanx

OBJS_MASTER = master/ofmo-master-main.o master/ofmo-data-struct.o \
	  master/ofmo-put-get-master.o ${OBJS_TOP0} ${OBJS_BASIS}
PROG_MASTER = ofmo-master

OBJS_WORKER = worker/ofmo-worker-main.o worker/ofmo-mserv-cont.o \
	  worker/ofmo-init-dens.o worker/ofmo-inter-frag.o \
	  worker/ofmo-monomer-data.o worker/ofmo-projection.o \
	  worker/ofmo-calc-frag.o worker/ofmo-counter.o \
	  ${OBJS_TOP0} ${OBJS_BASIS} ${OBJS_MAT} \
	  ${OBJS_SCF} ${OBJS_INTEG} \
          ${OBJS_CUDA}
PROG_WORKER = ofmo-worker

OBJS_MSERV = mserv/ofmo-mserv-main.o mserv/ofmo-worker-cont.o \
	 ${OBJS_TOP0} ${OBJS_BASIS}
PROG_MSERV = ofmo-mserv


OBJS_RHF = rhf/skel-rhf-main.o rhf/skel-rhf-data.o rhf/skel-rhf-calc.o \
           rhf/skel-w2e.o \
           common/ofmo-string.o common/ofmo-misc.o common/ofmo-prof.o \
           ${OBJS_BASIS} ${OBJS_MAT} ${OBJS_SCF} ${OBJS_INTEG} \
           ${OBJS_CUDA}
PROG_RHF = skel-rhf

all : help

help :
	@echo "usage: make <target>"
	@echo ""
	@echo "	original	build ofmo-master, ofmo-worker, ofmo-mserv."
	@echo "	falanx		build ofmo-falanx."
	@echo "	rhf		build skel-rhf."
	@echo "	clean		remove build files."
	@echo "	help		print this message."
	@echo ""


original: TARGET_DEFS=
original : master mserv worker

master : master_compile
	${LD} -o ${PROG_MASTER} ${LDFLAGS} ${OBJS_MASTER} ${LIBS}

mserv : mserv_compile
	${LD} -o ${PROG_MSERV} ${LDFLAGS} ${OBJS_MSERV} ${LIBS}

worker : worker_compile
	${LD} -o ${PROG_WORKER} ${LDFLAGS} ${OBJS_WORKER} ${CUDALIBS} ${LIBS}

falanx: TARGET_DEFS=-DUSE_FALANX
falanx : falanx_compile
	${LD} -o ${PROG_FALANX} ${LDFLAGS} ${OBJS_FALANX} ${LIBS_FALANX} ${CUDALIBS} ${LIBS}

rhf: TARGET_DEFS=-DOFMO_SKELETON
rhf : rhf_compile
	${LD} -o ${PROG_RHF} ${LDFLAGS} ${OBJS_RHF} ${CUDALIBS} ${LIBS}

dir_mk :
	for dir in ${target_dirs}; do \
		(cd $$dir ; ${MAKE} -f Makefile) ; \
	done
ifdef xcCUDA
	for dir in cuda; do \
		(cd $$dir ; ${MAKE} -f Makefile CC=$(MPICC)) ; \
	done
endif

master_compile mserv_compile worker_compile : dir_mk
	_target=`echo $@ | sed s/_compile//`; \
	(cd $$_target ; ${MAKE} -f Makefile)

falanx_compile : master_compile worker_compile
	_target=`echo $@ | sed s/_compile//`; \
	(cd $$_target ; ${MAKE} -f Makefile)

rhf_compile : dir_mk
	_target=`echo $@ | sed s/_compile//`; \
	(cd $$_target ; ${MAKE} -f Makefile)

clean: clean_mk clean_cuda clean_prog clean_scr

clean_mk :
	_target_dirs="${target_dirs} master mserv worker falanx rhf";\
	for dir in $${_target_dirs}; do \
		(cd $$dir ; ${MAKE} -f Makefile clean) ; \
	done
	-rm -f *.dat

clean_prog :
	-rm -f ${PROG_MASTER} ${PROG_WORKER} ${PROG_MSERV} ${PROG_FALANX} ${PROG_RHF}

clean_cuda :
ifdef xcCUDA
	for dir in cuda; do \
		(cd $$dir ; ${MAKE} -f Makefile clean) ; \
	done
endif

clean_scr:
	-rm -f scr/*

