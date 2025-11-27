
#--------------------------------------------------------------------------------------------------------------------------------

UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
PLATFORM = PLATFORM_LINUX
endif
ifeq ($(UNAME), Darwin)
PLATFORM = PLATFORM_OSX
endif

ifeq ($(PLATFORM),PLATFORM_LINUX)
CXX       =  g++
CUDAROOT  =  /usr/local/cuda-11.8
CUDAINC   = -I$(CUDAROOT)/samples/common/inc
CUDALIB   = $(CUDAROOT)/lib64
CUDAINC2  = /home/markus/Downloads/cuda-samples-11.8/Common
GLUI      = /home/markus/Downloads/glui-2.37
GLINC     = -I$(GLUI)/include
GLLIB     = $(GLUI)/lib/libglui.a -lGL -lGLU -lglut -lGLEW
CUDAGEN   = -gencode=arch=compute_80,code=\"sm_80,compute_80\"
endif

ifeq ($(PLATFORM),PLATFORM_OSX)

# ALEX: Apparently on my Mac it's critical to have
#       export MACOSX_DEPLOYMENT_TARGET=10.7
#       done before building, otherwise clang complains about tr1/ headers and the like.

CXX       = /usr/bin/clang++
CUDAROOT  = /Developer/NVIDIA/CUDA-9.0
CUDAINC   = -I$(CUDAROOT)/samples/common/inc 
CUDALIB   = $(CUDAROOT)/lib64
GLINC     = 
GLLIB     = -framework OpenGL -framework GLUT -framework GLUI /usr/lib/libGLEW.a
CUDAGEN   = #-gencode arch=compute_12,code=\"sm_12,compute_12\"
endif


#--------------------------------------------------------------------------------------------------------------------------------

CCFLAGS   = -I. -Iinclude -m64 -O2 -I$(CUDAROOT)/include $(CUDAINC) -I$(CUDAINC2) -D$(PLATFORM) 
CXXFLAGS  = $(CCFLAGS) $(GLINC)
NVCC = $(CUDAROOT)/bin/nvcc -ccbin $(CXX)
.SUFFIXES: .o .c .cpp .cu


LIBOBJECTS  = gdrawing.o cpubundling.o gpubundling.o sparsematrix.o polyline.o utils.o
DEMOOBJECTS = main.o
TERMINALOBJECT = myBatch.o
DEMO	= cubu
TERMINAL = cubu_terminal
CUBULIB = libcubu.a




all: $(DEMO) $(TERMINAL) $(CUBULIB)

$(CUBULIB): $(LIBOBJECTS)
	ar -rv $(CUBULIB) $(LIBOBJECTS)

$(DEMO): $(CUBULIB) $(DEMOOBJECTS)
	$(CXX) -o $(DEMO) $(DEMOOBJECTS) $(CUBULIB) $(CXXFLAGS) -L$(CUDALIB) $(GLLIB) -L$(CUDAROOT)/lib -lcudart -lstdc++

$(TERMINAL): $(CUBULIB) $(TERMINALOBJECT)
	$(CXX) -o $(TERMINAL) $(TERMINALOBJECT) $(CUBULIB) $(CXXFLAGS) -L$(CUDALIB) $(GLLIB) -L$(CUDAROOT)/lib -lcudart -lstdc++

.cpp.o:
	$(CXX) -c $(CXXFLAGS)  -o $@ $<

.cu.o:
	$(NVCC) -c $(CCFLAGS) -use_fast_math $(CUDAGEN) --ptxas-options=-v --maxrregcount 50 -o $@ $<

clean:
	-@rm *.o $(DEMO) $(CUBULIB) $(TERMINAL) 2>/dev/null || true

