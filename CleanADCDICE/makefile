CC = gcc
CXX = g++
ARGS = -O3 -Wall -std=c++0x
OBJFILES  = ./src/CDICE.o ./src/FAIR/FAIR.o ./src/config/config.o \
	./src/dvars/dvars.o ./src/econ/econ.o
EMODPSOBJS = ./src/moeaframework/moeaframework.o ./src/emodps/ncRBF.o \
	./src/emodps/ann.o ./src/emodps/ann_mo.o ./src/emodps/param_function.o \
	./src/emodps/param_policy.o ./src/emodps/utils.o \
	./src/emodps/pwLinear.o ./src/emodps/rbf.o

all: ./ADCDICE2016

./ADCDICE2016: ${EMODPSOBJS} ./src/CDICE.o ./src/Main.cpp 
	${CXX} ${ARGS} ./src/Main.cpp ${OBJFILES} ${EMODPSOBJS} -o ./ADCDICE2016

./src/CDICE.o: ./src/CDICE.cpp ./src/FAIR/FAIR.o \
	./src/config/config.o ./src/dvars/dvars.o ./src/econ/econ.o
	${CXX} ${ARGS} -c ./src/CDICE.cpp -o ./src/CDICE.o

./src/config/config.o: ./src/config/config.cpp
	${CXX} ${ARGS} -c ./src/config/config.cpp -o ./src/config/config.o

./src/FAIR/FAIR.o: ./src/FAIR/FAIR.cpp
	${CXX} ${ARGS} -c ./src/FAIR/FAIR.cpp -o ./src/FAIR/FAIR.o

./src/dvars/dvars.o: ./src/dvars/dvars.cpp
	${CXX} ${ARGS} -c ./src/dvars/dvars.cpp -o ./src/dvars/dvars.o

./src/econ/econ.o: ./src/econ/econ.cpp
	${CXX} ${ARGS} -c ./src/econ/econ.cpp -o ./src/econ/econ.o

${EMODPSOBJS}: ./src/moeaframework/moeaframework.c ./src/emodps/ncRBF.cpp ./src/emodps/ann.cpp ./src/emodps/ann_mo.cpp ./src/emodps/param_function.cpp ./src/emodps/param_policy.cpp ./src/emodps/utils.cpp
	${CC} -O3 -c ./src/moeaframework/moeaframework.c -o ./src/moeaframework/moeaframework.o
	$(CXX) ${ARGS} -c ./src/emodps/utils.cpp -o ./src/emodps/utils.o
	$(CXX) ${ARGS} -c ./src/emodps/param_policy.cpp -o ./src/emodps/param_policy.o
	$(CXX) ${ARGS} -c ./src/emodps/param_function.cpp -o ./src/emodps/param_function.o
	$(CXX) ${ARGS} -c ./src/emodps/ncRBF.cpp -o ./src/emodps/ncRBF.o
	$(CXX) ${ARGS} -c ./src/emodps/pwLinear.cpp -o ./src/emodps/pwLinear.o
	$(CXX) ${ARGS} -c ./src/emodps/rbf.cpp -o ./src/emodps/rbf.o
	$(CXX) ${ARGS} -c ./src/emodps/ann.cpp -o ./src/emodps/ann.o
	$(CXX) ${ARGS} -c ./src/emodps/ann_mo.cpp -o ./src/emodps/ann_mo.o
	$(CXX) ${ARGS} -c ./src/emodps/param_function.cpp -o ./src/emodps/param_function.o

.PHONY: clean

clean:
	find . -name '*.o' -delete
	rm ADCDICE2016
