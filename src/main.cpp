/* Copyright 2016 Kristofer Björnson
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** @package TBTKtemp
 *  @file main.cpp
 *  @brief Self-consistent superconductivity using diagonalization
 *
 *  Basic example of self-consistent superconducting order-parameter for a 2D
 *  tight-binding model with t = 1, mu = -1, and V_sc = 2. Lattice with edges
 *  and a size of 20x20 sites.
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Array.h"
#include "TBTK/Model.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/TBTK.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"
#include "TBTK/Smooth.h"

#include <complex>
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace TBTK;
using namespace Visualization::MatPlotLib;

const complex<double> i(0, 1);

//Lattice size
const int SIZE_X = 30;
const int SIZE_Y = 30;

//Order parameter. The two buffers are alternatively swaped by setting
//deltaCounter = 0 or 1. One buffer contains the order parameter used in the
//previous calculation, while the other is used to store the newly obtained
//order parameter that will be used in the next calculation.
Array<complex<double>> Delta({SIZE_X, SIZE_Y});
unsigned int deltaCounter = 0;

//Superconducting pair potential, convergence limit, max iterations, and initial guess
const double V_sc = 2.;
const double CONVERGENCE_LIMIT = 0.000001;
const int MAX_ITERATIONS = 50;
const complex<double> DELTA_INITIAL_GUESS = 0.3 + 0.1*i;
const bool PERIODIC_BC = true;
const bool USE_GPU = true;


bool selfConsistencyStep(Solver::Diagonalizer solver){
	PropertyExtractor::Diagonalizer property_extractor(solver);
	Array<complex<double>> delta_old = Delta;
	//Clear the order parameter of the next step
	for(unsigned int x = 0; x < SIZE_X; x++)
		for(unsigned int y = 0; y < SIZE_Y; y++)
			Delta[{x, y}] = 0.;


	//Calculate new order parameter from gap equation for each site
	for(unsigned int x = 0; x < SIZE_X; x++){
		for(unsigned int y = 0; y < SIZE_Y; y++){
			for(unsigned spin = 0; spin < 2; ++spin){
				// Gap equation
				Delta[{x, y}] += 0.5*V_sc *(-1.+2*spin) *property_extractor.calculateExpectationValue(
					{x,y, (spin+1)%2, 1 },
					{x,y, spin, 0 }
				);
			}
		}
	}

	//Calculate convergence parameter
	double maxError = 0.;
	for(unsigned int x = 0; x < SIZE_X; x++){
		for(unsigned int y = 0; y < SIZE_Y; y++){
			double error = abs(Delta[{x,y}] - delta_old[{x,y}]);
			if(error > maxError)
				maxError = error;
		}
	}

	//Return true or false depending on whether the result has converged or not
	if(maxError < CONVERGENCE_LIMIT)
		return true;
	else
		return false;

}

//Callback function responsible for determining the value of the order
//parameter D_{to,from}c_{to}c_{from} where to and from are indices of the form
//(x, y, spin).
class DeltaCallback : public HoppingAmplitude::AmplitudeCallback{
	complex<double> getHoppingAmplitude(
		const Index &to,
		const Index &from
	) const{
		//Obtain indices
		unsigned int x = from[0];
		unsigned int y = from[1];
		unsigned int spin = from[2];
		unsigned int particleHole = from[3];

		if(spin == 0 && particleHole == 0)
			return conj(Delta[{x, y}]);
		else if(spin == 1 && particleHole == 0)
			return -conj(Delta[{x, y}]);
		else if(spin == 0 && particleHole == 1)
			return -Delta[{x, y}];
		else
			return Delta[{x, y}];
	}
} deltaCallback;

//Function responsible for initializing the order parameter
void initDelta(){
	const double rand_spread = 1.5*abs(DELTA_INITIAL_GUESS);
	srand (static_cast <unsigned> (time(0)));
	for(unsigned int x = 0; x < SIZE_X; x++){
		for(unsigned int y = 0; y < SIZE_Y; y++){
			double a = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
			double b = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
			Delta[{x, y}] = DELTA_INITIAL_GUESS + rand_spread*(a+i*b);
		}
	}
}

int main(int argc, char **argv){
	//Initialize TBTK.
	Initialize();

	//Parameters.
	complex<double> mu = -1.0;
	complex<double> t = 1.0;

	//Create model and set up hopping parameters.
	Model model;
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			for(int s = 0; s < 2; s++){
				//Add hopping ampltudes corresponding to
				//chemical potential.
				model << HoppingAmplitude(
					-mu,
					{x, y, s, 0},
					{x, y, s, 0}
				);
				model << HoppingAmplitude(
					mu,
					{x, y, s, 1},
					{x, y, s, 1}
				);

				//Add hopping parameters corresponding to t.
				if(x+1 < SIZE_X || PERIODIC_BC){
					model << HoppingAmplitude(
						-t,
						{(x+1)%SIZE_X, y, s, 0},
						{x, y, s, 0}
					) + HC;
					model << HoppingAmplitude(
						t,
						{(x+1)%SIZE_X, y, s, 1},
						{x, y, s, 1}
					) + HC;
				}
				if(y+1 < SIZE_Y || PERIODIC_BC){
					model << HoppingAmplitude(
						-t,
						{x, (y+1)%SIZE_Y, s, 0},
						{x, y, s, 0}
					) + HC;
					model << HoppingAmplitude(
						t,
						{x, (y+1)%SIZE_Y, s, 1},
						{x, y, s, 1}
					) + HC;
				}
				model << HoppingAmplitude(
					deltaCallback,
					{x, y, (s+1)%2, 1},
					{x, y, s, 0}
				) + HC;
			}
		}
	}

	//Construct model
	model.construct();

	//Initialize D
	initDelta();

	//Setup and run Solver::Diagonalizer
	Solver::Diagonalizer solver;
	solver.setModel(model);
	// solver.setMaxIterations(MAX_ITERATIONS);
	// solver.setSelfConsistencyCallback(selfConsistencyCallback);
	solver.setUseGPUAcceleration(USE_GPU);
	solver.run();

	// Selfconsistency loop
	for(int loop_counter = 0; loop_counter < MAX_ITERATIONS; ++loop_counter){
		cout << "Sc loop nr: " << loop_counter << endl;
		cout << Delta[{0,0}] << endl;
		if(selfConsistencyStep(solver)){
			break; // Exit loop if self consistency condition is achieved
		}
		solver.run();
	}

	//Calculate abs(D) and arg(D)
	Array<double> DeltaAbs({SIZE_X, SIZE_Y});
	Array<double> DeltaArg({SIZE_X, SIZE_Y});
	for(unsigned int x = 0; x < SIZE_X; x++){
		for(unsigned int y = 0; y < SIZE_Y; y++){
			DeltaAbs[{x, y}] = abs(Delta[{x, y}]);
			DeltaArg[{x, y}] = arg(Delta[{x, y}]);
		}
	}

	//Plot Delta.
	Plotter plotter;
	plotter.plot(DeltaAbs);
	plotter.save("../../figures/DeltaAbs.png");
	plotter.clear();
	plotter.plot(DeltaArg);
	plotter.save("../../figures/DeltaArg.png");
	plotter.clear();

	//Plot DOS
	//Set up the PropertyExtractor.
	const double LOWER_BOUND = -1;
	const double UPPER_BOUND = 1;
	const unsigned int RESOLUTION = 1000;
	PropertyExtractor::Diagonalizer propertyExtractor(solver);
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);
	Property::DOS dos = propertyExtractor.calculateDOS();
	//Smooth the DOS.
	const double SMOOTHING_SIGMA = 0.05;
	const unsigned int SMOOTHING_WINDOW = 201;
	dos = Smooth::gaussian(dos, SMOOTHING_SIGMA, SMOOTHING_WINDOW);
	plotter.plot(dos);
	plotter.save("../../figures/DOS.png");




	return 0;
}
