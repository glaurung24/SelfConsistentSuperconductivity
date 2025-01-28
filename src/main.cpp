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
 *  and a size of 20x20 sites. Additionally, a Zeeman field can be set via Vz and Rashba-soc via alpha_x and alpha_y.
 *
 *  @author Andreas Theiler
 */

#include "TBTK/Array.h"
#include "TBTK/Model.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/TBTK.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"
#include "TBTK/Smooth.h"
#include "TBTK/Timer.h"

#include <complex>
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace TBTK;
using namespace Visualization::MatPlotLib;

const complex<double> i(0, 1);

//Lattice size
const int SIZE_X = 40;
const int SIZE_Y = 10;

//Order parameter. 
Array<complex<double>> Delta({SIZE_X, SIZE_Y});

//Superconducting pair potential, convergence limit, max iterations, and initial guess
// and other Parameters.
const complex<double> mu = 0.0;
const complex<double> t_x = 0.25;
const complex<double> t_y = 0.5;
const double V_sc = 3.2;
const double alpha_x = 0.4321; //SOC in x
const double alpha_y = 0.0;	//SOC in y
const double Vz = 1.00; // Zeeman field strength
const int MAX_ITERATIONS = 50;
const double CONVERGENCE_LIMIT = 0.000001;
const complex<double> DELTA_INITIAL_GUESS = 0.3 + 0.0*i;
const double DELTA_INITIAL_GUESS_RANDOM_WINDOW = 0.0;
const bool PERIODIC_BC_X = false;
const bool PERIODIC_BC_Y = false;
const bool SELF_CONSISTENCY = false;
const bool USE_GPU = true;
const bool USE_MULTI_GPU = false;


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
				Delta[{x, y}] += 0.5*V_sc *(-1.+2*spin) * property_extractor.calculateExpectationValue(
					{x,y, (spin+1)%2, 1 },
					{x,y, spin, 0 }
				);;
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
	const double rand_spread = DELTA_INITIAL_GUESS_RANDOM_WINDOW*abs(DELTA_INITIAL_GUESS);
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
	Timer::tick("One execution");
	//Create model and set up hopping parameters.
	Model model;
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			for(int s = 0; s < 2; s++){
				for(int ph = 0; ph < 2; ++ph){ // ph: particle-hole index 0: particle, 1: hole
					//Add hopping ampltudes corresponding to
					//chemical potential.
					double tau_z = (1-2*ph);
					model << HoppingAmplitude(
						-mu*tau_z,
						{x, y, s, ph},
						{x, y, s, ph}
					);
					// Zeeman term
					model << HoppingAmplitude(
						Vz*tau_z*(1-2*s),
						{x, y, s, ph},
						{x, y, s, ph}
					);
					//Add hopping parameters corresponding to t.
					if(x+1 < SIZE_X || PERIODIC_BC_X){
						model << HoppingAmplitude(
							-tau_z*t_x,
							{(x+1)%SIZE_X, y, s, ph},
							{x, y, s, ph}
						) + HC;
					}
					if(y+1 < SIZE_Y || PERIODIC_BC_Y){
						model << HoppingAmplitude(
							-tau_z*t_y,
							{x, (y+1)%SIZE_Y, s, ph},
							{x, y, s, ph}
						) + HC;

					}
				}
				// Order parameter is set by a function called deltaCallback
				model << HoppingAmplitude(
					deltaCallback,
					{x, y, (s+1)%2, 1},
					{x, y, s, 0}
				) + HC;
				// SOC in x direction
				if(x+1 < SIZE_X || PERIODIC_BC_X){
					model << HoppingAmplitude(
							alpha_x, 
							{(x+1)%SIZE_X,y,(s+1)%2, 0}, 
							{x,y,s, 0}
						) + HC;
					model << HoppingAmplitude(
							-alpha_x,
							{x,y,s, 1},
							{(x+1)%SIZE_X,y,(s+1)%2, 1}
						) + HC;
				}
				// SOC in y direction
				if(y+1 < SIZE_Y || PERIODIC_BC_Y){
					model << HoppingAmplitude(
							i*alpha_y, 
							{x,(y+1)%SIZE_Y,(s+1)%2, 0}, 
							{x,y,s, 0}
						) + HC;
					model << HoppingAmplitude(
							-i*alpha_y, 
							{x,y,s, 1},
							{x,(y+1)%SIZE_Y,(s+1)%2, 1}
						) + HC;
				}
			}
		}
	}

	//Construct model
	model.construct();

	if(!model.isHermitian()){
		cout << "Model not Hermitian"  << endl;
		exit(0);
	}

	//Initialize D
	initDelta();

	//Setup and run Solver::Diagonalizer
	Solver::Diagonalizer solver;
	solver.setModel(model);
	solver.setUseGPUAcceleration(USE_GPU);
	solver.setUseMultiGPUAcceleration(USE_MULTI_GPU);
	solver.setVerbose(false);
	Streams::out << model.getBasisSize() << endl;
	solver.run();
	Timer::tock();

	if(SELF_CONSISTENCY){
		// Selfconsistency loop
		for(int loop_counter = 0; loop_counter < MAX_ITERATIONS; ++loop_counter){
			cout << "Sc loop nr: " << loop_counter << endl;
			cout << Delta[{0,0}] << endl;
			if(selfConsistencyStep(solver)){
				break; // Exit loop if self consistency condition is achieved
			}
			solver.run();
		}
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
	plotter.clear();

	Property::EigenValues ev = propertyExtractor.getEigenValues();
	plotter.plot(ev.getData());
	plotter.save("../../figures/Eigenvalues.png");
	plotter.clear();




	return 0;
}
