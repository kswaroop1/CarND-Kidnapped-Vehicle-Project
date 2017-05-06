/*
 * main.cpp
 * Reads in data and runs 2D particle filter.
 *  Created on: Dec 13, 2016
 *      Author: Tiffany Huang
 */

#include <iostream>
#include <ctime>
#include <iomanip>
#include <random>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

int main(int argc, char* argv[]) {
	// parameters related to grading.
	int time_steps_before_lock_required = 100;  // number of time steps before accuracy is checked by grader.
	double max_runtime = 45;                    // Max allowable runtime to pass [sec]
	double max_translation_error = 1;           // Max allowable translation error to pass [m]
	double max_yaw_error = 0.05;                // Max allowable yaw error [rad]

	int start = clock();      // Start timer.
	
	//Set up parameters here
	double delta_t = 0.1;     // Time elapsed between measurements [sec]
	double sensor_range = 50; // Sensor range [m]

	/*
	 * Sigmas - just an estimate, usually comes from uncertainty of sensor, but
	 * if you used fused data from multiple sensors, it's difficult to find
	 * these uncertainties directly.
	 */
	double sigma_pos [3] = {0.3, 0.3, 0.01};  // GPS measurement uncertainty [x [m], y [m], theta [rad]]
	double sigma_landmark [2] = {0.3, 0.3};   // Landmark measurement uncertainty [x [m], y [m]]

	// noise generation
	default_random_engine gen;
	normal_distribution<double> N_x_init(0, sigma_pos[0]), N_y_init(0, sigma_pos[1]);
	normal_distribution<double> N_theta_init(0, sigma_pos[2]), N_obs_x(0, sigma_landmark[0]), N_obs_y(0, sigma_landmark[1]);

	Map map;                                // Read map data
	if (!read_map_data("data/map_data.txt", map)) {
		cerr << "Error: Could not open map file" << endl;
		return -1;
	}
  int particleCount = map.landmark_list.size();
  if (argc > 1) particleCount = atoi(argv[1]);
	vector<control_s> position_meas;        // Read position data
	if (!read_control_data("data/control_data.txt", position_meas)) {
		cerr << "Error: Could not open position/control measurement file" << endl;
		return -1;
	}
	vector<ground_truth> gt;                // Read ground truth data
	if (!read_gt_data("data/gt_data.txt", gt)) {
		cerr << "Error: Could not open ground truth data file" << endl;
		return -1;
	}

	// Run particle filter!
	int num_time_steps = position_meas.size();
  ParticleFilter pf{ particleCount };
	double total_error[3] = {0,0,0};
	double cum_mean_error[3] = {0,0,0};

	for (int i = 0; i < num_time_steps; ++i) {
		cerr << "Time step: " << i << endl;

    ostringstream file;           // Read in landmark observations for current time step.
		file << "data/observation/observations_" << setfill('0') << setw(6) << i+1 << ".txt";
		vector<LandmarkObs> observations;
		if (!read_landmark_data(file.str(), observations)) {
			cerr << "Error: Could not open observation file " << i+1 << endl;
			return -1;
		}

		if (!pf.initialized()) {    // Initialize particle filter if this is the first time step.
			double n_x = N_x_init(gen), n_y = N_y_init(gen), n_theta = N_theta_init(gen);
			pf.init(gt[i].x + n_x, gt[i].y + n_y, gt[i].theta + n_theta, sigma_pos);
		}	else {                    // Predict the vehicle's next state (noiseless).
			pf.prediction(delta_t, sigma_pos, position_meas[i-1].velocity, position_meas[i-1].yawrate);
		}
		// simulate the addition of noise to noiseless observation data.
    vector<LandmarkObs> noisy_observations; noisy_observations.reserve(observations.size());
		for (const auto& o : observations) {
      noisy_observations.push_back(LandmarkObs{ o.id, o.x + N_obs_x(gen), o.y + N_obs_y(gen) });
		}

		// Update the weights and resample
		pf.updateWeights(sensor_range, sigma_landmark, noisy_observations, map);
		pf.resample();

		// Calculate and output the average weighted error of the particle filter over all time steps so far.
		vector<Particle> particles = pf.particles;
		int num_particles = particles.size();
		double highest_weight = 0.0;
		Particle best_particle;
		for (int i = 0; i < num_particles; ++i) {
			if (particles[i].weight > highest_weight) {
				highest_weight = particles[i].weight;
				best_particle = particles[i];
			}
		}
		double *avg_error = getError(gt[i].x, gt[i].y, gt[i].theta, best_particle.x, best_particle.y, best_particle.theta);

		for (int j = 0; j < 3; ++j) {
			total_error[j] += avg_error[j];
			cum_mean_error[j] = total_error[j] / (double)(i + 1);
		}

		// Print the cumulative weighted error
		cerr << "Cumulative mean weighted error: x " << cum_mean_error[0] << " y " << cum_mean_error[1] << " yaw " << cum_mean_error[2] << endl;
		if (i+1== num_time_steps)
      cout << "LAST Cumulative mean weighted error: x " << cum_mean_error[0] << " y " << cum_mean_error[1] << " yaw " << cum_mean_error[2] << " : #p " << particleCount << endl;

    // If the error is too high, say so and then exit.
		if (i >= time_steps_before_lock_required) {
			if (cum_mean_error[0] > max_translation_error || cum_mean_error[1] > max_translation_error || cum_mean_error[2] > max_yaw_error) {
				if (cum_mean_error[0] > max_translation_error) {
					cerr << "Your x error, " << cum_mean_error[0] << " is larger than the maximum allowable error, " << max_translation_error << endl;
				} else if (cum_mean_error[1] > max_translation_error) {
					cerr << "Your y error, " << cum_mean_error[1] << " is larger than the maximum allowable error, " << max_translation_error << endl;
				} else {
					cerr << "Your yaw error, " << cum_mean_error[2] << " is larger than the maximum allowable error, " << max_yaw_error << endl;
				}
				return -1;
			}
		}
	}

	// Output the runtime for the filter.
	int stop = clock();
	double runtime = (stop - start) / double(CLOCKS_PER_SEC);
	cout << "Runtime (sec): " << runtime << endl;

	// Print success if accuracy and runtime are sufficient (and this isn't just the starter code).
	if (runtime < max_runtime && pf.initialized()) {
		cout << "Success! Your particle filter passed!" << endl;
	} else if (!pf.initialized()) {
		cout << "This is the starter code. You haven't initialized your filter." << endl;
	} else {
		cout << "Your runtime " << runtime << " is larger than the maximum allowable runtime, " << max_runtime << endl;
		return -1;
	}

	return 0;
}


