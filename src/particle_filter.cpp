/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 100;
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(0, std[0]), dist_y(0, std[1]), dist_theta(0, std[2]);
  particles.reserve(num_particles);
  weights.reserve(num_particles);
  for (int i = 0; i < num_particles; ++i) {
    Particle p{ i, x + dist_x(gen), y + dist_y(gen), theta + dist_theta(gen), 1.0 };
    particles.push_back(p);
    weights.push_back(0.0);
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(0, std_pos[0]), dist_y(0, std_pos[1]), dist_theta(0, std_pos[2]);  

  for (int i = 0; i < num_particles; ++i) {
    auto theta0 = particles[i].theta;
    if (fabs(yaw_rate) < 0.0001) {
      particles[i].x += velocity*delta_t*cos(theta0);
      particles[i].y += velocity*delta_t*sin(theta0);
    } else {
      auto theta_inc = delta_t * yaw_rate;
      particles[i].theta += theta_inc;
      auto theta1 = particles[i].theta;
      particles[i].x += velocity / yaw_rate*(sin(theta1) - sin(theta0));
      particles[i].y += velocity / yaw_rate*(cos(theta0) - cos(theta1));
    }
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  if (predicted.size() < 1) return;
  for (size_t l = 0; l < observations.size(); ++l) {
    int nearest = 0;
    auto nearest_distance = dist(observations[l].x, observations[l].y, predicted[nearest].x, predicted[nearest].y);
    for (size_t p = 1; p < predicted.size(); ++p) {
      auto d1 = dist(observations[l].x, observations[l].y, predicted[p].x, predicted[p].y);
      if (d1 < nearest_distance) {
        nearest_distance = d1;
        nearest = p;
      }
    }
    observations[l].id = predicted[nearest].id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
  std::vector<LandmarkObs> landmarks;
  auto neta = 1.0 / (2*M_PI*std_landmark[0]*std_landmark[1]);
  double deno[2] = { 2 * std_landmark[0] * std_landmark[0], 2 * std_landmark[1] * std_landmark[1] };
  for (const auto& l : map_landmarks.landmark_list)
    landmarks.push_back(LandmarkObs{ l.id_i, l.x_f, l.y_f });
  for (auto p = 0; p < num_particles; ++p) {
    auto cosT = cos(particles[p].theta);
    auto sinT = sin(particles[p].theta);
    std::vector<LandmarkObs> t_obs; t_obs.reserve(observations.size());
    weights[p] = 1.0;
    for (const auto& o : observations) {
      t_obs.push_back(LandmarkObs{ o.id, o.x * cosT + o.y * sinT + particles[p].x, o.x * sinT + o.y * cosT + particles[p].y });
    }
    dataAssociation(landmarks, t_obs);
    auto landmark_count = 0;
    for (const auto& o : t_obs) {
      auto xe = o.x - landmarks[o.id].x; xe *= xe;
      auto ye = o.y - landmarks[o.id].y; ye *= ye;
      if (sqrt(xe + ye) < sensor_range) {
        weights[p] *= neta*exp(-(xe / deno[0] + ye / deno[1]));
        ++landmark_count;
      }
    }
    if (landmark_count == 0) weights[p] = 0; // no landmarks in sensor range
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::default_random_engine gen;
  std::uniform_int_distribution<> dist_k(0, num_particles-1);

  auto wm = *std::max_element(weights.begin(), weights.end());
  std::uniform_real_distribution<double> dist_wt(0, wm*2.0);
  auto beta = 0.0;
  std::vector<Particle> p3; p3.reserve(num_particles);
  for (auto k = 0; k < num_particles; ++k) {
    auto i = dist_k(gen);
    beta += dist_wt(gen);
    while (weights[i] < beta) {
      beta -= weights[i];
      i = (i + 1) % num_particles;
    }
    p3.push_back(particles[i]);
  }
  particles = p3;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
