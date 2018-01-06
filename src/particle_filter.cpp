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
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void test(Particle &particle);

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 20;
  default_random_engine gen;
  double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];

  // This line creates a normal (Gaussian) distribution for x
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);


  for (int i = 0; i < num_particles; ++i) {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;
    particles.push_back(particle);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine gen;
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);
  for (auto &particle:particles) {
    if (yaw_rate > 0.0001) {
      particle.x = particle.x + velocity / yaw_rate * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
      particle.y = particle.y + velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
      particle.theta = particle.theta + delta_t * yaw_rate;
    } else {
      particle.x = particle.x + velocity * delta_t * cos(particle.theta);
      particle.y = particle.y + velocity * delta_t * sin(particle.theta);
      particle.theta = particle.theta + delta_t * yaw_rate;
    }
    particle.x = particle.x + dist_x(gen);
    particle.y = particle.y + dist_y(gen);
    particle.theta = particle.theta + dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> &transformed_observations,
                                     const std::vector<Map::single_landmark_s> &landmarks) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.
  for (auto &obs:transformed_observations) {
    int id = landmarks[0].id_i;
    double min_dist = hypot(obs.x - landmarks[0].x_f, obs.y - landmarks[0].y_f);
    for (auto &landmark: landmarks) {
      double distance = hypot(obs.x - landmark.x_f, obs.y - landmark.y_f);
      if (distance < min_dist) {
        id = landmark.id_i;
        min_dist = distance;
      }
    }
    obs.id = id;
//    std::cout << "transform obs: x: " << obs.x << " y: " << obs.y << "\n";
//    std::cout << "asso id:" << id << " x: " << landmarks[id - 1].x_f << " y: " << landmarks[id - 1].y_f << "\n";

  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html
  for (auto &particle: particles) {
//    std::cout << "particle: x: " << particle.x << " y: " << particle.y << "\n";
    std::vector<LandmarkObs> transformed_observations;
    for (int i = 0; i < observations.size(); i++) {
      LandmarkObs origin = observations[i];
//      std::cout << "obs: x: " << origin.x << " y: " << origin.y << "\n";

      LandmarkObs obs;
      obs.x = particle.x + cos(particle.theta) * origin.x - sin(particle.theta) * origin.y;
      obs.y = particle.y + sin(particle.theta) * origin.x + cos(particle.theta) * origin.y;
      transformed_observations.push_back(obs);
    }

    dataAssociation(transformed_observations, map_landmarks.landmark_list);
    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;
    for (auto &obs:transformed_observations) {
      associations.push_back(obs.id);
      sense_x.push_back(obs.x);
      sense_y.push_back(obs.y);
    }
    SetAssociations(particle, associations, sense_x, sense_y);
    double w = 1;
    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
    for (int i = 0; i < transformed_observations.size(); i++) {
      int landmark_id = transformed_observations[i].id - 1;
      double gauss_norm = (1 / (2 * M_PI * std_x * std_y));

      double exponent =
          -pow((transformed_observations[i].x - map_landmarks.landmark_list[landmark_id].x_f), 2) / (2 * std_x * std_x)
          - pow(transformed_observations[i].y - map_landmarks.landmark_list[landmark_id].y_f, 2) / (2 * std_y * std_y);

//      std::cout << "gauss_norm: " << gauss_norm << " exponent: " << exponent << "\n";
      double ww = gauss_norm * exp(exponent);
      w = w * ww;
    }
//    std::cout << "w: " << w << "\n";
    particle.weight = w;
  }
}

void test(Particle &particle) {

}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  weights.clear();
  for (auto &particle : particles) {
    weights.push_back(particle.weight);
  }
  std::discrete_distribution<int> dist(weights.begin(), weights.end());
  default_random_engine engine;
  std::vector<Particle> p;
  for (int i = 0; i < num_particles; i++) {
    int index = dist(engine);
    p.push_back(particles[index]);
  }
  particles = p;
}

Particle ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
                                         const std::vector<double> &sense_x, const std::vector<double> &sense_y) {
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
