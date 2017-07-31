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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  default_random_engine gen;
  num_particles = 40;

  // define normal distributions for sensor noise
  normal_distribution<double> ndX (0, std[0]);
    normal_distribution<double> ndY (0, std[1]);
    normal_distribution<double> ndT (0, std[2]);

    // initialize particles
    for (int i = 0; i < num_particles; i++) {
      Particle p;
      p.id = i;
      p.x = x;
      p.y = y;
      p.theta = theta;
      p.weight = 0.5;

      // add noise
      // where "gen" is the random engine initialized 
      p.x += ndX (gen);
      p.y += ndY (gen);
      p.theta += ndT (gen);

      particles.push_back(p);
      weights.push_back(p.weight);

    }

    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  default_random_engine gen;

  normal_distribution<double> ndX(0, std_pos[0]);
    normal_distribution<double> ndY(0, std_pos[1]);
    normal_distribution<double> ndT(0, std_pos[2]);

    for (int i = 0; i < num_particles; i++) {

      if (fabs(yaw_rate) < 0.00001) {  
          particles[i].x += velocity * delta_t * cos(particles[i].theta);
          particles[i].y += velocity * delta_t * sin(particles[i].theta);
      } 
      else {
          particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
          particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
          particles[i].theta += yaw_rate * delta_t;
      }

      particles[i].x += ndX(gen);
      particles[i].y += ndY(gen);
      particles[i].theta += ndT(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  for (int i = 0; i < observations.size(); ++i) {
      LandmarkObs& o = observations[i];
      double minDist = numeric_limits<double>::max();
      
      for (int j = 0; j < predicted.size(); ++j) {
        LandmarkObs p = predicted[j];
        double distance = dist(o.x, o.y, p.x, p.y);
        if (distance < minDist) {
          minDist = distance;
          o.id = j;
        }
      }
    }


}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
    std::vector<LandmarkObs> observations, Map map_landmarks) {
    for (int i = 0; i < num_particles; i++) {

    Particle &p = particles[i];
    double p_x = p.x;
      double p_y = p.y;
      double p_theta = p.theta;

    vector<LandmarkObs> transformedOs;
    for (int j = 0; j < observations.size(); ++j) {
      LandmarkObs obs = observations[j];
      LandmarkObs transformedObs;
      transformedObs.x = obs.x * cos(p_theta) - obs.y * sin(p_theta) + p_x;
      transformedObs.y = obs.x * sin(p_theta) + obs.y * cos(p_theta) + p_y;

      transformedOs.push_back(transformedObs);
    }
    vector<LandmarkObs> predictions;
    for (int k = 0; k < map_landmarks.landmark_list.size(); ++k) {
        int lm_id = map_landmarks.landmark_list[k].id_i;
          float lm_x = map_landmarks.landmark_list[k].x_f;
          float lm_y = map_landmarks.landmark_list[k].y_f;

      if (dist(p_x, p_y, lm_x, lm_y) < sensor_range) {
        LandmarkObs pred = {lm_id, lm_x, lm_y};
        predictions.push_back(pred);
      }
    }

    dataAssociation(predictions, transformedOs);

    p.weight = 1;
    for (int l = 0; l < transformedOs.size(); ++l) {
      LandmarkObs obs = transformedOs[l];
      LandmarkObs pred = predictions[obs.id];

      double stdX = std_landmark[0]; // sigma_x
      double stdY = std_landmark[1]; // sigma_y
  
      double obs_w =  exp( -( pow(pred.x-obs.x,2)/(2*pow(stdX, 2)) + (pow(pred.y-obs.y,2)/(2*pow(stdY, 2))) ) )/(2*M_PI*stdX*stdY);

      p.weight *= obs_w;
    }
  
    weights[i] = p.weight;
  }

}

void ParticleFilter::resample() {
  
  vector<Particle> newParticles;
  default_random_engine gen;

    uniform_int_distribution<int> uniintdist(0, num_particles-1);
    auto index = uniintdist(gen);

    double max_weight = *max_element(weights.begin(), weights.end());

    uniform_real_distribution<double> unirealdist(0.0, max_weight);

    double beta = 0.0;

    for (int i = 0; i < num_particles; i++) {
      beta += unirealdist(gen) * 2.0;
      while (beta > weights[index]) {
          beta -= weights[index];
          index = (index + 1) % num_particles;
      }
      newParticles.push_back(particles[index]);
    }

    particles = newParticles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}