#pragma once
#include <set>
#include <yaml-cpp/yaml.h>
#include "G1Constants.hpp"

namespace teleop {
inline Joints joint_vector(const YAML::Node& node,const std::string& name,bool positive=false){
    auto values=node.as<std::vector<double>>();
    if(values.size()!=29||!std::all_of(values.begin(),values.end(),[positive](double x){return std::isfinite(x)&&(!positive||x>0.);}))throw std::runtime_error("Invalid 29-joint parameter: "+name);
    return Eigen::Map<const Joints>(values.data());
}
struct PolicyParams { Joints defaults,kp,kd; };
inline PolicyParams vq_params(const YAML::Node& cfg){return {joint_vector(cfg["g1"]["default_dof_pos"],"VQ defaults"),joint_vector(cfg["g1"]["kp"],"VQ kp",true),joint_vector(cfg["g1"]["kd"],"VQ kd",true)};}
inline PolicyParams wbc_params(const YAML::Node& cfg){
    const auto names=cfg["joint_names"].as<std::vector<std::string>>();
    if(names!=std::vector<std::string>(motor_names.begin(),motor_names.end()))throw std::runtime_error("WBC joint order differs from Unitree SDK");
    return {joint_vector(cfg["default_joint_pos"],"WBC defaults"),joint_vector(cfg["stiffness"],"WBC stiffness",true),joint_vector(cfg["damping"],"WBC damping",true)};
}
} // namespace teleop
