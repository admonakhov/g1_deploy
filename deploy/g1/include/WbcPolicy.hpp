#pragma once
#include "OnnxPolicy.hpp"
#include "PolicyParams.hpp"

namespace teleop {
struct WbcReference {
    Joints q=standing();
    double height=.8382;
    Eigen::Vector3d linear=Eigen::Vector3d::Zero(),angular=Eigen::Vector3d::Zero(),gravity{0.,0.,-1.};
};
class WbcPolicy {
    YAML::Node cfg_;
    PolicyParams params_;
    Joints scale_;
    OnnxPolicy model_;
    std::vector<float> previous_=std::vector<float>(29,0.f),observation_;
public:
    explicit WbcPolicy(const std::filesystem::path& directory):cfg_(YAML::LoadFile((directory/"params/config.yaml").string())),
        params_(wbc_params(cfg_)),scale_(joint_vector(cfg_["action"]["scale"],"WBC action scale",true)),
        model_(directory/"params/policy.onnx",{{"obs",132}},29) {
        const std::vector<std::string> terms={"ref_base_height","ref_base_lin_vel_b","ref_base_ang_vel_b","ref_gravity_b","ref_joint_pos","base_ang_vel","projected_gravity","joint_pos","joint_vel","actions"};
        if(cfg_["schema_version"].as<std::string>()!="wbc_tracking_params_v1"||cfg_["robot_id"].as<std::string>()!="g1"
            ||std::abs(cfg_["policy_step_dt"].as<double>()-.02)>1e-8||cfg_["action"]["action_mode"].as<std::string>()!="reference_residual"
            ||cfg_["tracking"]["actor_history_length"].as<int>()!=1||cfg_["tracking"]["has_state_estimation"].as<bool>()
            ||cfg_["tracking"]["anchor_body_name"].as<std::string>()!="torso_link"||cfg_["tracking"]["actor_observation_names"].as<std::vector<std::string>>()!=terms)
            throw std::runtime_error("Unsupported WBC training contract");
        const std::vector<int> dims={1,3,3,3,29,3,3,29,29,29};
        for(size_t i=0;i<terms.size();++i){auto block=cfg_["actor_observations"][terms[i]];auto scales=block["scale"].as<std::vector<double>>();
            if(block["dim"].as<int>()!=dims[i]||scales.size()!=size_t(dims[i])||!std::all_of(scales.begin(),scales.end(),[](double x){return x==1.;}))throw std::runtime_error("Unsupported WBC observation scaling/order");}
    }
    const PolicyParams& params()const{return params_;}
    void reset(){std::fill(previous_.begin(),previous_.end(),0.f);}
    const std::vector<float>& observations()const{return observation_;}
    Joints step(const WbcReference& ref,const RobotState& robot){
        observation_.clear();observation_.reserve(132);observation_.push_back(float(ref.height));
        auto append=[this](const auto& values){for(auto x:values)observation_.push_back(float(x));};
        append(ref.linear);append(ref.angular);append(ref.gravity);append(ref.q);
        append(robot.angular_velocity);append(robot.root.conjugate()*-Eigen::Vector3d::UnitZ());
        append((robot.q-params_.defaults).eval());append(robot.dq);append(previous_);
        auto raw=model_.run({{"obs",observation_}});previous_=raw;
        Joints result;for(int i=0;i<29;++i)result[i]=ref.q[i]+scale_[i]*raw[i];
        return result;
    }
};
} // namespace teleop
