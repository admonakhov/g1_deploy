#pragma once
#include <deque>
#include <fstream>
#include <sstream>
#include "OnnxPolicy.hpp"
#include "PolicyParams.hpp"
#include "RobotChecks.hpp"

namespace teleop {
// The get_up branch of ROS2 run_vq_policy. Only the GetUp decoder and its
// front/back takes are bundled; the punch and punch-walking policies are absent.
class GetUpPolicy {
    YAML::Node cfg_;
    PolicyParams params_;
    OnnxPolicy decoder_;
    std::vector<int> ids_;
    std::vector<std::vector<float>> front_,back_;
    const std::vector<std::vector<float>>* motion_=nullptr;
    std::deque<std::vector<float>> history_;
    std::vector<float> previous_=std::vector<float>(23,0.f),observations_,command_;
    Eigen::Quaternionf offset_=Eigen::Quaternionf::Identity();
    size_t frame_=0;
    bool front_selected_=false;
    static std::vector<std::vector<float>> load(const std::filesystem::path& directory,const YAML::Node& take){
        std::ifstream file(directory/"aux"/(take["file"].as<std::string>()+".csv"));
        if(!file)throw std::runtime_error("Missing GetUp CSV");
        std::vector<std::vector<float>> rows;
        for(std::string line;std::getline(file,line);){
            if(line.empty())continue;
            std::vector<float> row;std::istringstream stream(line);
            for(std::string cell;std::getline(stream,cell,',');){const float x=std::stof(cell);if(!std::isfinite(x))throw std::runtime_error("Invalid GetUp CSV value");row.push_back(x);}
            if(row.size()<71)throw std::runtime_error("GetUp CSV requires 71 values");
            const Eigen::Quaterniond q(row[67],row[68],row[69],row[70]);tilt_angle(q);
            rows.push_back(std::move(row));
        }
        const double fps=take["fps"].as<double>(50.);
        if(!std::isfinite(fps)||fps!=50.)throw std::runtime_error("GetUp requires 50 Hz takes");
        const auto begin=std::min(rows.size(),size_t(std::lround(take["start"].as<double>()*fps)));
        const auto end=std::min(rows.size(),size_t(std::lround(take["end"].as<double>()*fps))+1);
        if(begin>=end)throw std::runtime_error("Empty GetUp take");
        return {rows.begin()+begin,rows.begin()+end};
    }
public:
    explicit GetUpPolicy(const std::filesystem::path& directory):cfg_(YAML::LoadFile((directory/"source_config.yaml").string())),params_(vq_params(cfg_)),
        decoder_(directory/"aux/getup.onnx",{{"p_obs",750},{"cmd",73}},23){
        ids_=cfg_["g1"]["control_joints"].as<std::vector<int>>();
        const std::vector<int> expected={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,22,23,24,25};
        if(ids_!=expected||cfg_["g1"]["obs_scheme"].as<std::string>()!="punch"||std::abs(cfg_["control"]["step_dt"].as<double>()-.02)>1e-8)throw std::runtime_error("Unsupported GetUp contract");
        const auto p=cfg_["g1"]["punch"];
        front_=load(directory,p["trajs"][p["front_get_up"].as<std::string>()]);
        back_=load(directory,p["trajs"][p["back_get_up"].as<std::string>()]);
    }
    const PolicyParams& params()const{return params_;}
    void reset(const RobotState& robot){
        validate_robot_state(robot,false);history_.clear();std::fill(previous_.begin(),previous_.end(),0.f);frame_=0;
        const auto quat=robot.root.cast<float>().normalized();
        const Eigen::Vector3f gravity=quat.conjugate()*-Eigen::Vector3f::UnitZ();
        front_selected_=gravity.x()<-.1;motion_=front_selected_?&front_:&back_;
        const auto& row=motion_->front();const Eigen::Quaternionf reference(row[67],row[68],row[69],row[70]);
        const auto rotation=(quat*reference.conjugate()).normalized().toRotationMatrix();
        const float yaw=std::atan2(rotation(1,0),rotation(0,0));offset_=Eigen::Quaternionf(Eigen::AngleAxisf(yaw,Eigen::Vector3f::UnitZ()));
    }
    bool finished()const{return motion_&&frame_>=motion_->size();}
    bool front_selected()const{return front_selected_;}
    size_t frame()const{return frame_;}
    size_t frames()const{return motion_?motion_->size():0;}
    const std::vector<float>& observations()const{return observations_;}
    const std::vector<float>& command()const{return command_;}
    Joints step(const RobotState& robot){
        validate_robot_state(robot,false);
        if(!motion_)throw std::runtime_error("GetUp not initialized");
        if(finished())throw std::runtime_error("GetUp complete; transition to return_state");
        std::vector<float> frame;frame.reserve(75);
        for(auto x:robot.angular_velocity)frame.push_back(float(x)*.2f);
        const auto quat=robot.root.cast<float>().normalized();
        const Eigen::Vector3f gravity=quat.conjugate()*-Eigen::Vector3f::UnitZ();for(auto x:gravity)frame.push_back(float(x));
        for(int id:ids_)frame.push_back(float(robot.q[id])-float(params_.defaults[id]));
        for(int id:ids_)frame.push_back(float(robot.dq[id])*.05f);
        frame.insert(frame.end(),previous_.begin(),previous_.end());for(auto& x:frame)x=std::clamp(x,-100.f,100.f);
        history_.push_back(frame);if(history_.size()>10)history_.pop_front();observations_.clear();observations_.reserve(750);
        for(int i=0;i<10;++i){const auto& h=history_[std::max(0,i-(10-int(history_.size())))];observations_.insert(observations_.end(),h.begin(),h.end());}
        const auto& row=(*motion_)[frame_];command_.assign(73,0.f);std::copy_n(row.begin(),64,command_.begin());
        const Eigen::Quaternionf reference(row[67],row[68],row[69],row[70]);
        const auto rotation=(offset_*reference*quat.conjugate()).normalized().toRotationMatrix();
        for(int r=0;r<3;++r){command_[67+2*r]=float(rotation(r,0));command_[68+2*r]=float(rotation(r,1));}
        auto raw=decoder_.run({{"p_obs",observations_},{"cmd",command_}});for(auto& x:raw)x=std::clamp(x,-20.f,20.f);previous_=raw;
        Joints target=params_.defaults;for(size_t i=0;i<ids_.size();++i)target[ids_[i]]=float(params_.defaults[ids_[i]])+raw[i];
        for(int i:{19,20,21,26,27,28})target[i]=0.;
        ++frame_;return target;
    }
};
}
