#pragma once
#include "WbcPolicy.hpp"
#include "CameraReferenceScope.hpp"
#include "PdTorqueClip.hpp"
#include "RobotChecks.hpp"

namespace teleop {
// WBC consumes the trained 39-D reference verbatim. No EMA, joint slew,
// height/speed clamp, entry blend or motor-position clipping is applied here.
class WbcControl {
    WbcPolicy policy_;
    PdTorqueClip torque_clip_;
    CameraReferenceScope scope_;
    double height_;
    WbcReference reference_;
    bool waiting_for_camera_=true;
public:
    explicit WbcControl(const YAML::Node& cfg):policy_(cfg["policy_dir"].as<std::string>()),
        torque_clip_(cfg["pd_torque_clipping"]),scope_(cfg["retarget_scope"].as<std::string>("wholebody")),
        height_(cfg["wbc_base_height"].as<double>()) {
        if(!std::isfinite(height_))throw std::runtime_error("Invalid WBC torso reference height");
    }
    const PolicyParams& params()const{return policy_.params();}
    const WbcReference& reference()const{return reference_;}
    bool waiting_for_camera()const{return waiting_for_camera_;}
    const std::vector<float>& observations()const{return policy_.observations();}
    static Eigen::Quaterniond torso_orientation(const Reference& ref){
        return (ref.root*Eigen::AngleAxisd(ref.q[12],Eigen::Vector3d::UnitZ())
            *Eigen::AngleAxisd(ref.q[13],Eigen::Vector3d::UnitX())*Eigen::AngleAxisd(ref.q[14],Eigen::Vector3d::UnitY())).normalized();
    }
    void begin(const RobotState&,double){
        policy_.reset();waiting_for_camera_=true;reference_={};reference_.q=params().defaults;reference_.height=height_;
    }
    Joints step(const RobotState& measured,Reference camera,double,bool check_orientation=true){
        validate_robot_state(measured,check_orientation);
        waiting_for_camera_=!camera.ready||camera.stale||camera.stopped;
        if(waiting_for_camera_){camera=Reference{};camera.q=params().defaults;}
        for(int i=0;i<29;++i)if(!camera.active[i]){camera.q[i]=params().defaults[i];camera.dq[i]=0.;}
        if(!camera.base_tracked){camera.root=Eigen::Quaterniond::Identity();camera.root_angular_world.setZero();}
        std::array<Reference,10> frames;frames.fill(camera);frames=scope_.apply(frames);
        const auto& ref=frames[0];
        reference_={};reference_.q=ref.q;reference_.height=height_;
        if(ref.torso.valid){
            // TRACE sends the same torso link-origin FK/velocities as the
            // training NPZ, including all three linear and angular components.
            const auto inverse=ref.torso.orientation.conjugate();
            reference_.height=ref.torso.position.z();
            reference_.linear=inverse*ref.torso.linear;
            reference_.angular=inverse*ref.torso.angular;
            reference_.gravity=inverse*-Eigen::Vector3d::UnitZ();
        }else{
            const Eigen::Quaterniond yaw=ref.root*Eigen::AngleAxisd(ref.q[12],Eigen::Vector3d::UnitZ());
            const Eigen::Quaterniond roll=yaw*Eigen::AngleAxisd(ref.q[13],Eigen::Vector3d::UnitX());
            const auto anchor=torso_orientation(ref);
            reference_.gravity=anchor.conjugate()*-Eigen::Vector3d::UnitZ();
            if(ref.base_tracked){
                // Training G1: the full 44 mm offset belongs to waist_roll_link
                // before roll/pitch. GMR's 35+9 mm chain is a different asset.
                const Eigen::Vector3d offset=yaw*Eigen::Vector3d(-.0039635,0.,.044);
                reference_.height=(ref.position+offset).z();
                const Eigen::Vector3d yaw_omega=ref.root_angular_world+ref.dq[12]*(ref.root*Eigen::Vector3d::UnitZ());
                const Eigen::Vector3d omega=yaw_omega+ref.dq[13]*(yaw*Eigen::Vector3d::UnitX())+ref.dq[14]*(roll*Eigen::Vector3d::UnitY());
                reference_.linear=anchor.conjugate()*(ref.linear_velocity_world+yaw_omega.cross(offset));
                reference_.angular=anchor.conjugate()*omega;
            }
        }
        auto command=policy_.step(reference_,measured);
        if(!command.allFinite())throw std::runtime_error("Invalid WBC motor command");
        return torque_clip_.apply(command,measured,params());
    }
};
} // namespace teleop
