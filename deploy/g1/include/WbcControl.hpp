#pragma once
#include "WbcPolicy.hpp"
#include "ReferenceTransition.hpp"
#include "ReferenceSmoother.hpp"
#include "CameraReferenceScope.hpp"
#include "CommandLimits.hpp"
#include "RobotChecks.hpp"
#include "PdTorqueClip.hpp"

namespace teleop {
class WbcControl {
    WbcPolicy policy_;
    PdTorqueClip torque_clip_;
    ReferenceTransition transition_;
    ReferenceSmoother smoother_;
    CameraReferenceScope scope_;
    CommandLimits limits_{std::string(TELEOP_ROOT)+"/deploy/g1/config/joint_limits.json"};
    double height_,previous_=0.,filtered_height_=0.,tau_;
    Eigen::Quaterniond previous_anchor_=Eigen::Quaterniond::Identity();
    WbcReference reference_;
    bool waiting_for_camera_=true;
public:
    explicit WbcControl(const YAML::Node& cfg):policy_(cfg["policy_dir"].as<std::string>()),torque_clip_(cfg["pd_torque_clipping"]),
        transition_(cfg["transition_duration_s"].as<double>()),smoother_(cfg["reference_smoothing_s"].as<double>(),cfg["reference_max_speed"].as<double>()),
        scope_(cfg["retarget_scope"].as<std::string>("wholebody")),height_(cfg["wbc_base_height"].as<double>()),tau_(cfg["reference_smoothing_s"].as<double>()){
        if(!std::isfinite(height_)||height_<.4||height_>1.3)throw std::runtime_error("Invalid WBC torso reference height");
    }
    const PolicyParams& params()const{return policy_.params();}
    const WbcReference& reference()const{return reference_;}
    bool waiting_for_camera()const{return waiting_for_camera_;}
    const std::vector<float>& observations()const{return policy_.observations();}
    static Eigen::Quaterniond torso_orientation(const Reference& ref){
        return (ref.root*Eigen::AngleAxisd(ref.q[12],Eigen::Vector3d::UnitZ())
            *Eigen::AngleAxisd(ref.q[13],Eigen::Vector3d::UnitX())*Eigen::AngleAxisd(ref.q[14],Eigen::Vector3d::UnitY())).normalized();
    }
    void begin(const RobotState& measured,double now){
        policy_.reset();waiting_for_camera_=true;previous_=now;filtered_height_=height_;Reference target;target.q=params().defaults;
        transition_.begin(measured,target,now);smoother_.reset();
        Reference start;start.q=measured.q;start.root=heading(measured.root).conjugate()*measured.root;
        previous_anchor_=torso_orientation(start);
        reference_={};reference_.q=measured.q;reference_.height=height_;
        reference_.gravity=previous_anchor_.conjugate()*-Eigen::Vector3d::UnitZ();
    }
    Joints step(const RobotState& measured,Reference camera,double now,bool check_orientation=true){
        validate_robot_state(measured,check_orientation);const double dt=std::clamp(now-previous_,0.,.1);
        waiting_for_camera_=!camera.ready||camera.stale||camera.stopped;
        if(waiting_for_camera_){
            // Missing camera data is a reference condition, not a policy fault.
            // Keep the learned balance controller and its history running; the
            // existing transition/smoother handles both loss and resumption.
            camera.q=params().defaults;camera.dq.setZero();camera.active.fill(false);
            camera.root=Eigen::Quaterniond::Identity();camera.base_tracked=false;
        }
        // Every absent goal falls back to the policy stance, never to inferred
        // hidden human joints. Learned body corrections still remain active.
        for(int i=0;i<29;++i)if(!camera.active[i]){camera.q[i]=params().defaults[i];camera.dq[i]=0.;}
        if(!camera.base_tracked)camera.root=Eigen::Quaterniond::Identity();
        std::array<Reference,10> frames;frames.fill(camera);frames=scope_.apply(frames);
        for(auto& frame:frames)frame.active.fill(true); // explicit stance goals for absent parts
        const auto transitioned=transition_.sample(frames,now);frames=smoother_.sample(transitioned,dt);scope_.restore_stance(frames,transitioned);
        const auto& ref=frames[0];const auto anchor=torso_orientation(ref);
        reference_={};reference_.q=ref.q;reference_.height=height_;
        // torso_link offset for the GMR G1 model; raw translation is used only
        // when both full leg chains were observed. Arms keeps the fixed height.
        if(ref.base_tracked){
            const auto yaw=ref.root*Eigen::AngleAxisd(ref.q[12],Eigen::Vector3d::UnitZ());
            const auto roll=yaw*Eigen::AngleAxisd(ref.q[13],Eigen::Vector3d::UnitX());
            reference_.height=std::clamp((ref.position+yaw*Eigen::Vector3d(-.0039635,0,.035)+roll*Eigen::Vector3d(0,0,.009)).z(),.4,1.2);
        }
        reference_.gravity=anchor.conjugate()*-Eigen::Vector3d::UnitZ();
        const double height_before=filtered_height_,a=tau_>0.?-std::expm1(-dt/tau_):1.;
        filtered_height_+=std::clamp(a*(reference_.height-filtered_height_),-.5*dt,.5*dt);
        reference_.height=filtered_height_;
        if(dt>1e-6)reference_.linear=anchor.conjugate()*Eigen::Vector3d(0.,0.,(filtered_height_-height_before)/dt);
        if(dt>1e-6){Eigen::Quaterniond delta=previous_anchor_.conjugate()*anchor;if(delta.w()<0.)delta.coeffs()*=-1.;
            const Eigen::AngleAxisd aa(delta);reference_.angular=aa.axis()*aa.angle()/dt;
            const double n=reference_.angular.norm();if(n>3.)reference_.angular*=3./n;}
        previous_anchor_=anchor;previous_=now;
        // Root translation is held by the existing camera retargeter. No camera
        // walking velocity is synthesized for this in-place teleop mode.
        return torque_clip_.apply(limits_.clamp(policy_.step(reference_,measured)),measured,params());
    }
};
} // namespace teleop
