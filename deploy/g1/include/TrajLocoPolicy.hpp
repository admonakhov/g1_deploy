#pragma once
#include <deque>
#include "OnnxPolicy.hpp"
#include "PolicyParams.hpp"

namespace teleop {
// Port of reset_vq/vq_path/run_vq_policy from the supplied ROS2 deployment.
// All indices and actions stay in SDK motor order; no SONIC encoder/planner.
class TrajLocoPolicy {
    YAML::Node cfg_,vc_;
    PolicyParams params_;
    OnnxPolicy decoder_,adjust_,estimator_;
    std::vector<int> ids_;
    std::vector<float> anchor_,max_v_,max_acc_,deadzone_,last_=std::vector<float>(23,0.f);
    std::deque<std::vector<float>> history_;
    std::vector<float> history_flat_,command_;
    Eigen::Vector3f pose_=Eigen::Vector3f::Zero(),velocity_=Eigen::Vector3f::Zero(),target_=Eigen::Vector3f::Zero(),filter_=Eigen::Vector3f::Zero();
    Eigen::Vector2f odom_=Eigen::Vector2f::Zero();
    float alpha_,back_,omega_drop_,path_dt_,estimator_cmd_;
    int adjust_ticks_,tick_=0;
    bool adjusting_=false;
    float path_speed_limit_=5.f,planar_limit_=0.f;
    static float wrap(float x){return std::atan2(std::sin(x),std::cos(x));}
    float acceleration(float value,int axis)const{
        const float t=axis<2?(std::abs(value)-.5f)/.5f:std::abs(value)/.01f;
        if(t<=0.f)return 1.f;
        if(t>=1.f)return max_acc_[axis];
        return 1.f+t*t*(3.f-2.f*t)*(max_acc_[axis]-1.f);
    }
    void step_velocity(Eigen::Vector3f& value,float dt)const{
        for(int i=0;i<3;++i){const float step=acceleration(value[i],i)*dt;value[i]+=std::clamp(target_[i]-value[i],-step,step);}
        const float linear=std::hypot(value[0],value[1]);if(planar_limit_>0.f&&linear>planar_limit_)value.head<2>()*=planar_limit_/linear;
    }
    std::array<Eigen::Vector3f,10> path(double lx,double ly,double rx,bool idle){
        // ROS2 source normalizes velocity_command by max_v before this step.
        Eigen::Vector3f stick(float(ly),float(-lx),float(-rx));
        if(idle)stick.setZero();
        for(int i=0;i<3;++i){if(!std::isfinite(stick[i]))throw std::runtime_error("Invalid gamepad axis");
            stick[i]=std::clamp(stick[i],-1.f,1.f);if(std::abs(stick[i])<deadzone_[i])stick[i]=0.f;
            filter_[i]=(1.f-alpha_)*stick[i]+alpha_*filter_[i];}
        float wz=filter_[2]*max_v_[2];const float planar=std::hypot(velocity_[0],velocity_[1]);
        if(planar>.8f)wz/=1.f+(planar-.8f)*omega_drop_;
        target_={std::clamp(max_v_[0]*filter_[0],-back_,max_v_[0]),max_v_[1]*filter_[1],wz};
        const float requested=std::hypot(target_[0],target_[1]);
        if(planar_limit_>0.f&&requested>planar_limit_)target_.head<2>()*=planar_limit_/requested;
        step_velocity(velocity_,.02f);
        const float cy=std::cos(pose_[2]),sy=std::sin(pose_[2]);
        pose_[0]+=.02f*(cy*velocity_[0]-sy*velocity_[1]);pose_[1]+=.02f*(sy*velocity_[0]+cy*velocity_[1]);
        pose_[2]=wrap(pose_[2]+.02f*velocity_[2]);
        std::array<Eigen::Vector3f,10> raw,out;auto p=pose_,v=velocity_;float previous=0.f;
        for(int i=0;i<10;++i){const float t=i*path_dt_,dt=t-previous;previous=t;step_velocity(v,dt);
            const float c=std::cos(p[2]),s=std::sin(p[2]);p[0]+=dt*(c*v[0]-s*v[1]);p[1]+=dt*(s*v[0]+c*v[1]);p[2]=wrap(p[2]+dt*v[2]);raw[i]=p;}
        out[0]=raw[0];float cs=std::cos(raw[0][2]),sn=std::sin(raw[0][2]);
        for(int i=1;i<10;++i){cs=(1.f-alpha_)*std::cos(raw[i][2])+alpha_*cs;sn=(1.f-alpha_)*std::sin(raw[i][2])+alpha_*sn;
            out[i]={(1.f-alpha_)*raw[i][0]+alpha_*out[i-1][0],(1.f-alpha_)*raw[i][1]+alpha_*out[i-1][1],std::atan2(sn,cs)};}
        return out;
    }
public:
    explicit TrajLocoPolicy(const std::filesystem::path& directory):cfg_(YAML::LoadFile((directory/"config.yaml").string())),vc_(cfg_["g1"]["vq"]),params_(vq_params(cfg_)),
        decoder_(directory/"checkpoints/2026-02-27_18-31-24_decoder.onnx",{{"p_obs",750},{"cmd",163}},23),
        adjust_(directory/"aux/stand_adjust.onnx",{{"p_obs",750},{"cmd",163}},23),estimator_(directory/"aux/estimator.onnx",{{"p_obs",750},{"cmd",1}},2){
        ids_=cfg_["g1"]["control_joints"].as<std::vector<int>>();anchor_=vc_["anchor"].as<std::vector<float>>();
        max_v_=vc_["max_v"].as<std::vector<float>>();max_acc_=vc_["max_acc"].as<std::vector<float>>();deadzone_=vc_["deadzone"].as<std::vector<float>>();
        alpha_=vc_["ema_alpha"].as<float>();back_=vc_["back_v_max"].as<float>();omega_drop_=vc_["omega_drop_scale"].as<float>();
        path_dt_=vc_["dt_path_points"].as<float>();adjust_ticks_=vc_["stand_adjust_ticks"].as<int>();estimator_cmd_=vc_["estimator_cmd"].as<float>();
        const std::vector<int> expected={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,22,23,24,25};
        if(ids_!=expected||anchor_.size()!=64||cfg_["g1"]["obs_scheme"].as<std::string>()!="vq"||cfg_["g1"]["memory_size"].as<int>()!=10
            ||std::abs(cfg_["control"]["step_dt"].as<double>()-.02)>1e-8||vc_["num_path_points"].as<int>()!=10||path_dt_!=.1f
            ||max_v_.size()!=3||max_acc_.size()!=3||deadzone_.size()!=3||alpha_<0.f||alpha_>=1.f||adjust_ticks_<0)
            throw std::runtime_error("Unsupported traj_loco training contract");
        for(const auto* v:{&anchor_,&max_v_,&max_acc_,&deadzone_})for(float x:*v)if(!std::isfinite(x))throw std::runtime_error("Invalid VQ configuration");
        for(int i=0;i<3;++i)if(max_v_[i]<=0.f||max_acc_[i]<=0.f||deadzone_[i]<0.f||deadzone_[i]>=1.f)throw std::runtime_error("Invalid VQ limits");
        if(!std::isfinite(back_)||back_<=0.f||!std::isfinite(omega_drop_)||omega_drop_<0.f||!std::isfinite(estimator_cmd_))throw std::runtime_error("Invalid VQ trajectory parameters");
    }
    const PolicyParams& params()const{return params_;}
    static void validate_velocity_limits(const YAML::Node& limits){
        if(!limits)return;
        for(const auto key:{"forward","backward","lateral"}){const auto x=limits[key].as<float>();if(!std::isfinite(x)||x<=0.f)throw std::runtime_error("Invalid TrajLoco velocity_limits");}
        const float planar=limits["planar"].as<float>(0.f);
        if(!std::isfinite(planar)||planar<0.f)throw std::runtime_error("Invalid TrajLoco velocity_limits.planar");
    }
    // Per-state command limits; the copied training configuration is immutable.
    void configure_velocity_limits(const YAML::Node& limits){
        validate_velocity_limits(limits);
        max_v_=vc_["max_v"].as<std::vector<float>>();back_=vc_["back_v_max"].as<float>();planar_limit_=0.f;
        if(limits){
            max_v_[0]=limits["forward"].as<float>();back_=limits["backward"].as<float>();max_v_[1]=limits["lateral"].as<float>();
            planar_limit_=limits["planar"].as<float>(0.f);
            if(!std::isfinite(max_v_[0])||max_v_[0]<=0.f||!std::isfinite(back_)||back_<=0.f
                ||!std::isfinite(max_v_[1])||max_v_[1]<=0.f||!std::isfinite(planar_limit_)||planar_limit_<0.f)
                throw std::runtime_error("Invalid TrajLoco velocity_limits");
        }
        // The original 5 m/s path-position clip must not silently cap Run.
        path_speed_limit_=std::max(5.f,std::max(back_,planar_limit_>0.f?planar_limit_:std::hypot(max_v_[0],max_v_[1])));
    }
    void reset(const RobotState& robot){history_.clear();std::fill(last_.begin(),last_.end(),0.f);pose_={0,0,float(std::atan2(robot.root.toRotationMatrix()(1,0),robot.root.toRotationMatrix()(0,0)))};
        velocity_.setZero();target_.setZero();filter_.setZero();odom_.setZero();tick_=0;}
    const std::vector<float>& observations()const{return history_flat_;}
    const std::vector<float>& command()const{return command_;}
    bool adjusting()const{return adjusting_;}
    Eigen::Vector3f target_velocity()const{return target_;}
    Eigen::Vector3f planned_velocity()const{return velocity_;}
    float path_speed_limit()const{return path_speed_limit_;}
    Joints step(const RobotState& robot,double lx,double ly,double rx,bool idle=false){
        std::vector<float> frame;frame.reserve(75);
        for(auto x:robot.angular_velocity)frame.push_back(float(x)*.2f);
        const Eigen::Vector3d gravity=robot.root.conjugate()*-Eigen::Vector3d::UnitZ();for(auto x:gravity)frame.push_back(float(x));
        for(int id:ids_)frame.push_back(float(robot.q[id])-float(params_.defaults[id]));
        for(int id:ids_)frame.push_back(float(robot.dq[id])*.05f);
        frame.insert(frame.end(),last_.begin(),last_.end());for(auto& x:frame)x=std::clamp(x,-100.f,100.f);
        history_.push_back(frame);if(history_.size()>10)history_.pop_front();history_flat_.clear();history_flat_.reserve(750);
        for(int i=0;i<10;++i){const auto& h=history_[std::max(0,i-(10-int(history_.size())))];history_flat_.insert(history_flat_.end(),h.begin(),h.end());}
        const auto points=path(lx,ly,rx,idle);const auto rotation=robot.root.cast<float>().normalized().toRotationMatrix();
        const float yaw=std::atan2(rotation(1,0),rotation(0,0)),cy=std::cos(yaw),sy=std::sin(yaw);
        command_.assign(163,0.f);std::copy(anchor_.begin(),anchor_.end(),command_.begin());
        for(int i=0;i<10;++i){const float dx=points[i][0]-odom_[0],dy=points[i][1]-odom_[1],ct=std::cos(points[i][2]),st=std::sin(points[i][2]);const int o=64+9*i;const float limit=path_speed_limit_*(.02f+i*path_dt_);
            command_[o]=std::clamp(cy*dx+sy*dy,-limit,limit);command_[o+1]=std::clamp(-sy*dx+cy*dy,-limit,limit);
            for(int r=0;r<3;++r){command_[o+3+2*r]=rotation(0,r)*ct+rotation(1,r)*st;command_[o+4+2*r]=-rotation(0,r)*st+rotation(1,r)*ct;}}
        std::copy_n(command_.begin()+64,9,command_.begin()+154);
        const float distance=(odom_-points.back().head<2>()).norm(),yaw_error=std::abs(wrap(yaw-points.back()[2]));
        adjusting_=tick_<adjust_ticks_||((distance>=.01f||yaw_error>=.05f)&&distance<.5f&&yaw_error<1.f);
        auto raw=(adjusting_?adjust_:decoder_).run({{"p_obs",history_flat_},{"cmd",command_}});for(auto& x:raw)x=std::clamp(x,-20.f,20.f);last_=raw;
        if(velocity_.norm()>0.f){auto e=estimator_.run({{"p_obs",history_flat_},{"cmd",{estimator_cmd_}}});const float ex=std::clamp(e[0],-8.f,8.f),ey=std::clamp(e[1],-8.f,8.f),wx=ex*cy-ey*sy,wy=ex*sy+ey*cy;
            if(std::abs(wx)>.01f)odom_[0]+=.02f*wx;
            if(std::abs(wy)>.01f)odom_[1]+=.02f*wy;
        }else odom_=pose_.head<2>();
        Joints result=params_.defaults;for(size_t i=0;i<ids_.size();++i)result[ids_[i]]+=raw[i];for(int i:{19,20,21,26,27,28})result[i]=0.;++tick_;return result;
    }
};
} // namespace teleop
