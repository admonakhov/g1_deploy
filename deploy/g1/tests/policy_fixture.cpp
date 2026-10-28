// JSONL adapter: real policies and standalone PD state, no DDS initialized.
#include <iostream>
#include "DeployConfig.hpp"
#include "WbcControl.hpp"
#include "PolicyRecovery.hpp"
#include "ReferenceStream.hpp"
int main(int argc,char** argv){try{
    using namespace teleop;using nlohmann::json;
    const std::string mode=argc>1?argv[1]:"walk";const auto root=std::filesystem::path(TELEOP_ROOT)/"deploy/g1/config";
    auto cfg=load_deployment(root/"config.yaml",root)["FSM"];
    if(mode=="wbc_reference"){
        OnnxPolicy model(std::filesystem::path(cfg["Teleop"]["policy_dir"].as<std::string>())/"params/policy.onnx",{{"obs",132}},29);
        std::cout<<json({{"mode","raw WBC ONNX"}}).dump()<<std::endl;
        for(std::string line;std::getline(std::cin,line);){const auto input=json::parse(line);
            std::cout<<json({{"raw",model.run({{"obs",input.at("obs").get<std::vector<float>>()}})}}).dump()<<std::endl;
        }
        return 0;
    }
    if(mode=="getup_reference"){
        // Tensor adapter only. Tests construct ROS2 observations/commands
        // independently and compare the final targets, including large hips.
        const auto directory=std::filesystem::path(cfg["GetUp"]["policy_dir"].as<std::string>());
        OnnxPolicy model(directory/"aux/getup.onnx",{{"p_obs",750},{"cmd",73}},23);
        std::cout<<json({{"mode","raw GetUp reference"}}).dump()<<std::endl;
        for(std::string line;std::getline(std::cin,line);){auto input=json::parse(line);
            auto raw=model.run({{"p_obs",input.at("p_obs").get<std::vector<float>>()},{"cmd",input.at("cmd").get<std::vector<float>>()}});
            for(auto& x:raw)x=std::clamp(x,-20.f,20.f);
            std::cout<<json({{"raw",raw}}).dump()<<std::endl;
        }
        return 0;
    }
    if(mode!="walk"&&mode!="run"&&mode!="getup"&&mode!="wbc"&&mode!="arms"&&mode!="fixstand")throw std::runtime_error("Expected walk, run, getup, wbc, arms or fixstand");
    std::unique_ptr<TrajLocoPolicy> walking;std::unique_ptr<WbcControl> wbc;std::unique_ptr<GetUpPolicy> getup;
    FixedPoseProgram program(cfg["FixStand"]);bool fixed=mode=="fixstand";
    const auto state_cfg=cfg[mode=="walk"?"Walk":mode=="run"?"Run":mode=="getup"?"GetUp":mode=="arms"?"TeleopArms":"Teleop"];
    bool orientation=state_cfg["enable_bad_orientation_passive"].as<bool>();
    if(argc>2)orientation=std::string(argv[2])!="--no-posture-check";
    if(mode=="walk"||mode=="run"){walking=std::make_unique<TrajLocoPolicy>(state_cfg["policy_dir"].as<std::string>());walking->configure_velocity_limits(state_cfg["velocity_limits"]);}
    else if(mode=="getup")getup=std::make_unique<GetUpPolicy>(state_cfg["policy_dir"].as<std::string>());
    else if(!fixed)wbc=std::make_unique<WbcControl>(state_cfg);
    PolicyParams stand_params{joint_vector(cfg["FixStand"]["qs"][1],"stand target"),joint_vector(cfg["FixStand"]["kp"],"stand kp"),joint_vector(cfg["FixStand"]["kd"],"stand kd")};
    auto params=fixed?stand_params:walking?walking->params():getup?getup->params():wbc->params();
    auto values=[](const Joints& v){return std::vector<double>(v.data(),v.data()+29);};
    std::cout<<json({{"names",motor_names},{"standing",values(params.defaults)},{"kp",values(params.kp)},{"kd",values(params.kd)}}).dump()<<std::endl;
    bool begun=false;
    RecoveryInput recovery_input;PolicyFailureLatch failure;
    ReferenceStream camera_stream(50.,cfg["Teleop"]["buffer_delay_s"].as<double>(),cfg["Teleop"]["stream_timeout_s"].as<double>());
    for(std::string line;std::getline(std::cin,line);){
        const auto input=json::parse(line);RobotState robot;
        for(int i=0;i<29;++i){robot.q[i]=input.at("q").at(i);robot.dq[i]=input.at("dq").at(i);}
        auto q=input.at("root").get<std::vector<double>>();robot.root=Eigen::Quaterniond(q.at(0),q.at(1),q.at(2),q.at(3));
        for(int i=0;i<3;++i)robot.angular_velocity[i]=input.at("gyro").at(i);
        const double t=input.value("time",0.);
        if(input.value("teleop_error",false))failure.report(PolicyFailure::Teleop);
        if(failure.walk()){
            validate_robot_state(robot,false);fixed=false;begun=false;
            walking=std::make_unique<TrajLocoPolicy>(cfg["Walk"]["policy_dir"].as<std::string>());
            walking->configure_velocity_limits(cfg["Walk"]["velocity_limits"]);
            orientation=cfg["Walk"]["enable_bad_orientation_passive"].as<bool>();recovery_input.reset(true);failure.clear();
        }
        if(input.value("switch_stand",false)&&!fixed){fixed=true;begun=false;}
        if(input.value("switch_walk",false)&&fixed){fixed=false;begun=false;walking=std::make_unique<TrajLocoPolicy>(cfg["Walk"]["policy_dir"].as<std::string>());orientation=cfg["Walk"]["enable_bad_orientation_passive"].as<bool>();walking->configure_velocity_limits(cfg["Walk"]["velocity_limits"]);}
        validate_robot_state(robot,orientation&&!fixed,state_cfg["bad_orientation_limit_rad"].as<double>());
        if(!begun||input.value("reset",false)){if(fixed)program.begin(robot,t);else if(walking)walking->reset(robot);else if(getup)getup->reset(robot);else{wbc->begin(robot,t);camera_stream.reset(robot.q);}begun=true;}
        Joints command;json extra;
        if(fixed){command=program.sample(t);params=stand_params;extra={{"backend","pd"},{"finished",program.finished(t)}};}
        else if(walking){const auto axes=recovery_input.sample(input.value("lx",0.),input.value("ly",0.),input.value("rx",0.),t);
            command=walking->step(robot,axes[0],axes[1],axes[2],input.value("idle",false));
            params=walking->params();const auto target=walking->target_velocity(),velocity=walking->planned_velocity();
            extra={{"observations",walking->observations()},{"path_command",walking->command()},{"adjusting",walking->adjusting()},
                {"target_velocity",{target[0],target[1],target[2]}},{"planned_velocity",{velocity[0],velocity[1],velocity[2]}},{"path_speed_limit",walking->path_speed_limit()},
                {"backend","Walk"},{"recovery_waiting",recovery_input.waiting()}};}
        else if(getup){command=getup->step(robot);params=getup->params();
            extra={{"observations",getup->observations()},{"getup_command",getup->command()},{"finished",getup->finished()},
                {"front",getup->front_selected()},{"frame",getup->frame()},{"frames",getup->frames()}};}
        else{Reference ref;params=wbc->params();ref.q=params.defaults;ref.active.fill(true);ref.ready=true;
            if(input.contains("reference"))for(int i=0;i<29;++i)ref.q[i]=input["reference"][i];
            if(input.contains("active"))for(int i=0;i<29;++i)ref.active[i]=input["active"][i];
            if(input.value("camera_stream",false)){
                if(input.contains("camera_packets"))for(const auto& packet:input["camera_packets"])camera_stream.ingest(packet,t);
                ref=camera_stream.sample(t);
            }
            command=wbc->step(robot,ref,t,false);extra={{"observations",wbc->observations()},{"reference",values(wbc->reference().q)},
                {"backend",mode=="arms"?"TeleopArms":"Teleop"},{"camera_waiting",wbc->waiting_for_camera()},
                {"camera_diagnostics",camera_stream.diagnostics(t)}};
        }
        extra["kp"]=values(params.kp);extra["kd"]=values(params.kd);extra["command"]=values(command);
        std::cout<<extra.dump()<<std::endl;
    }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
