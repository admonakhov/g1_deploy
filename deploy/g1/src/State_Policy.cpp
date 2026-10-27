#include "State_Policy.hpp"

struct State_Policy::Engine{
    teleop::TrajLocoPolicy walking;
    teleop::GetUpPolicy getup;
    std::unique_ptr<teleop::WbcControl> whole,arms;
    teleop::WbcControl* wbc=nullptr;
    teleop::CommandLimits limits{std::string(TELEOP_ROOT)+"/deploy/g1/config/joint_limits.json"};
    teleop::ReferenceStream stream;
    std::atomic<bool> running{false},completed{false},recover_walk{false};
    teleop::PolicyFailureLatch failure;
    teleop::RecoveryInput recovery_input;
    std::thread worker,receiver;
    std::mutex mutex;
    teleop::Joints command=teleop::standing();
    double lx=0.,ly=0.,rx=0.;
    explicit Engine():walking(param::config["FSM"]["Walk"]["policy_dir"].as<std::string>()),
        getup(param::config["FSM"]["GetUp"]["policy_dir"].as<std::string>()),
        whole(std::make_unique<teleop::WbcControl>(param::config["FSM"]["Teleop"])),
        arms(std::make_unique<teleop::WbcControl>(param::config["FSM"]["TeleopArms"])),
        stream(50.,param::config["FSM"]["Teleop"]["buffer_delay_s"].as<double>(),param::config["FSM"]["Teleop"]["stream_timeout_s"].as<double>()){}
    ~Engine(){stop();}
    void stop(){running=false;if(worker.joinable())worker.join();if(receiver.joinable())receiver.join();}
    void start(Mode mode,const YAML::Node& cfg,const teleop::RobotState& robot,bool recovering=false){
        const bool is_walk=mode==Mode::Walking,is_getup=mode==Mode::GetUp;
        stop();failure.clear();completed=false;recovery_input.reset(recovering);const double began=teleop::steady_seconds();
        {std::lock_guard<std::mutex> lock(mutex);command=robot.q;lx=ly=rx=0.;}
        if(is_walk){walking.configure_velocity_limits(cfg["velocity_limits"]);walking.reset(robot);}
        else if(is_getup){getup.reset(robot);spdlog::info("GetUp: {} take, {} frames at 50 Hz",getup.front_selected()?"t061":"t067",getup.frames());}
        else{
            wbc=cfg["retarget_scope"].as<std::string>("wholebody")=="arms"?arms.get():whole.get();
            stream.reset(robot.q);wbc->begin(robot,began);
            spdlog::info("WBC: waiting for camera poses; balance remains active with the initial stance");
            auto input=std::make_unique<teleop::UdpInput>(cfg["udp_host"].as<std::string>(),cfg["udp_port"].as<int>());
            running=true;
            receiver=std::thread([this,input=std::move(input)]{try{
                while(running&&!failure.active()){input->poll(stream);std::this_thread::sleep_for(std::chrono::milliseconds(1));}
            }catch(const std::exception& e){spdlog::error("Camera receiver: {}; switching to Walk",e.what());failure.report(teleop::PolicyFailure::Teleop);}});
        }
        running=true;
        worker=std::thread([this,is_walk,is_getup,began,cfg]{try{
            auto deadline=std::chrono::steady_clock::now();double report=began,previous=began,total_ms=0.,max_ms=0.,max_period=0.;uint64_t ticks=0,late=0;
            bool camera_waiting=true;
            const bool orientation=cfg["enable_bad_orientation_passive"].as<bool>();
            const double tilt_limit=cfg["bad_orientation_limit_rad"].as<double>();
            while(running&&!failure.active()){
                const double now=teleop::steady_seconds();teleop::RobotState robot;
                try{robot=State_Policy::measured(false);}
                catch(...){failure.report(teleop::PolicyFailure::InvalidRobotData);throw;}
                teleop::validate_robot_state(robot,orientation,tilt_limit);
                double x,y,r;{std::lock_guard<std::mutex> lock(mutex);x=lx;y=ly;r=rx;}
                const double started=teleop::steady_seconds();teleop::Joints target;
                if(is_walk){const auto input=recovery_input.sample(x,y,r,now);target=walking.step(robot,input[0],input[1],input[2],cfg["idle_only"].as<bool>(false));}
                else if(is_getup)target=getup.step(robot);
                else{
                    auto ref=stream.sample(now);
                    target=wbc->step(robot,ref,now,false); // Orientation check uses this state's flag/limit above.
                    if(wbc->waiting_for_camera()!=camera_waiting){
                        camera_waiting=wbc->waiting_for_camera();
                        spdlog::info("WBC: {}",camera_waiting?"waiting for camera poses; returning smoothly to the initial stance, balance remains active":"camera reference ready; resuming teleop smoothly");
                    }
                }
                const double elapsed=1000.*(teleop::steady_seconds()-started);total_ms+=elapsed;max_ms=std::max(max_ms,elapsed);max_period=std::max(max_period,1000.*(now-previous));previous=now;++ticks;
                {std::lock_guard<std::mutex> lock(mutex);command=target;}
                if(is_getup&&getup.finished()){completed=true;break;}
                deadline+=std::chrono::milliseconds(20);const auto finished=std::chrono::steady_clock::now();if(deadline<finished){++late;deadline=finished;}
                if(now-report>=5.){
                    spdlog::info("{} timing: policy={:.1f} Hz, inference mean/max={:.2f}/{:.2f} ms, max period={:.2f} ms, missed={}; {}",
                        is_walk?"traj_loco":"WBC",ticks/(now-report),total_ms/ticks,max_ms,max_period,late,
                        is_walk?(walking.adjusting()?"stand_adjust":"walking"):stream.diagnostics(now).dump());
                    report=now;ticks=late=0;total_ms=max_ms=max_period=0.;
                }
                std::this_thread::sleep_until(deadline);
            }
        }catch(const std::exception& e){
            failure.report(!is_walk&&!is_getup?teleop::PolicyFailure::Teleop:teleop::PolicyFailure::OtherPolicy);
            spdlog::error("Policy control: {}; switching to {}",e.what(),failure.walk()?"Walk":"Passive");
        }});
    }
};
State_Policy::State_Policy(int state,std::string name,Mode mode):FSMState(state,name),cfg_(param::config["FSM"][name]),mode_(mode){
    static std::weak_ptr<Engine> shared;engine_=shared.lock();
    if(!engine_){spdlog::info("Preloading traj_loco, WBC and GetUp ...");engine_=std::make_shared<Engine>();shared=engine_;spdlog::info("Policy bundles ready");}
    registered_checks.insert(registered_checks.begin(),{[this]{return engine_->failure.passive();},FSMStringMap.right.at("Passive")});
    if(mode_==Mode::Teleop)registered_checks.insert(registered_checks.begin(),{[this]{
        if(!engine_->failure.walk())return false;
        try{measured(false);}catch(const std::exception& e){
            engine_->failure.report(teleop::PolicyFailure::InvalidRobotData);spdlog::error("Walk recovery blocked by robot data: {}",e.what());return false;
        }
        engine_->recover_walk=true;return true;
    },FSMStringMap.right.at(cfg_["failure_state"].as<std::string>("Walk"))});
    if(mode_==Mode::GetUp)registered_checks.push_back({[this]{return valid_&&engine_->completed.load();},FSMStringMap.right.at(cfg_["return_state"].as<std::string>())});
}
State_Policy::~State_Policy(){exit();}
teleop::RobotState State_Policy::measured(bool check_orientation){
    teleop::RobotState out;std::lock_guard<std::mutex> lock(lowstate->mutex_);const auto& msg=lowstate->msg_;const auto& q=msg.imu_state().quaternion();
    out.root=Eigen::Quaterniond(q[0],q[1],q[2],q[3]);for(int i=0;i<29;++i){out.q[i]=msg.motor_state()[i].q();out.dq[i]=msg.motor_state()[i].dq();}
    for(int i=0;i<3;++i)out.angular_velocity[i]=msg.imu_state().gyroscope()[i];
    teleop::validate_robot_state(out,check_orientation);out.root.normalize();return out;
}
void State_Policy::enter(){
    valid_=false;
    bool measured_valid=false;
    try{
        const auto robot=measured(false);measured_valid=true;const bool is_walk=mode_==Mode::Walking;
        const bool recovering=is_walk&&engine_->recover_walk.exchange(false);
        engine_->start(mode_,cfg_,robot,recovering);
        const auto& params=is_walk?engine_->walking.params():mode_==Mode::GetUp?engine_->getup.params():engine_->wbc->params();
        for(int i=0;i<29;++i){auto& motor=lowcmd->msg_.motor_cmd()[i];motor.q()=robot.q[i];motor.kp()=params.kp[i];motor.kd()=params.kd[i];motor.dq()=motor.tau()=0.;}
        valid_=true;spdlog::info("{}: {} at 50 Hz; reset policy history; bad_orientation_passive={}",getStateString(),
            is_walk?"traj_loco":mode_==Mode::GetUp?"VQ GetUp":"WBC camera reference",cfg_["enable_bad_orientation_passive"].as<bool>());
        if(recovering)spdlog::warn("Walk recovery: balance active, zero velocity until sticks are centered for 0.2 s");
    }catch(const std::exception& e){
        engine_->stop();engine_->failure.clear();
        if(mode_==Mode::Walking)engine_->recover_walk=false;
        engine_->failure.report(!measured_valid?teleop::PolicyFailure::InvalidRobotData:mode_==Mode::Teleop?teleop::PolicyFailure::Teleop:teleop::PolicyFailure::OtherPolicy);
        spdlog::error("Cannot start {}: {}; switching to {}",getStateString(),e.what(),engine_->failure.walk()?"Walk":"Passive");
    }
}
void State_Policy::run(){if(!valid_||engine_->failure.active())return;std::lock_guard<std::mutex> lock(engine_->mutex);engine_->lx=lowstate->joystick.lx();engine_->ly=lowstate->joystick.ly();engine_->rx=lowstate->joystick.rx();for(int i=0;i<29;++i)lowcmd->msg_.motor_cmd()[i].q()=engine_->command[i];}
void State_Policy::exit(){engine_->stop();valid_=false;}
