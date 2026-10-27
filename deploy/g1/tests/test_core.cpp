#include <iostream>
#include "ReferenceStream.hpp"
#include "ReferenceTransition.hpp"
#include "RobotChecks.hpp"
#include "WbcPolicy.hpp"
#include "FixedPoseProgram.hpp"
#include "PdTorqueClip.hpp"
#include "PolicyRecovery.hpp"
int main(){try{
 teleop::PolicyFailureLatch failures;
 failures.report(teleop::PolicyFailure::Teleop);if(!failures.walk()||failures.passive())return 18;
 failures.report(teleop::PolicyFailure::InvalidRobotData);
 failures.report(teleop::PolicyFailure::Teleop);if(failures.walk()||!failures.passive())return 19;
 failures.clear();failures.report(teleop::PolicyFailure::OtherPolicy);if(!failures.passive())return 20;
 failures.clear();if(failures.active())return 21;
 teleop::RecoveryInput input;input.reset(true);
 if(input.sample(0.,1.,0.,0.).norm()!=0.||input.sample(0.,1.,0.,1.).norm()!=0.||!input.waiting())return 22;
 input.sample(0.,0.,0.,1.);input.sample(0.,0.,0.,1.1);if(!input.waiting())return 23;
 input.sample(0.,1.,0.,1.11);input.sample(0.,0.,0.,1.2);input.sample(0.,0.,0.,1.4);if(input.waiting())return 24;
 if(input.sample(0.,.5,0.,1.42).y()!=.5)return 25;
 teleop::RobotState robot;robot.root=Eigen::Quaterniond(Eigen::AngleAxisd(81.*M_PI/180.,Eigen::Vector3d::UnitY()));
 teleop::validate_robot_state(robot,false);
 bool rejected=false;try{teleop::validate_robot_state(robot,true);}catch(const std::exception&){rejected=true;}if(!rejected)return 16;
 teleop::validate_robot_state(robot,true,1.5); // A configurable limit, not a second hardcoded 1 rad check.
 robot.root.coeffs().setZero();rejected=false;
 try{teleop::validate_robot_state(robot,false);}catch(const std::exception&){rejected=true;}if(!rejected)return 17;
 robot.root=Eigen::Quaterniond::Identity();
 teleop::Reference ref;ref.q=teleop::standing();teleop::ReferenceTransition blend(2.);robot.q.setZero();blend.begin(robot,ref,0.);
 std::array<teleop::Reference,10> frames;frames.fill(ref);if(blend.sample(frames,0.)[0].q.norm()>1e-8)return 2;
 if((blend.sample(frames,2.)[0].q-teleop::standing()).norm()>1e-8)return 3;
 teleop::WbcPolicy wbc(std::string(TELEOP_ROOT)+"/deploy/g1/policies/wbc");teleop::WbcReference r;robot.q=r.q;
 wbc.step(r,robot);if(wbc.observations().size()!=132)return 4;
 teleop::ReferenceStream stream(50.,0.,1.);stream.reset(teleop::standing());
 auto packet=nlohmann::json{{"schema","gmr-g1-v1"},{"robot","unitree_g1"},{"status","tracking"},{"person_id",1},
    {"stream_id","test"},{"joint_units","radians"},{"joint_names",teleop::motor_names},{"active_joint_mask",std::vector<bool>(29,true)},
    {"base_tracked",false},{"latency_s",0.}};
 std::vector<double> q(29);
 // Avoid temporary Eigen storage: each packet owns its joint values.
 for(int i=0;i<29;++i)q[i]=teleop::standing()[i];
 q[15]=.1;packet["joint_positions"]=q;packet["frame"]=0;packet["reference_monotonic_s"]=1.;if(!stream.ingest(packet,1.))return 5;
 q[15]=.9;packet["joint_positions"]=q;packet["frame"]=1;packet["reference_monotonic_s"]=1.2;if(!stream.ingest(packet,1.2))return 6;
 auto middle=stream.sample(1.32);if(!middle.ready||std::abs(middle.alpha-.5)>1e-6||std::abs(middle.q[15]-.5)>1e-6)return 7;
 auto active=std::vector<bool>(29,true);active[5]=false;packet["active_joint_mask"]=active;packet["frame"]=2;packet["reference_monotonic_s"]=1.4;
 if(!stream.ingest(packet,1.4))return 10;
 const auto partial=stream.sample(1.52);for(int i=0;i<6;++i)if(partial.active[i])return 11;
 if(!stream.sample(2.5).stale)return 8;
 packet["status"]="stopped";if(!stream.ingest(packet,2.4)||!stream.sample(2.4).stopped)return 9;
 std::cout<<"Core transition and WBC contract passed\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
