#include <iostream>
#include "DeployConfig.hpp"
#include "WbcControl.hpp"
#include "UdpInput.hpp"
int main(){try{
 const auto root=std::filesystem::path(TELEOP_ROOT)/"deploy/g1/config";
 const auto cfg=teleop::load_deployment(root/"config.yaml",root)["FSM"];
 teleop::TrajLocoPolicy loco(cfg["Walk"]["policy_dir"].as<std::string>());teleop::WbcPolicy wbc(cfg["Teleop"]["policy_dir"].as<std::string>());
 teleop::RobotState state;state.q=loco.params().defaults;loco.reset(state);
 for(bool walking:{false,true}){double total=0.,maximum=0.;for(int i=0;i<110;++i){auto t=teleop::steady_seconds();const auto q=loco.step(state,0,walking?.25:0.,0);double ms=1000.*(teleop::steady_seconds()-t);if(!q.allFinite())return 1;if(i>=10){total+=ms;maximum=std::max(maximum,ms);}}
 std::cout<<"traj_loco "<<(walking?"walking":"idle")<<" mean/max="<<total/100.<<"/"<<maximum<<" ms\n";}
 state.q=wbc.params().defaults;teleop::WbcReference ref;ref.q=state.q;double total=0.,maximum=0.;for(int i=0;i<110;++i){auto t=teleop::steady_seconds();const auto q=wbc.step(ref,state);double ms=1000.*(teleop::steady_seconds()-t);if(!q.allFinite())return 1;if(i>=10){total+=ms;maximum=std::max(maximum,ms);}}
 std::cout<<"WBC mean/max="<<total/100.<<"/"<<maximum<<" ms\nSynthetic observations only; no DDS or hardware.\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
