// G1 entry point adapted from Unitree RL Lab deploy/robots/g1_29dof/main.cpp.
// Passive and DDS come from Unitree. All states and transitions are
// declared in config.yaml; Velocity and Mimic are excluded from the build.
#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "State_Policy.hpp"
#include "DeployConfig.hpp"
#include "State_SmoothStand.hpp"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;

int main(int argc, char** argv) {
    try {
        namespace po = boost::program_options;
        po::options_description options("G1 traj_loco walking + WBC camera");
        options.add_options()("help,h", "Show help")
            ("network,n", po::value<std::string>()->default_value("lo"), "DDS network interface")
            ("domain", po::value<int>()->default_value(0), "DDS domain (Unitree / unitree_mujoco default 0)")
            ("reference-smoothing", po::value<double>(), "WBC reference EMA time constant in seconds (overrides YAML)")
            ("config", po::value<std::string>()->default_value(std::string(TELEOP_ROOT) + "/deploy/g1/config/config.yaml"), "Unitree FSM config.yaml")
            ("deploy", po::value<std::string>(), "Override shared deploy.yaml")
            ("check-config", "Validate FSM, policy paths and model contracts without DDS");
        po::variables_map args;
        po::store(po::parse_command_line(argc, argv, options), args); po::notify(args);
        if (args.count("help")) { std::cout << options << std::endl; return 0; }
        spdlog::info("G1 deploy: executable={} | FSM config={}",
            param::get_bin_path().string(), std::filesystem::absolute(args["config"].as<std::string>()).string());
        param::proj_dir = std::filesystem::path(TELEOP_ROOT) / "deploy/g1";
        param::config_dir = param::proj_dir / "config";
        param::config = teleop::load_deployment(args["config"].as<std::string>(), param::proj_dir,
            args.count("deploy") ? std::filesystem::path(args["deploy"].as<std::string>()) : std::filesystem::path{});
        const auto enabled = param::config["FSM"]["_"];
        for (const auto& entry : enabled) {
            const auto name=entry.first.as<std::string>(), type=entry.second["type"].as<std::string>(name);
            if (!getFsmMap().count("State_"+type)) throw std::runtime_error("Unsupported configured FSM type: "+type);
            for (const auto& transition : param::config["FSM"][name]["transitions"]) {
                unitree::common::dsl::Parser parser(transition.second.as<std::string>());
                auto ast=parser.Parse(); unitree::common::dsl::Compile(*ast);
            }
            if (args.count("reference-smoothing") && (type=="WbcTeleop" || type=="TrajLoco")) {
                const double smoothing=args["reference-smoothing"].as<double>();
                if (!std::isfinite(smoothing)||smoothing<0.) throw std::runtime_error("Invalid reference smoothing time");
                param::config["FSM"][name]["reference_smoothing_s"]=smoothing;
            }
        }
        if (args.count("check-config")) {
            for (const auto& entry : enabled) {
                const auto name=entry.first.as<std::string>();
                std::cout << name << " id=" << entry.second["id"].as<int>();
                std::cout << "\n";
            }
            std::cout << "FSM and policy deploy.yaml configurations valid; no DDS initialized\n"; return 0;
        }
        const int domain = args["domain"].as<int>();
        if (domain < 0 || domain > 230) throw std::runtime_error("Invalid DDS domain");
        unitree::robot::ChannelFactory::Instance()->Init(domain, args["network"].as<std::string>());
        auto other = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        if (!other->isTimeout()) throw std::runtime_error("Another process is already publishing rt/lowcmd");
        other.reset();
        FSMState::lowcmd = std::make_unique<LowCmd_t>();
        FSMState::lowstate = std::make_shared<LowState_t>();
        FSMState::lowstate->set_timeout_ms(int(1000.*param::config["deployment"]["max_sensor_age_s"].as<double>()));
        spdlog::info("Waiting for G1 LowState on DDS domain {} / {}", domain, args["network"].as<std::string>());
        FSMState::lowstate->wait_for_connection();
        FSMState::lowcmd->msg_.mode_machine() = 5;
        if (!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) throw std::runtime_error("A 29-DoF G1 is required");
        auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
        const int passive=FSMStringMap.right.at("Passive");
        for (auto& state : fsm->states)
            std::stable_partition(state->registered_checks.begin(),state->registered_checks.end(),
                [passive](const auto& check){return check.second==passive;});
        const auto initial=param::config["deployment"]["initial_state"].as<std::string>();
        const auto first=std::find_if(fsm->states.begin(),fsm->states.end(),[&](const auto& state){return state->getStateString()==initial;});
        std::rotate(fsm->states.begin(),first,std::next(first));
        fsm->start();
        std::cout << "L2+Up: FixStand; repeat in FixStand: GetUp -> Walk. R1+X: Walk; R1+Y: Run. L1+Up: Idle.\n"
                     "R2+A: WBC Teleop WHB. R2+X: WBC TeleopArms. L2+B: Passive.\n"
                     "Left stick move, right stick turn; centered sticks stop walking.\n";
        while (true) std::this_thread::sleep_for(std::chrono::seconds(1));
    } catch (const std::exception& error) { std::cerr << "G1 controller: " << error.what() << std::endl; return 1; }
}
