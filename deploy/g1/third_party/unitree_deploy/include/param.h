#pragma once
#include <filesystem>
#include <unistd.h>
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include <spdlog/spdlog.h>
namespace param {
inline std::filesystem::path proj_dir,config_dir;
inline YAML::Node config;
inline std::filesystem::path get_bin_path(){
    char path[4096];const auto size=readlink("/proc/self/exe",path,sizeof(path)-1);
    if(size<0)throw std::runtime_error("Cannot determine executable path");
    path[size]='\0';return path;
}
}
