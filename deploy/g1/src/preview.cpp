// Observe interpolated camera references; never sends robot commands.
#include <fstream>
#include <iostream>
#include <thread>
#include "UdpInput.hpp"
int main(int argc,char** argv){try{
 using namespace teleop;std::string host="127.0.0.1",output;int port=9870;double seconds=60.,hz=50.;
 for(int i=1;i<argc;++i){std::string key=argv[i];if(key=="--help"){std::cout<<"--host IP --port N --seconds N --hz N --output JSONL\n";return 0;}if(i+1>=argc)throw std::runtime_error("Missing option value");std::string value=argv[++i];
 if(key=="--host")host=value;else if(key=="--port")port=std::stoi(value);else if(key=="--seconds")seconds=std::stod(value);else if(key=="--hz")hz=std::stod(value);else if(key=="--output")output=value;else throw std::runtime_error("Unknown option: "+key);}
 if(!std::isfinite(seconds)||seconds<=0.)throw std::runtime_error("Invalid duration");
 ReferenceStream stream(hz);UdpInput input(host,port);std::ofstream file;if(!output.empty())file.open(output);
 const double began=steady_seconds();double next=began;
 while(steady_seconds()-began<seconds){input.poll(stream);const auto ref=stream.sample(steady_seconds());if(ref.stopped)break;
 if(ref.ready){nlohmann::json row={{"frame_a",ref.previous_frame},{"frame_b",ref.next_frame},{"alpha",ref.alpha},{"q",std::vector<double>(ref.q.data(),ref.q.data()+29)},{"active",ref.active},{"diagnostics",stream.diagnostics(steady_seconds())}};if(file.is_open())file<<row.dump()<<'\n';else std::cout<<row.dump()<<'\n';}
 next+=1./hz;std::this_thread::sleep_until(std::chrono::steady_clock::time_point(std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(next))));}
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
