#pragma once
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

namespace teleop {
// One CPU session per network, no hidden worker pool competing with control.
class OnnxPolicy {
    struct Input { std::string name; std::vector<int64_t> shape; size_t size; };
    Ort::Session session_{nullptr};
    std::vector<Input> inputs_;
    std::string output_;
    size_t output_size_=0;
    static Ort::Env& environment(){static Ort::Env env(ORT_LOGGING_LEVEL_WARNING,"g1-deploy");return env;}
public:
    OnnxPolicy(const std::filesystem::path& file,const std::map<std::string,size_t>& expected,size_t output) {
        Ort::SessionOptions options;options.SetIntraOpNumThreads(1);options.SetInterOpNumThreads(1);
        options.AddConfigEntry("session.intra_op.allow_spinning","0");
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_=Ort::Session(environment(),file.c_str(),options);
        if(session_.GetInputCount()!=expected.size()||session_.GetOutputCount()!=1)throw std::runtime_error("Wrong ONNX input/output count: "+file.string());
        Ort::AllocatorWithDefaultOptions allocator;
        for(size_t i=0;i<session_.GetInputCount();++i){
            Input in;in.name=session_.GetInputNameAllocated(i,allocator).get();
            auto type=session_.GetInputTypeInfo(i);auto info=type.GetTensorTypeAndShapeInfo();in.shape=info.GetShape();
            if(info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||in.shape.empty()||in.shape[0]>1)throw std::runtime_error("ONNX requires float32 and batch one");
            in.shape[0]=1;in.size=1;
            for(auto dim:in.shape){if(dim<=0)throw std::runtime_error("Unsupported dynamic ONNX feature dimension");in.size*=size_t(dim);}
            if(!expected.count(in.name)||expected.at(in.name)!=in.size)throw std::runtime_error("Wrong ONNX input contract: "+in.name+" in "+file.string());
            inputs_.push_back(std::move(in));
        }
        auto type=session_.GetOutputTypeInfo(0);auto info=type.GetTensorTypeAndShapeInfo();auto shape=info.GetShape();
        if(info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||shape.empty()||shape[0]>1)throw std::runtime_error("Invalid ONNX output");
        shape[0]=1;output_size_=1;for(auto dim:shape){if(dim<=0)throw std::runtime_error("Dynamic ONNX output");output_size_*=size_t(dim);}
        if(output_size_!=output)throw std::runtime_error("Wrong ONNX output width: "+file.string());
        output_=session_.GetOutputNameAllocated(0,allocator).get();
    }
    std::vector<float> run(const std::map<std::string,std::vector<float>>& data) {
        auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        std::vector<Ort::Value> tensors;std::vector<const char*> names;
        for(const auto& in:inputs_){
            const auto& values=data.at(in.name);
            if(values.size()!=in.size||!std::all_of(values.begin(),values.end(),[](float x){return std::isfinite(x);}))throw std::runtime_error("Invalid ONNX observation: "+in.name);
            names.push_back(in.name.c_str());
            tensors.push_back(Ort::Value::CreateTensor<float>(memory,const_cast<float*>(values.data()),values.size(),in.shape.data(),in.shape.size()));
        }
        const char* output=output_.c_str();auto result=session_.Run(Ort::RunOptions{nullptr},names.data(),tensors.data(),tensors.size(),&output,1);
        if(result[0].GetTensorTypeAndShapeInfo().GetElementCount()!=output_size_)throw std::runtime_error("ONNX output size changed");
        const auto* p=result[0].GetTensorData<float>();std::vector<float> values(p,p+output_size_);
        if(!std::all_of(values.begin(),values.end(),[](float x){return std::isfinite(x);}))throw std::runtime_error("Nonfinite ONNX action");
        return values;
    }
};
} // namespace teleop
