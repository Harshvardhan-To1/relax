/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * \file src/runtime/vm/tensor_cache_support.cc
 * \brief Runtime to support tensor cache file loading.
 *
 * This file provides a minimum support for tensor cache file loading.
 *
 * The main focus of this implementation is to enable loading
 * with minimum set of intermediate files while also being
 * compatible to some of the multi-shard files that are more
 * friendly in some of the environments.
 *
 * Tensor cache also provides a way to do system-wide
 * parameter sharing across multiple VMs.
 *
 * There are likely other ways to load the parameters ndarray-ache.
 * We will keep the impact minimum by puting it as a private
 * runtime builtin provide as in this file.
 */
#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/tensor.h>
#include <tvm/runtime/vm/tensor_cache_support.h>

#include <future>
#include <string>
#include <vector>

#include "../../support/utils.h"
#include "../file_utils.h"

namespace tvm {
namespace runtime {
namespace vm {

namespace json = tvm::ffi::json;

TensorCacheMetadata::FileRecord::ParamRecord JSONAsParamRecord(const json::Object& json) {
  std::vector<ffi::Shape::index_type> shape;
  {
    json::Array shape_json = json["shape"].cast<json::Array>();
    shape.reserve(shape_json.size());
    for (const ffi::Any& d : shape_json) {
      shape.push_back(d.cast<int64_t>());
    }
  }
  TensorCacheMetadata::FileRecord::ParamRecord result;
  std::string dtype = json["dtype"].cast<ffi::String>();
  result.name = json["name"].cast<ffi::String>();
  result.dtype = ffi::StringToDLDataType(dtype);
  result.format = json["format"].cast<ffi::String>();
  result.nbytes = json["nbytes"].cast<int64_t>();
  result.byte_offset = json["byteOffset"].cast<int64_t>();
  result.shape = ffi::Shape(std::move(shape));
  return result;
}

TensorCacheMetadata::FileRecord JSONAsFileRecord(const json::Object& json) {
  json::Array records = json["records"].cast<json::Array>();
  TensorCacheMetadata::FileRecord result;
  result.data_path = json["dataPath"].cast<ffi::String>();
  result.format = json["format"].cast<ffi::String>();
  result.nbytes = json["nbytes"].cast<int64_t>();
  result.records.reserve(records.size());
  for (const ffi::Any& item : records) {
    result.records.push_back(JSONAsParamRecord(item.cast<json::Object>()));
  }
  return result;
}

TensorCacheMetadata JSONAsTensorCacheMetadata(const json::Object& json) {
  json::Array records = json["records"].cast<json::Array>();
  TensorCacheMetadata result;
  result.records.reserve(records.size());
  for (const ffi::Any& item : records) {
    result.records.push_back(JSONAsFileRecord(item.cast<json::Object>()));
  }
  return result;
}

TensorCacheMetadata TensorCacheMetadata::LoadFromStr(const std::string& json_str,
                                                     const std::string& path) {
  ffi::String err;
  json::Value json_info = json::Parse(json_str, &err);
  if (!err.empty()) {
    TVM_FFI_THROW(InternalError) << "Failed to parse JSON: " << err
                                 << ". The JSON string is:" << json_str;
  }
  TVM_FFI_CHECK(json_info.as<json::Object>(), ValueError)
      << "The given string is not a JSON object: " << json_str;
  TensorCacheMetadata result = JSONAsTensorCacheMetadata(json_info.cast<json::Object>());
  result.path = path;
  return result;
}

TVM_RUNTIME_DLL TensorCacheMetadata TensorCacheMetadata::Load(const std::string& path) {
  std::string json_str;
  LoadBinaryFromFile(path + "/tensor-cache.json", &json_str);
  ffi::String err;
  json::Value json_info = json::Parse(json_str, &err);
  if (!err.empty()) {
    TVM_FFI_THROW(InternalError) << "Failed to parse JSON: " << err
                                 << ". The JSON string is:" << json_str;
  }
  TVM_FFI_CHECK(json_info.as<json::Object>(), ValueError)
      << "The given string is not a JSON object: " << json_str;
  TensorCacheMetadata result = JSONAsTensorCacheMetadata(json_info.cast<json::Object>());
  result.path = path;
  return result;
}

void CopyTensorFromBytes(Tensor param, const void* data, size_t nbytes,
                         ffi::Optional<Tensor>* staging_buffer) {
  Device device = param->device;
  if (device.device_type != kDLOpenCL || staging_buffer == nullptr) {
    param.CopyFromBytes(data, nbytes);
    return;
  }
  // Special handle for OpenCL runtime.
  // It creates a host side memory mirror, for every cl_mem that tries to copy data from host
  // which can cause memory issue. Her we use a large staging buffer to postpone deallocation
  if (staging_buffer->has_value()) {
    size_t curr_size = ffi::GetDataSize(*(staging_buffer->value().operator->()));
    if (curr_size < nbytes) {
      *staging_buffer = std::nullopt;
    }
  }
  if (!staging_buffer->has_value()) {
    *staging_buffer = Tensor::Empty(param.Shape(), param->dtype, param->device);
  }
  Tensor staging_view = staging_buffer->value().CreateView(param.Shape(), param->dtype);
  staging_view.CopyFromBytes(data, nbytes);
  param.CopyFrom(staging_view);
  DeviceAPI::Get(device)->StreamSync(device, nullptr);
}

/*!
 * \brief Load a parameter without waiting for the host-to-device copy to finish.
 * The copy may still read from `raw_data` after this returns, so the caller must
 * keep `raw_data` alive and unmodified until it synchronizes the device stream.
 * Syncing once per file instead of once per tensor lets backends such as Metal
 * batch the uploads instead of round-tripping to the GPU for every tensor.
 */
static Tensor LoadParamWithoutSync(const TensorCacheMetadata::FileRecord::ParamRecord& rec,
                                   Device device, const std::string* raw_data,
                                   ffi::Optional<Tensor>* staging_buffer) {
  Tensor arr = Tensor::Empty(rec.shape, rec.dtype, device);
  const char* data = raw_data->data() + rec.byte_offset;
  if (rec.dtype == DLDataType{kDLFloat, 32, 1} && rec.format == "f32-to-bf16") {
    // decode bf16 to f32; `decoded` is local, so this copy must be synchronous
    const uint16_t* buffer = reinterpret_cast<const uint16_t*>(data);
    std::vector<uint32_t> decoded(rec.nbytes / 2);
    for (size_t i = 0; i < decoded.size(); ++i) {
      decoded[i] = static_cast<uint32_t>(buffer[i]) << 16;
    }
    CopyTensorFromBytes(arr, decoded.data(), decoded.size() * sizeof(uint32_t), staging_buffer);
  } else if (device.device_type == kDLOpenCL && staging_buffer != nullptr) {
    CopyTensorFromBytes(arr, data, rec.nbytes, staging_buffer);
  } else {
    TVM_FFI_ICHECK_EQ(ffi::GetDataSize(*arr.operator->()), static_cast<size_t>(rec.nbytes))
        << "Size mismatch when loading parameter " << rec.name;
    DLTensor from = *arr.operator->();
    from.data = const_cast<char*>(data);
    from.device = Device{kDLCPU, 0};
    from.strides = nullptr;
    from.byte_offset = 0;
    DeviceAPI::Get(device)->CopyDataFromTo(&from, const_cast<DLTensor*>(arr.operator->()),
                                           nullptr);
  }
  return arr;
}

Tensor TensorCacheMetadata::FileRecord::ParamRecord::Load(
    Device device, const std::string* raw_data, ffi::Optional<Tensor>* staging_buffer) const {
  Tensor arr = LoadParamWithoutSync(*this, device, raw_data, staging_buffer);
  DeviceAPI::Get(device)->StreamSync(device, nullptr);
  return arr;
}

/*! \brief Load all parameters of a file whose contents are already in `raw_data`. */
static ffi::Array<Tensor> LoadFileRecordFromBuffer(const TensorCacheMetadata::FileRecord& file,
                                                   Device device, const std::string* raw_data,
                                                   ffi::Optional<Tensor>* staging_buffer) {
  TVM_FFI_CHECK_EQ(file.format, "raw-shard", ValueError) << "Only `raw-shard` format is supported";
  TVM_FFI_CHECK_EQ(file.nbytes, raw_data->length(), ValueError)
      << "Encountered an corrupted parameter shard. It means it is not downloaded "
         "completely or downloading is interrupted. Please try to download again.";
  ffi::Array<Tensor> result;
  result.reserve(file.records.size());
  for (const auto& nd_rec : file.records) {
    result.push_back(LoadParamWithoutSync(nd_rec, device, raw_data, staging_buffer));
  }
  // Wait for all copies from `raw_data` before the caller reuses it.
  DeviceAPI::Get(device)->StreamSync(device, nullptr);
  return result;
}

TVM_RUNTIME_DLL ffi::Array<Tensor> TensorCacheMetadata::FileRecord::Load(
    Device device,
    const std::string& path_prefix,  //
    std::string* raw_data_buffer,    //
    ffi::Optional<Tensor>* staging_buffer) const {
  LoadBinaryFromFile(path_prefix + "/" + this->data_path, raw_data_buffer);
  return LoadFileRecordFromBuffer(*this, device, raw_data_buffer, staging_buffer);
}

/*!
 * A Tensor cache to store pre-loaded arrays in the system.
 */
class TensorCache {
 public:
  static TensorCache* Global() {
    static TensorCache* inst = new TensorCache();
    return inst;
  }

  static void Update(ffi::String name, Tensor arr, bool override) {
    TensorCache* pool = Global();
    if (!override) {
      TVM_FFI_ICHECK_EQ(pool->pool_.count(name), 0)
          << "Name " << name << " already exists in the cache";
    }
    pool->pool_.Set(name, arr);
  }

  static ffi::Optional<Tensor> Get(ffi::String name) {
    TensorCache* pool = Global();
    auto it = pool->pool_.find(name);
    if (it != pool->pool_.end()) {
      return (*it).second;
    } else {
      return std::nullopt;
    }
  }

  static void Remove(ffi::String name) {
    TensorCache* pool = Global();
    pool->pool_.erase(name);
  }

  static void Clear() { Global()->pool_.clear(); }

  /*!
   * \brief Load parameters from path and append them.
   * \param cache_path The cache to path.
   * \param device_type The type of device to be loaded.
   * \param device_id The device id.
   */
  static void Load(const std::string& cache_path, int device_type, int device_id) {
    DLDevice device{static_cast<DLDeviceType>(device_type), device_id};
    TensorCacheMetadata metadata = TensorCacheMetadata::Load(cache_path);
    ffi::Optional<Tensor> staging_buffer;
    // Double buffering: file i+1 is read from disk on a background thread while
    // file i is uploaded to the device.
    std::string raw_data[2];
    std::future<void> next_file_read;  // declared after raw_data so it is joined first
    auto read_file = [&](size_t i) {
      LoadBinaryFromFile(cache_path + "/" + metadata.records[i].data_path, &raw_data[i % 2]);
    };
    ffi::Array<Tensor> params;
    for (size_t i = 0; i < metadata.records.size(); ++i) {
      const TensorCacheMetadata::FileRecord& shard_rec = metadata.records[i];
      try {
        if (i == 0) {
          read_file(0);
        } else {
          next_file_read.get();
        }
        // The other buffer held file i-1, whose upload has completed.
        if (i + 1 < metadata.records.size()) {
          next_file_read = std::async(std::launch::async, read_file, i + 1);
        }
        params = LoadFileRecordFromBuffer(shard_rec, device, &raw_data[i % 2], &staging_buffer);
      } catch (const std::runtime_error& e) {
        TVM_FFI_THROW(ValueError) << "Error when loading parameters from " << shard_rec.data_path
                                  << ": " << e.what();
      }
      int num_params = params.size();
      for (int i = 0; i < num_params; ++i) {
        Update(shard_rec.records[i].name, params[i], true);
      }
    }
  }

 private:
  ffi::Map<ffi::String, Tensor> pool_;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("vm.builtin.tensor_cache.get", TensorCache::Get)
      .def_packed("vm.builtin.tensor_cache.update",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    TVM_FFI_ICHECK(args.size() == 2 || args.size() == 3);
                    ffi::String name = args[0].cast<ffi::String>();
                    bool is_override = args.size() == 2 ? false : args[2].cast<bool>();

                    Tensor arr;
                    if (auto opt_nd = args[1].as<Tensor>()) {
                      arr = opt_nd.value();
                    } else {
                      // We support converting DLTensors to Tensors as RPC references are always
                      // DLTensors
                      auto tensor = args[1].cast<DLTensor*>();
                      std::vector<int64_t> shape;
                      for (int64_t i = 0; i < tensor->ndim; i++) {
                        shape.push_back(tensor->shape[i]);
                      }
                      arr = Tensor::Empty(shape, tensor->dtype, tensor->device);
                      arr.CopyFrom(tensor);
                      DeviceAPI::Get(arr->device)->StreamSync(arr->device, nullptr);
                    }

                    TensorCache::Update(name, arr, is_override);
                  })
      .def("vm.builtin.tensor_cache.remove", TensorCache::Remove)
      .def("vm.builtin.tensor_cache.clear", TensorCache::Clear)
      .def("vm.builtin.tensor_cache.load", TensorCache::Load);
}

// This param module node can be useful to get param dict in RPC mode
// when the remote already have loaded parameters from file.
class ParamModuleNode : public ffi::ModuleObj {
 public:
  const char* kind() const final { return "param_module"; }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final {
    if (name == "get_params") {
      auto params = params_;
      return ffi::Function([params](ffi::PackedArgs args, ffi::Any* rv) { *rv = params; });
    } else {
      return ffi::Function();
    }
  }

  static ffi::Array<Tensor> GetParams(const ffi::String& prefix, int num_params) {
    ffi::Array<Tensor> params;
    for (int i = 0; i < num_params || num_params == -1; ++i) {
      std::string name = prefix + "_" + std::to_string(i);
      auto opt = TensorCache::Get(name);
      if (opt) {
        params.push_back(opt.value());
      } else {
        if (num_params == -1) return params;
        TVM_FFI_THROW(InternalError) << "Cannot find " << name << " in cache";
      }
    }
    return params;
  }

  static ffi::Array<Tensor> GetParamByName(const ffi::Array<ffi::String>& names) {
    ffi::Array<Tensor> result;
    result.reserve(names.size());
    for (const ffi::String& name : names) {
      if (ffi::Optional<Tensor> opt = TensorCache::Get(name)) {
        result.push_back(opt.value());
      } else {
        TVM_FFI_THROW(ValueError) << "Cannot find parameter in cache: " << name;
      }
    }
    return result;
  }

  static ffi::Module Create(const std::string& prefix, int num_params) {
    auto n = ffi::make_object<ParamModuleNode>();
    n->params_ = GetParams(prefix, num_params);
    return ffi::Module(n);
  }

  static ffi::Module CreateByName(const ffi::Array<ffi::String>& names) {
    auto n = ffi::make_object<ParamModuleNode>();
    n->params_ = GetParamByName(names);
    return ffi::Module(n);
  }

 private:
  ffi::Array<Tensor> params_;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("vm.builtin.param_module_from_cache", ParamModuleNode::Create)
      .def("vm.builtin.param_module_from_cache_by_name", ParamModuleNode::CreateByName)
      .def("vm.builtin.param_array_from_cache", ParamModuleNode::GetParams)
      .def("vm.builtin.param_array_from_cache_by_name", ParamModuleNode::GetParamByName)
      .def_packed("vm.builtin.param_array_from_cache_by_name_unpacked",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    ffi::Array<ffi::String> names;
                    names.reserve(args.size());
                    for (int i = 0; i < args.size(); ++i) {
                      if (!args[i].try_cast<ffi::String>()) {
                        TVM_FFI_THROW(ValueError) << "Expect string as input, but get "
                                                  << args[i].GetTypeKey() << " at " << i;
                      }
                      names.push_back(args[i].cast<ffi::String>());
                    }
                    *rv = ParamModuleNode::GetParamByName(names);
                  });
}

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
