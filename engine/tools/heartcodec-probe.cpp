// Fixed-input HeartCodec parity on the GGML CPU backend. No Python runtime.
#include "heartmula/codec-decode.h"
#include "backend.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <regex>
#include <sstream>
#include <type_traits>

namespace fs = std::filesystem;
using namespace heartcodec;

struct Npy {
    std::vector<int64_t> shape;
    std::string type;
    std::vector<uint8_t> data;
    explicit Npy(const fs::path & path) {
        std::ifstream in(path, std::ios::binary);
        require(bool(in), "cannot open " + path.string());
        char magic[6]; in.read(magic, 6);
        require(in && std::memcmp(magic, "\x93NUMPY", 6) == 0, "invalid NPY magic");
        uint8_t major = uint8_t(in.get()), minor = uint8_t(in.get());
        require((major == 1 || major == 2) && minor == 0, "unsupported NPY version");
        uint32_t length = 0;
        for (int b = 0; b < (major == 1 ? 2 : 4); ++b) {
            int byte = in.get(); require(byte >= 0, "truncated NPY header");
            length |= uint32_t(byte) << (8 * b);
        }
        require(length < 65536, "NPY header too large");
        std::string header(length, ' '); in.read(header.data(), length);
        require(bool(in), "truncated NPY header");
        std::smatch match;
        require(std::regex_search(header, match, std::regex("'descr':\\s*'([^']+)'")), "missing NPY dtype");
        type = match[1];
        require(type == "<f4" || type == "<i4" || type == "<i8", "expected little-endian f32, i32 or i64 NPY");
        require(std::regex_search(header, match, std::regex("'fortran_order':\\s*(True|False)")), "missing NPY order");
        bool fortran = match[1] == "True";
        require(std::regex_search(header, match, std::regex("'shape':\\s*\\(([^)]*)\\)")), "missing NPY shape");
        std::string dims = match[1]; std::replace(dims.begin(), dims.end(), ',', ' ');
        std::istringstream numbers(dims);
        size_t count = 1;
        for (int64_t n; numbers >> n;) {
            require(n > 0 && n < 100000000 && count <= 100000000 / size_t(n), "invalid NPY dimensions");
            shape.push_back(n); count *= size_t(n);
        }
        size_t bytes = count * (type == "<i8" ? 8 : 4);
        data.resize(bytes); in.read(reinterpret_cast<char *>(data.data()), std::streamsize(bytes));
        require(bool(in) && in.peek() == EOF, "NPY payload size mismatch");
        if (fortran) {
            auto column_major = data;
            size_t element = type == "<i8" ? 8 : 4;
            for (size_t i = 0; i < count; ++i) {
                size_t rest = i, offset = 0, stride = count;
                for (size_t d = shape.size(); d-- > 0;) {
                    stride /= size_t(shape[d]);
                    offset += (rest % size_t(shape[d])) * stride;
                    rest /= size_t(shape[d]);
                }
                std::memcpy(data.data() + i * element, column_major.data() + offset * element, element);
            }
        }
    }
    template<class T> std::vector<T> values() const {
        if constexpr (std::is_same<T, int64_t>::value) {
            if (type == "<i4") {
                std::vector<T> out(data.size() / 4);
                for (size_t i = 0; i < out.size(); ++i) {
                    int32_t value; std::memcpy(&value, data.data() + i * 4, 4); out[i] = T(value);
                }
                return out;
            }
        }
        require((std::is_same<T, float>::value && type == "<f4") ||
                (std::is_same<T, int64_t>::value && type == "<i8"), "NPY dtype mismatch");
        std::vector<T> out(data.size() / sizeof(T));
        std::memcpy(out.data(), data.data(), data.size()); return out;
    }
};

static void save_npy(const fs::path & path, const std::vector<float> & values, const std::vector<int64_t> & shape) {
    size_t count = 1;
    std::string dims;
    for (auto n : shape) { require(n > 0, "invalid output shape"); count *= size_t(n); dims += std::to_string(n) + ", "; }
    require(count == values.size(), "output shape mismatch");
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + dims + "), }";
    header.append((16 - ((10 + header.size() + 1) % 16)) % 16, ' '); header += '\n';
    std::ofstream out(path, std::ios::binary);
    out.write("\x93NUMPY\x01\x00", 8);
    uint16_t len = uint16_t(header.size()); out.write(reinterpret_cast<const char *>(&len), 2);
    out.write(header.data(), std::streamsize(header.size()));
    out.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * 4));
    require(bool(out), "cannot save " + path.string());
}

static void save_wav(const fs::path & path, const std::vector<float> & planar) {
    require(planar.size() % 2 == 0 && planar.size() < (UINT32_MAX - 36) / 4, "WAV size invalid");
    std::ofstream out(path, std::ios::binary);
    auto u16 = [&](uint16_t v) { out.write(reinterpret_cast<const char *>(&v), 2); };
    auto u32 = [&](uint32_t v) { out.write(reinterpret_cast<const char *>(&v), 4); };
    out.write("RIFF", 4); u32(uint32_t(36 + planar.size() * 4)); out.write("WAVEfmt ", 8);
    u32(16); u16(3); u16(2); u32(sample_rate); u32(sample_rate * 8); u16(8); u16(32);
    out.write("data", 4); u32(uint32_t(planar.size() * 4));
    size_t samples = planar.size() / 2;
    for (size_t t = 0; t < samples; ++t) for (int ch = 0; ch < 2; ++ch)
        out.write(reinterpret_cast<const char *>(&planar[ch * samples + t]), 4);
    require(bool(out), "cannot save WAV");
}

static bool compare(const std::vector<float> & got, const Npy & reference, double abs_limit, double rel_limit) {
    auto expected = reference.values<float>();
    require(got.size() == expected.size(), "reference shape mismatch");
    double maximum = 0, square = 0, norm = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        require(std::isfinite(got[i]) && std::isfinite(expected[i]), "nonfinite comparison");
        double d = double(got[i]) - expected[i];
        maximum = std::max(maximum, std::abs(d)); square += d * d; norm += double(expected[i]) * expected[i];
    }
    double relative = std::sqrt(square / std::max(norm, 1e-30));
    bool pass = maximum <= abs_limit && relative <= rel_limit;
    std::cout << (pass ? "PASS" : "FAIL") << " max_abs=" << maximum << " relative_l2=" << relative
              << " abs_limit=" << abs_limit << " rel_limit=" << rel_limit << '\n';
    return pass;
}

static void self_test() {
    for (int frames : {1, 8, 100, 371, 372, 373, 375, 400, 692, 3000}) {
        std::vector<int64_t> codes(size_t(frames * 8));
        for (int q = 0; q < 8; ++q) for (int i = 0; i < frames; ++i) codes[size_t(q * frames + i)] = q * 1000 + i % 1000;
        auto padded = pad_codes(codes, frames);
        auto length = padded_code_length(frames);
        require(length >= frames && length >= chunk_codes && (length - overlap_codes) % hop_codes == 0, "chunk geometry");
        for (int q = 0; q < 8; ++q) for (int64_t i = 0; i < length; ++i)
            require(padded[size_t(q * length + i)] == codes[size_t(q * frames + i % frames)], "repeat padding");
    }
    const int64_t n = chunk_codes * code_samples;
    std::vector<float> output(size_t(2 * n), 1), next(size_t(2 * n), 3);
    append_crossfade(output, next, n);
    require(output[hop_codes * code_samples] == 1 && output[n - 1] == 3 &&
            output.size() == size_t(2 * (chunk_codes + hop_codes) * code_samples), "crossfade endpoints");
    std::vector<float> noise(3 * 256, 0.5f), context(3 * 256, 2), mu(3 * 512, 0);
    Velocity constant = [](const std::vector<float> &, int64_t f, const std::vector<float> & ts) {
        std::vector<float> v(size_t(ts.size() * f * 256), 1);
        if (ts.size() == 2) std::fill(v.begin() + f * 256, v.end(), 3.0f);
        return v;
    };
    auto solved = solve_euler(constant, noise, context, mu, 3, 1, 10, 1.25f);
    require(solved[0] == 2 && std::abs(solved[256] - 4) < 2e-6f, "Euler CFG/context");
    bool refused = false;
    try { padded_code_length(0); } catch (const std::runtime_error &) { refused = true; }
    require(refused, "empty code rejection");
    refused = false;
    try { solve_euler(constant, noise, context, mu, 3, 0, 10, 1, [] { return true; }); }
    catch (const Cancelled &) { refused = true; }
    require(refused, "cancellation");
    refused = false;
    try { pad_codes(std::vector<int64_t>(8, 8192), 1); }
    catch (const std::runtime_error &) { refused = true; }
    require(refused, "invalid code rejection");
    refused = false;
    Weights rvq, flow, scalar;
    rvq.file_hash = "first"; flow.rvq_hash = "second"; scalar.rvq_hash = "first";
    try { decode(rvq, flow, scalar, nullptr, std::vector<int64_t>(8, 0), 1, {}); }
    catch (const std::runtime_error & e) { refused = std::string(e.what()).find("dependency hash") != std::string::npos; }
    require(refused, "mismatched codebook rejection");
    std::cout << "PASS geometry/repetition/crossfade/CFG/overlap/cancel/validation\n";
}

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "heartcodec-probe --self-test\n"
                         "heartcodec-probe --mode rvq|velocity|latent|scalar|decode --models DIR --fixtures CASE\n"
                         "  [--index N] [--out DIR] [--trace DIR] [--threads N] [--backend-dir DIR]\n"
                         "  [--atol NUMBER] [--rtol NUMBER] [--device gpu]\n"
                         "CPU unless --device gpu. CASE contains S1-style NPY intermediates and codes.\n"
                         "--out writes NPY and float32 48 kHz stereo WAV for scalar/decode.\n"
                         "--trace writes and checks every RVQ, velocity, latent and scalar output during decode.\n"
                         "Output directories must be outside CASE.\n";
            return 0;
        }
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (key == "--self-test") { self_test(); return 0; }
            require(key.rfind("--", 0) == 0 && i + 1 < argc, "expected --key value");
            args[key] = argv[++i];
        }
        require(args.count("--mode") && args.count("--models") && args.count("--fixtures"),
                "usage: heartcodec-probe --mode rvq|velocity|latent|scalar|decode --models DIR --fixtures CASE [--index N] [--out DIR] [--threads N] [--backend-dir DIR]");
        std::string mode = args.at("--mode");
        require(mode == "rvq" || mode == "velocity" || mode == "latent" || mode == "scalar" || mode == "decode", "unknown mode");
        fs::path models = args.at("--models"), fixtures = args.at("--fixtures");
        for (const char * key : {"--out", "--trace"}) if (args.count(key)) {
            auto destination = fs::weakly_canonical(fs::absolute(args.at(key)));
            for (auto parent = destination; !parent.empty();) {
                require(!fs::exists(parent) || !fs::equivalent(parent, fixtures), "outputs must be outside the reference fixtures");
                auto next = parent.parent_path(); if (next == parent) break; parent = next;
            }
        }
        int index = args.count("--index") ? std::stoi(args.at("--index")) : 0;
        require(index >= 0, "negative fixture index");
        auto array = [&](const std::string & name, int n = -1) {
            return Npy(fixtures / (n < 0 ? name + ".npy" : name + "/" + std::to_string(n) + ".npy"));
        };
        fs::path backend_dir = args.count("--backend-dir") ? fs::path(args.at("--backend-dir")) : fs::absolute(argv[0]).parent_path();
#ifdef _WIN32
        fs::path cpu_library = backend_dir / "ggml-cpu.dll";
#elif defined(__APPLE__)
        fs::path cpu_library = backend_dir / "libggml-cpu.dylib";
#else
        fs::path cpu_library = backend_dir / "libggml-cpu.so";
#endif
        // Load only CPU, never CUDA or Metal, while another slice holds the GPU.
        auto registered_cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        ggml_backend_reg_t reg = registered_cpu ? ggml_backend_dev_backend_reg(registered_cpu) : nullptr;
#ifdef _WIN32
        // Multi-variant builds export initialization from the suffixed DLLs.
        // ggml_backend_load checks each DLL's feature score before calling it.
        if (!reg) for (const char * variant : {"icelake", "cascadelake", "skylakex", "cannonlake", "alderlake", "haswell", "sandybridge", "sse42", "x64"}) {
            auto candidate = backend_dir / (std::string("ggml-cpu-") + variant + ".dll");
            if (fs::exists(candidate)) reg = ggml_backend_load(candidate.string().c_str());
            if (reg) break;
        }
#endif
        if (!reg) reg = ggml_backend_load(cpu_library.string().c_str());
        require(reg != nullptr, "cannot load CPU backend " + cpu_library.string());
        auto dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        require(dev != nullptr, "CPU backend is unavailable");
        auto backend = ggml_backend_dev_init(dev, nullptr);
        require(backend != nullptr, "CPU backend initialization failed");
        struct Free { ggml_backend_t b; ~Free() { ggml_backend_free(b); } } cleanup{backend};
        auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
        int threads = args.count("--threads") ? std::stoi(args.at("--threads")) : 8;
        require(threads > 0 && threads <= 256, "invalid thread count");
        if (set_threads) set_threads(backend, threads);
        // --device gpu: the engine's shared best device instead, for GPU codec parity.
        if (args.count("--device") && args.at("--device") == "gpu") {
            backend = backend_init("HeartCodec").backend;
            require(ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_CPU, "no GPU backend");
        }
        std::cout << "backend=" << ggml_backend_name(backend) << " threads=" << threads << std::endl;
        Weights rvq, flow, scalar;
        if (mode == "rvq" || mode == "latent" || mode == "decode") rvq.load((models / "heartcodec-rvq-F32.gguf").string(), "rvq", backend);
        if (mode == "velocity" || mode == "latent" || mode == "decode") flow.load((models / "heartcodec-flow-F16.gguf").string(), "flow", backend);
        if (mode == "scalar" || mode == "decode") scalar.load((models / "heartcodec-scalar-decoder-F16.gguf").string(), "scalar-decoder", backend);
        std::vector<float> result;
        bool intermediates_pass = true;
        std::vector<int64_t> shape;
        fs::path expected;
        auto start = std::chrono::steady_clock::now();
        if (mode == "rvq") {
            auto codes = array("rvq_indices", index);
            require(codes.shape.size() == 3 && codes.shape[0] == 1 && codes.shape[2] == 8, "expected [1,T,8] RVQ indices");
            auto ids = codes.values<int64_t>(); int64_t time = codes.shape[1];
            std::vector<int64_t> planar(ids.size());
            for (int q = 0; q < 8; ++q) for (int64_t t = 0; t < time; ++t) planar[size_t(q * time + t)] = ids[size_t(t * 8 + q)];
            result = rvq_decode(rvq, backend, planar, time); shape = {1, time, 512};
            expected = fixtures / "rvq_vectors" / (std::to_string(index) + ".npy");
        } else if (mode == "velocity") {
            auto x = array("flow_input", index); auto ts = array("flow_timestep", index);
            require(x.shape.size() == 3 && x.shape[2] == 1024 && ts.shape.size() == 1 && ts.shape[0] == x.shape[0], "flow fixture shape");
            result = flow_velocity(flow, backend, x.values<float>(), x.shape[1], ts.values<float>());
            shape = {x.shape[0], x.shape[1], 256};
            expected = fixtures / "flow_velocity" / (std::to_string(index) + ".npy");
        } else if (mode == "scalar") {
            auto x = array("scalar_input", index);
            require(x.shape.size() == 3 && x.shape[0] == 2 && x.shape[1] == 128, "scalar fixture shape");
            int64_t time = x.shape[2]; auto channels = x.values<float>();
            std::vector<float> latent(size_t(time * 256));
            for (int ch = 0; ch < 2; ++ch) for (int d = 0; d < 128; ++d) for (int64_t t = 0; t < time; ++t)
                latent[size_t(t * 256 + ch * 128 + d)] = channels[size_t((ch * 128 + d) * time + t)];
            result = scalar_decode(scalar, backend, latent, time); shape = {2, 1, time * 1920};
            expected = fixtures / "scalar_output" / (std::to_string(index) + ".npy");
        } else if (mode == "latent") {
            auto ids = array("rvq_indices", index); int64_t time = ids.shape.at(1);
            auto packed = ids.values<int64_t>(); std::vector<int64_t> planar(packed.size());
            for (int q = 0; q < 8; ++q) for (int64_t t = 0; t < time; ++t) planar[size_t(q * time + t)] = packed[size_t(t * 8 + q)];
            auto features = rvq_decode(rvq, backend, planar, time);
            auto mu = condition(flow, backend, features, time);
            auto context = array("true_latents", index).values<float>();
            int64_t overlap = array("incontext_length", index).values<int64_t>().at(0);
            auto noise = array("noise", 2 * index + 1).values<float>();
            Velocity velocity = [&](const std::vector<float> & x, int64_t f, const std::vector<float> & ts) {
                return flow_velocity(flow, backend, x, f, ts);
            };
            result = solve_euler(velocity, noise, context, mu, time * 2, overlap, 10, 1.25f);
            shape = {1, time * 2, 256}; expected = fixtures / "latent" / (std::to_string(index) + ".npy");
        } else {
            auto ids = array("codes");
            require(ids.shape.size() == 2 && ids.shape[0] == 8, "expected [8,T] codes");
            int noise_index = 0;
            Noise noise = [&](size_t count) { auto v = array("noise", noise_index++).values<float>(); require(v.size() == count, "fixture noise shape"); return v; };
            result = decode(rvq, flow, scalar, backend, ids.values<int64_t>(), ids.shape[1], noise, 10, 1.25f, {},
                            [](int n, int total) { std::cout << "chunk=" << n << '/' << total << std::endl; },
                            [&](const std::string & stage, int n, const std::vector<float> & values, const std::vector<int64_t> & dims) {
                                if (!args.count("--trace")) return;
                                fs::path dir = fs::path(args.at("--trace")) / stage; fs::create_directories(dir);
                                save_npy(dir / (std::to_string(n) + ".npy"), values, dims);
                                std::cout << "stage=" << stage << " index=" << n << ' ';
                                // Scalar rounding amplifies integrated latent drift. The
                                // isolated scalar mode still uses the strict component gate.
                                bool waveform = stage == "scalar_output";
                                bool pass = compare(values, Npy(fixtures / stage / (std::to_string(n) + ".npy")),
                                                    waveform ? 0.01 : 0.001, waveform ? 0.002 : 0.0001);
                                intermediates_pass = intermediates_pass && pass;
                            });
            shape = {2, ids.shape[1] * code_samples}; expected = fixtures / "waveform.npy";
        }
        std::cout << std::setprecision(9) << "mode=" << mode << " index=" << index << " seconds="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << '\n';
        if (args.count("--out")) {
            fs::path out = args.at("--out"); fs::create_directories(out);
            save_npy(out / (mode + "-" + std::to_string(index) + ".npy"), result, shape);
            if (mode == "decode" || mode == "scalar") save_wav(out / (mode + "-" + std::to_string(index) + ".wav"), result);
        }
        // End-to-end bounds are below one tenth of S1's multi-chunk
        // F32-to-F16 waveform drift (0.268 max, 0.0249 relative).
        double abs_limit = args.count("--atol") ? std::stod(args.at("--atol")) : (mode == "decode" ? 0.01 : 0.001);
        double rel_limit = args.count("--rtol") ? std::stod(args.at("--rtol")) : (mode == "decode" ? 0.002 : 0.0001);
        require(std::isfinite(abs_limit) && std::isfinite(rel_limit) && abs_limit >= 0 && rel_limit >= 0, "invalid tolerance");
        return compare(result, Npy(expected), abs_limit, rel_limit) && intermediates_pass ? 0 : 1;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n'; return 2;
    }
}
