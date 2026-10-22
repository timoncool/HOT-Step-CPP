// Merge is the opt-in runtime specification. Set GGML_BACKEND=CPU, CUDA0 or
// Vulkan0. No model downloads are needed. The first test fails on the old
// loader, which accepted declared DoRA without m and changed pending weights.
#include "adapter-runtime.h"
#include "backend.h"
#include "train/st-write.h"
#include "train/dit-resume.h"
#include "train/lm-resume.h"
#include <filesystem>
#include <fstream>
#include <limits>

static void jl(const char *, ...) {} // trainer event sink, unused by this fixture

static int failures=0;
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); ++failures; } } while(0)
static const std::string name="decoder.layers.0.self_attn.q_proj.weight";
static const std::string stem="base_model.model.layers.0.self_attn.q_proj";
static const std::filesystem::path root=std::filesystem::path(".tmp")/"hotstep545-dora-runtime";
static constexpr int in=32, out=3;
static std::vector<float> w, a, b, m;

static std::string write_adapter(const char * leaf, bool dora, int bad=0, float mult=1) {
    const auto dir=root/leaf;
    std::filesystem::create_directories(dir);
    auto mb=m;
    for (float & f:mb) f*=mult;
    if (bad==3) mb[0]=std::numeric_limits<float>::quiet_NaN();
    std::vector<STWTensor> ts{{stem+".lora_A.weight",{1,in},a.data()},
                             {stem+".lora_B.weight",{out,1},b.data()}};
    if (dora && bad!=1) ts.push_back({stem+".lora_magnitude_vector.default.weight",{bad==2?out-1:out},mb.data()});
    CHECK(st_write_file((dir/"adapter_model.safetensors").string().c_str(),ts,{},STW_F32));
    std::ofstream(dir/"adapter_config.json") << "{\"use_dora\":" << (dora?"true":"false") << ",\"r\":1,\"lora_alpha\":1}";
    return dir.string();
}

static std::string write_lokr() {
    const auto dir=root/"lokr";
    std::filesystem::create_directories(dir);
    const std::string prefix="lycoris_layers_0_self_attn_q_proj";
    float one=1;
    std::vector<float> full(in*out);
    for(int row=0;row<out;++row) for(int col=0;col<in;++col) full[row*in+col]=b[row]*a[col];
    CHECK(st_write_file((dir/"lokr_weights.safetensors").string().c_str(),
        {{prefix+".lokr_w1",{1,1},&one},{prefix+".lokr_w2",{out,in},full.data()},
         {prefix+".alpha",{1},&one},{prefix+".dora_scale",{out},m.data()}},
        {{"lokr_config","{\"linear_dim\":1}"}},STW_F32));
    return dir.string();
}

struct Weights {
    WeightCtx ctx{};
    ggml_tensor * tensor;
    std::vector<uint8_t> raw;
    Weights(ggml_type type=GGML_TYPE_F32) {
        wctx_init(&ctx,4);
        tensor=ggml_new_tensor_2d(ctx.ctx,type,in,out);
        ggml_set_name(tensor,name.c_str());
        if(type==GGML_TYPE_F32) ctx.pending.push_back({tensor,w.data(),w.size()*sizeof(float),0});
        else {
            raw.resize(ggml_nbytes(tensor));
            ggml_quantize_chunk(type,w.data(),raw.data(),0,out,in,nullptr);
            ctx.pending.push_back({tensor,raw.data(),raw.size(),0});
        }
    }
    ~Weights() { wctx_free(&ctx); }
};

static std::vector<float> read_delta(DiTLoRA & lora) {
    ggml_tensor * t=lora.layers[0].sa_q.delta;
    CHECK(t!=nullptr);
    if (!t) return {};
    std::vector<uint8_t> raw(ggml_nbytes(t));
    std::vector<float> v((size_t)ggml_nelements(t));
    ggml_backend_tensor_get(t,raw.data(),0,raw.size());
    ggml_get_type_traits(t->type)->to_float(raw.data(),v.data(),(int64_t)v.size());
    return v;
}

static void compare(const WeightSource & ws, ggml_backend_t backend,
                    const std::vector<AdapterSpec> & stack, const char * quant="bf16", bool lowrank=false,
                    ggml_type type=GGML_TYPE_F32) {
    Weights merged(type), base(type);
    std::vector<float> original=w;
    if(type!=GGML_TYPE_F32) ggml_get_type_traits(type)->to_float(base.raw.data(),original.data(),(int64_t)original.size());
    for (const auto & ad:stack) CHECK(adapter_merge(&merged.ctx,ws,ad.path.c_str(),ad.scale,backend,true));
    CHECK(wctx_alloc(&base.ctx,backend));
    DiTLoRA lora{};
    g_hotstep_params.adapter_mode=lowrank?"runtime_lowrank":"runtime";
    g_hotstep_params.adapter_runtime_dora=true;
    g_hotstep_params.adapter_runtime_quant=quant;
    CHECK(adapter_load_runtime_stack(&lora,&base.ctx,ws,stack,g_hotstep_params.adapter_group_scales,backend));
    auto delta=read_delta(lora);
    const float * expected=(const float *)merged.ctx.pending[0].src;
    double worst=0, magnitude=0;
    for (size_t i=0;i<delta.size();++i) {
        CHECK(std::isfinite(delta[i]));
        worst=std::max(worst,std::abs((double)original[i]+delta[i]-expected[i]));
        magnitude=std::max(magnitude,std::abs((double)expected[i]-original[i]));
    }
    const double tol=std::string(quant)=="q8_0"?0.01:0.004;
    CHECK(worst<=tol*std::max(1.0,magnitude));
    std::vector<uint8_t> unchanged(ggml_nbytes(base.tensor));
    ggml_backend_tensor_get(base.tensor,unchanged.data(),0,unchanged.size());
    CHECK(memcmp(unchanged.data(),type==GGML_TYPE_F32?(const void *)w.data():base.raw.data(),unchanged.size())==0);
    printf("runtime/merge %zu adapter(s), %s%s max_abs=%.8g\n",stack.size(),quant,lowrank?" lowrank fallback":"",worst);
    dit_lora_free(&lora);
}

static void check_lm_resume_norm(const BackendPair & bp) {
    Weights base;
    CHECK(wctx_alloc(&base.ctx,bp.backend));
    Qwen3LM model{};
    model.layers[0].q_proj=base.tensor;
    LmLora bank{};
    bank.model=&model; bank.dora=true; bank.layer_lo=0; bank.layer_hi=1;
    ggml_init_params ip{ggml_tensor_overhead()*5+1024,nullptr,true};
    bank.ctx=ggml_init(ip);
    auto & pr=bank.layers[0].p[QW_LORA_Q];
    pr.A=ggml_new_tensor_2d(bank.ctx,GGML_TYPE_F32,in,1);
    pr.B=ggml_new_tensor_2d(bank.ctx,GGML_TYPE_F32,1,out);
    pr.m=ggml_new_tensor_1d(bank.ctx,GGML_TYPE_F32,out);
    pr.nrm=ggml_new_tensor_1d(bank.ctx,GGML_TYPE_F32,out);
    pr.scale=1;
    bank.buf=ggml_backend_alloc_ctx_tensors(bank.ctx,bp.backend);
    std::vector<float> initial(out), expected(out), held(out);
    for(int row=0;row<out;++row) {
        double norm0=0,norm1=0;
        for(int col=0;col<in;++col) {
            double v=w[row*in+col]; norm0+=v*v;
            v+=b[row]*a[col]; norm1+=v*v;
        }
        initial[row]=(float)std::sqrt(norm0);
        expected[row]=(float)std::sqrt(norm1);
    }
    ggml_backend_tensor_set(pr.nrm,initial.data(),0,out*sizeof(float));
    const auto dir=root/"lm-resume";
    std::filesystem::create_directories(dir);
    const std::string key="base_model.model.model.layers.0.self_attn.q_proj";
    CHECK(st_write_file((dir/"adapter_model.safetensors").string().c_str(),
        {{key+".lora_A.weight",{1,in},a.data()},{key+".lora_B.weight",{out,1},b.data()},
         {key+".lora_magnitude_vector.weight",{out},m.data()}},{},STW_F32));
    int loaded=0; std::string err;
    CHECK(lm_resume_load(&bank,dir.string(),&loaded,&err));
    CHECK(loaded==3);
    ggml_backend_tensor_get(pr.nrm,held.data(),0,out*sizeof(float));
    CHECK(held==initial && held!=expected); // repro: resume leaves the fresh norm
    auto sched=backend_sched_new(bp,256);
    CHECK(lm_lora_dora_refresh(&bank,sched,&err));
    ggml_backend_tensor_get(pr.nrm,held.data(),0,out*sizeof(float));
    for(int row=0;row<out;++row) CHECK(std::isfinite(held[row]) && std::abs(held[row]-expected[row])<1e-5f);
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(bank.buf);
    ggml_free(bank.ctx);
    puts("PASS: resumed LM A/B/m need and receive a pre-window norm refresh");
}

int main() {
    if (!getenv("GGML_BACKEND")) {
#ifdef _WIN32
        _putenv_s("GGML_BACKEND","CPU");
#else
        setenv("GGML_BACKEND","CPU",1);
#endif
    }
    std::filesystem::create_directories(root);
    w.resize(in*out); a.resize(in); b={0.25f,-0.5f,0.75f}; m={2,3,4};
    for(size_t i=0;i<w.size();++i) w[i]=(float)(int(i%11)-5)*0.125f;
    for(size_t i=0;i<a.size();++i) a[i]=(float)(int(i%5)-2)*0.25f;
    CHECK(st_write_file((root/"base.safetensors").string().c_str(),{{name,{out,in},w.data()}},{},STW_F32));
    STMulti sm; sm.shards.emplace_back();
    CHECK(st_open(&sm.shards.back(),(root/"base.safetensors").string().c_str()));
    WeightSource ws; ws.is_st=true; ws.sm=&sm;
    BackendPair bp=backend_init("DoRA-runtime-fixture");
    for(int bad=1;bad<=3;++bad) {
        const auto path=write_adapter(("bad"+std::to_string(bad)).c_str(),true,bad);
        Weights weights;
        CHECK(!adapter_merge(&weights.ctx,ws,path.c_str(),1,bp.backend,true));
        CHECK(weights.ctx.pending[0].src==w.data() && weights.ctx.staging.empty());
    }
    const auto bad_config=write_adapter("bad-config",true);
    std::ofstream(std::filesystem::path(bad_config)/"adapter_config.json") << "{invalid";
    CHECK(!adapter_dora_preflight_file(bad_config.c_str(),ws));
    const auto first=write_adapter("first",true);
    const auto second=write_adapter("second",true,0,0.8f);
    const auto plain=write_adapter("plain",false);
    const auto lokr=write_lokr();
    for(float strength:{0.0f,0.5f,1.0f}) compare(ws,bp.backend,{{first,strength}});
    compare(ws,bp.backend,{{first,0.5f},{second,0.75f}});
    compare(ws,bp.backend,{{plain,0.5f},{first,0.75f},{plain,-0.25f}});
    compare(ws,bp.backend,{{first,1.0f}},"q8_0");
    compare(ws,bp.backend,{{first,1.0f}},"bf16",true);
    compare(ws,bp.backend,{{first,1.0f}},"bf16",false,GGML_TYPE_Q8_0);
    compare(ws,bp.backend,{{lokr,0.5f}});
    compare(ws,bp.backend,{{first,0.5f},{lokr,0.75f}});
    // Resume rejects a declared-DoRA file before touching A/B/m, while a
    // plain-LoRA source can still initialize DoRA with its fresh magnitude.
    DitAdapterLora ad;
    ad.lo=0; ad.hi=1; ad.n_sites=1; ad.dora=true; ad.layers.resize(1);
    ggml_init_params ip{ggml_tensor_overhead()*4+1024,nullptr,true};
    ad.ctx=ggml_init(ip);
    auto & pair=ad.layers[0][0];
    pair.A=ggml_new_tensor_2d(ad.ctx,GGML_TYPE_F32,in,1);
    pair.B=ggml_new_tensor_2d(ad.ctx,GGML_TYPE_F32,1,out);
    pair.m=ggml_new_tensor_1d(ad.ctx,GGML_TYPE_F32,out);
    ad.buf=ggml_backend_alloc_ctx_tensors(ad.ctx,bp.backend);
    std::vector<float> sentinel(in,17), got(in);
    ggml_backend_tensor_set(pair.A,sentinel.data(),0,in*sizeof(float));
    DitResumeStats stats;
    std::string err;
    CHECK(!dit_resume_load_lora(&ad,(root/"bad1").string(),&stats,&err));
    ggml_backend_tensor_get(pair.A,got.data(),0,in*sizeof(float));
    CHECK(got==sentinel);
    CHECK(dit_resume_load_lora(&ad,first,&stats,&err));
    std::vector<float> loaded_m(out);
    ggml_backend_tensor_get(pair.m,loaded_m.data(),0,out*sizeof(float));
    CHECK(loaded_m==m);
    CHECK(dit_resume_load_lora(&ad,plain,&stats,&err));
    ggml_backend_tensor_get(pair.m,loaded_m.data(),0,out*sizeof(float));
    CHECK(loaded_m==m);
    ad.free();
    auto runtime=[&](const std::string & path,bool opt) {
        Weights base; CHECK(wctx_alloc(&base.ctx,bp.backend));
        DiTLoRA lora{};
        g_hotstep_params.adapter_mode="runtime";
        g_hotstep_params.adapter_runtime_quant="bf16";
        g_hotstep_params.adapter_runtime_dora=opt;
        CHECK(adapter_load_runtime(&lora,&base.ctx,ws,path.c_str(),0.5f,g_hotstep_params.adapter_group_scales,bp.backend));
        auto v=read_delta(lora); dit_lora_free(&lora); return v;
    };
    // Opt-out retains LoRA-only application of a DoRA file. Plain LoRA keeps
    // exactly the same runtime bytes with the flag in either position.
    CHECK(runtime(first,false)==runtime(plain,false));
    CHECK(runtime(plain,false)==runtime(plain,true));
    g_hotstep_params.adapter_group_scales.self_attn=0.25f;
    compare(ws,bp.backend,{{first,0.5f}});
    check_lm_resume_norm(bp);
    backend_release(bp.backend,bp.cpu_backend);
    st_close(&sm.shards.back());
    printf("%s: DoRA runtime fixture (%d failures)\n",failures?"FAIL":"PASS",failures);
    return failures?1:0;
}
