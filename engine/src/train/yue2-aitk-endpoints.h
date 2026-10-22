#pragma once

// Native YuE2 endpoint seams. These are intentionally small graph calls:
// embedding lookup, AR final RMSNorm, NAR latent frontend, and NAR final
// projection. They own only temporary graph state and never update weights.

#include "yue2-aitk-graph.h"
#include "yue2-aitk-model.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace yue2_aitk_endpoints_detail {
struct Runtime { ggml_context * ctx=nullptr; ggml_backend_buffer_t buffer=nullptr; ~Runtime(){if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx);} };
inline bool fail(std::string *e,const char*m){if(e)*e=m;return false;} inline bool fail(std::string*e,const std::string&m){if(e)*e=m;return false;}
inline bool finite(const float*p,size_t n){if(!p)return false;for(size_t i=0;i<n;++i)if(!std::isfinite(p[i]))return false;return true;}
inline bool init(ggml_backend_t b,Runtime*r,std::string*e){if(!b||!r)return fail(e,"null endpoint backend");ggml_init_params p{};p.mem_size=32ull*1024ull*1024ull;p.no_alloc=true;r->ctx=ggml_init(p);return r->ctx?true:fail(e,"endpoint context allocation failed");}
inline bool alloc(Runtime*r,ggml_backend_t b,std::string*e){r->buffer=ggml_backend_alloc_ctx_tensors(r->ctx,b);return r->buffer?true:fail(e,"endpoint backend allocation failed");}
inline bool support(ggml_backend_t b,ggml_cgraph*g,std::string*e){auto n=ggml_graph_n_nodes(g);auto v=ggml_graph_nodes(g);for(int i=0;i<n;++i)if(!ggml_backend_supports_op(b,v[i]))return fail(e,"endpoint graph node is unsupported");return true;}
inline bool compute(ggml_backend_t b,ggml_cgraph*g,std::string*e){ggml_graph_reset(g);return ggml_backend_graph_compute(b,g)==GGML_STATUS_SUCCESS?true:fail(e,"endpoint graph compute failed");}
inline ggml_tensor*f32(ggml_context*c,ggml_tensor*x){return x->type==GGML_TYPE_F32?x:ggml_cast(c,x,GGML_TYPE_F32);}
inline bool shape2(const float*p,int64_t a,int64_t b){return a>0&&b>0&&p&&finite(p,size_t(a)*size_t(b));}
}

struct Yue2AitkEndpointHost { std::vector<float> values; std::vector<float> dx; };

class Yue2AitkEndpoints {
public:
    static bool token_embedding(ggml_backend_t backend, const Yue2AitkModel & model,
                                const int32_t * ids, int64_t count, Yue2AitkEndpointHost * out,
                                std::string * error=nullptr) {
        using namespace yue2_aitk_endpoints_detail; if(!out||!ids||count<=0||count>100000)return fail(error,"invalid embedding request");
        ggml_tensor* emb=model.embedding(); if(!emb||emb->ne[0]!=2048)return fail(error,"embedding is unavailable"); Runtime r;if(!init(backend,&r,error))return false;
        for (int64_t i=0;i<count;++i) if (ids[i] < 0 || ids[i] >= emb->ne[1]) return fail(error,"embedding id is out of range");
        ggml_tensor* id=ggml_new_tensor_1d(r.ctx,GGML_TYPE_I32,count); if(!id)return fail(error,"embedding id allocation failed"); ggml_tensor* y=ggml_get_rows(r.ctx,emb,id); if(!y)return fail(error,"embedding lookup construction failed"); y=f32(r.ctx,y);
        ggml_cgraph*g=ggml_new_graph_custom(r.ctx,2048,true);if(!g)return fail(error,"embedding graph allocation failed");ggml_build_forward_expand(g,y);if(!support(backend,g,error)||!alloc(&r,backend,error))return false;ggml_backend_tensor_set(id,ids,0,ggml_nbytes(id));if(!compute(backend,g,error))return false;
        out->values.resize(ggml_nelements(y));ggml_backend_tensor_get(y,out->values.data(),0,ggml_nbytes(y));return true;
    }

    static bool ar_final_norm(ggml_backend_t backend, const Yue2AitkModel & model,
                              const float * hidden, const float * dy, int64_t tokens,
                              Yue2AitkEndpointHost * out, std::string * error=nullptr);

    static bool nar_frontend(ggml_backend_t backend, const Yue2AitkModel & model,
                             const float * latents, int64_t frames, float timestep,
                             Yue2AitkEndpointHost * out, std::string * error=nullptr) {
        using namespace yue2_aitk_endpoints_detail; if(!out||!shape2(latents,64,frames)||!std::isfinite(timestep))return fail(error,"invalid NAR frontend request");ggml_tensor*vae=model.ordinary("model.diffusion_model.vae2llm.weight"),*vb=model.ordinary("model.diffusion_model.vae2llm.bias"),*tb0=model.ordinary("model.diffusion_model.time_embedder.mlp.0.bias"),*tb2=model.ordinary("model.diffusion_model.time_embedder.mlp.2.bias"),*pos=model.ordinary("model.diffusion_model.latent_pos_embed.pe");if(!vae||!vb||!tb0||!tb2||!pos)return fail(error,"NAR frontend weights unavailable");const int64_t n=frames+2;Runtime r;if(!init(backend,&r,error))return false;ggml_tensor*in=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,64,n);if(!in)return fail(error,"NAR frontend input allocation failed");std::vector<float> padded(size_t(64*n),0.0f),freq(256);for(int64_t j=0;j<frames;++j)for(int64_t i=0;i<64;++i)padded[size_t(i+64*(j+1))]=ggml_bf16_to_fp32(ggml_fp32_to_bf16(latents[size_t(i+64*j)]));const float tbf=ggml_bf16_to_fp32(ggml_fp32_to_bf16(timestep));for(int i=0;i<128;++i){float f=std::exp(-std::log(10000.0f)*float(i)/128.0f);freq[i]=ggml_bf16_to_fp32(ggml_fp32_to_bf16(std::cos(tbf*f)));freq[128+i]=ggml_bf16_to_fp32(ggml_fp32_to_bf16(std::sin(tbf*f)));} // Toolkit casts the time embedding input to BF16
        ggml_tensor*time_in=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,256,1);if(!time_in)return fail(error,"timestep allocation failed");
        ggml_tensor*x=yue2_aitk_graph::round(r.ctx,ggml_add(r.ctx,ggml_mul_mat(r.ctx,vae,in),f32(r.ctx,vb))); // rounded BF16 vae2llm
        ggml_tensor*tm=yue2_aitk_graph::linear(r.ctx,model.time0(),time_in,nullptr,tb0);tm=yue2_aitk_graph::round(r.ctx,ggml_silu(r.ctx,tm));tm=yue2_aitk_graph::linear(r.ctx,model.time2(),tm,nullptr,tb2);ggml_tensor*tm2=ggml_repeat(r.ctx,tm, x);x=yue2_aitk_graph::round(r.ctx,ggml_add(r.ctx,x,tm2));ggml_tensor*pv=ggml_cont(r.ctx,ggml_view_2d(r.ctx,pos,2048,n,pos->nb[1],0));x=yue2_aitk_graph::round(r.ctx,ggml_add(r.ctx,x,f32(r.ctx,pv)));
        ggml_cgraph*g=ggml_new_graph_custom(r.ctx,4096,true);if(!g)return fail(error,"NAR frontend graph allocation failed");ggml_build_forward_expand(g,x);if(!support(backend,g,error)||!alloc(&r,backend,error))return false;ggml_backend_tensor_set(in,padded.data(),0,ggml_nbytes(in));std::vector<float>time_host(256,0);for(int i=0;i<128;++i){time_host[i]=freq[i];time_host[128+i]=freq[128+i];}ggml_backend_tensor_set(time_in,time_host.data(),0,ggml_nbytes(time_in));if(!compute(backend,g,error))return false;out->values.resize(ggml_nelements(x));ggml_backend_tensor_get(x,out->values.data(),0,ggml_nbytes(x));return true;
    }

    static bool nar_final(ggml_backend_t backend, const Yue2AitkModel & model,
                          const float * hidden, const float * dy, int64_t tokens,
                          Yue2AitkEndpointHost * out, std::string * error=nullptr,
                          const ggml_tensor * device_hidden=nullptr, ggml_tensor * device_dx=nullptr) {
        // device_hidden replaces hidden (copied D2D); device_dx receives dx (D2D) and out->dx stays empty.
        using namespace yue2_aitk_endpoints_detail; if(!out||(device_hidden?(device_hidden->type!=GGML_TYPE_F32||ggml_nelements(device_hidden)!=2048*tokens):!shape2(hidden,2048,tokens))||!shape2(dy,64,tokens)||!model.nar().final_norm)return fail(error,"invalid NAR final request");ggml_tensor*bias=model.ordinary("model.diffusion_model.llm2vae.bias");if(!bias)return fail(error,"NAR output bias unavailable");Runtime r;if(!init(backend,&r,error))return false;ggml_tensor*x=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,2048,tokens),*up=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,64,tokens);if(!x||!up)return fail(error,"NAR final input allocation failed");ggml_set_param(x);ggml_tensor*n=yue2_aitk_graph::rms(r.ctx,yue2_aitk_graph::round(r.ctx,x),model.nar().final_norm,1e-6f);ggml_tensor*y=yue2_aitk_graph::linear(r.ctx,model.llm2vae(),n,nullptr,bias);ggml_tensor*loss=ggml_sum(r.ctx,ggml_mul(r.ctx,y,up));ggml_cgraph*g=ggml_new_graph_custom(r.ctx,8192,true);if(!loss||!g)return fail(error,"NAR final graph allocation failed");ggml_build_forward_expand(g,loss);ggml_set_loss(loss);ggml_build_backward_expand(r.ctx,g,nullptr);if(!support(backend,g,error)||!alloc(&r,backend,error))return false;if(device_hidden)ggml_backend_tensor_copy(const_cast<ggml_tensor*>(device_hidden),x);else ggml_backend_tensor_set(x,hidden,0,ggml_nbytes(x));ggml_backend_tensor_set(up,dy,0,ggml_nbytes(up));if(!compute(backend,g,error))return false;out->values.resize(ggml_nelements(y));ggml_backend_tensor_get(y,out->values.data(),0,ggml_nbytes(y));ggml_tensor*gx=ggml_graph_get_grad(g,x);if(!gx)return fail(error,"NAR final gradient missing");if(device_dx){if(ggml_nbytes(device_dx)!=ggml_nbytes(gx))return fail(error,"NAR final dx shape mismatch");out->dx.clear();ggml_backend_tensor_copy(gx,device_dx);return true;}out->dx.resize(ggml_nelements(gx));ggml_backend_tensor_get(gx,out->dx.data(),0,ggml_nbytes(gx));return true;
    }
};

// The AR final norm's graph, held open so its input, output and dy can stay on
// the device (Yue2AitkArHeadRows); ar_final_norm is the host wrapper over it.
struct Yue2AitkArNormGraph {
    yue2_aitk_endpoints_detail::Runtime r; ggml_tensor *x=nullptr,*up=nullptr,*y=nullptr,*gx=nullptr; ggml_cgraph *g=nullptr;
    bool build(ggml_backend_t backend,const Yue2AitkModel & model,int64_t tokens,std::string * error) {
        using namespace yue2_aitk_endpoints_detail; if(!model.ar().final_norm)return fail(error,"invalid AR final norm request"); if(!init(backend,&r,error))return false;
        x=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,2048,tokens);up=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,2048,tokens);if(!x||!up)return fail(error,"AR final norm input allocation failed");ggml_set_param(x);
        y=yue2_aitk_graph::rms(r.ctx,yue2_aitk_graph::round(r.ctx,x),model.ar().final_norm,1e-6f);ggml_tensor*loss=ggml_sum(r.ctx,ggml_mul(r.ctx,y,up));g=ggml_new_graph_custom(r.ctx,4096,true);if(!loss||!g)return fail(error,"AR final norm graph allocation failed");ggml_build_forward_expand(g,loss);ggml_set_loss(loss);ggml_build_backward_expand(r.ctx,g,nullptr);if(!support(backend,g,error)||!alloc(&r,backend,error))return false;
        gx=ggml_graph_get_grad(g,x);return gx?true:fail(error,"AR final norm gradient missing");
    }
};

inline bool Yue2AitkEndpoints::ar_final_norm(ggml_backend_t backend, const Yue2AitkModel & model,
                                             const float * hidden, const float * dy, int64_t tokens,
                                             Yue2AitkEndpointHost * out, std::string * error) {
    using namespace yue2_aitk_endpoints_detail; if(!out||!shape2(hidden,2048,tokens)||!shape2(dy,2048,tokens))return fail(error,"invalid AR final norm request");
    Yue2AitkArNormGraph n;if(!n.build(backend,model,tokens,error))return false;ggml_backend_tensor_set(n.x,hidden,0,ggml_nbytes(n.x));ggml_backend_tensor_set(n.up,dy,0,ggml_nbytes(n.up));if(!compute(backend,n.g,error))return false;
    out->values.resize(ggml_nelements(n.y));ggml_backend_tensor_get(n.y,out->values.data(),0,ggml_nbytes(n.y));out->dx.resize(ggml_nelements(n.gx));ggml_backend_tensor_get(n.gx,out->dx.data(),0,ggml_nbytes(n.gx));return true;
}

// Device-resident AR final norm around the head loss: the prediction rows are
// gathered on the device for the head kernel, and its F32 hidden gradient goes
// back as the norm's dy (BF16-rounded, times grad_scale, scattered to the
// prediction rows) without a host hop. The hidden states can arrive and dx
// can leave as device tensors too. Same graphs and rounding as the host path.
struct Yue2AitkArHeadRows {
    Yue2AitkArNormGraph adapted;
    yue2_aitk_endpoints_detail::Runtime r;
    ggml_tensor *pos=nullptr,*rows=nullptr,*base_rows=nullptr,*grad=nullptr; ggml_cgraph *gather=nullptr,*scatter=nullptr;
    // base_hidden null: no teacher, base_rows stays null.
    // A hidden state is either host data or a device tensor (a stack tape's
    // device_final), copied D2D into the norm input.
    struct Source { const std::vector<float> * host=nullptr; const ggml_tensor * device=nullptr; };
    bool init(ggml_backend_t backend,const Yue2AitkModel & model,Source hidden,const Source * base_hidden,
              int64_t tokens,const std::vector<size_t> & positions,float grad_scale,std::string * error) {
        using namespace yue2_aitk_endpoints_detail; const int64_t n=int64_t(positions.size());
        auto valid=[&](const Source & s){const size_t want=size_t(tokens)*2048;return s.host?s.host->size()==want:(s.device&&size_t(ggml_nelements(s.device))==want&&s.device->type==GGML_TYPE_F32);};
        if(tokens<=0||n<=0||!valid(hidden)||(base_hidden&&!valid(*base_hidden)))return fail(error,"invalid AR head rows request");
        std::vector<int32_t> index(positions.size());for(size_t i=0;i<positions.size();++i){if(positions[i]>=size_t(tokens))return fail(error,"invalid AR head rows request");index[i]=int32_t(positions[i]);}
        auto open=[&](Yue2AitkArNormGraph & g,const Source & h){if(!g.build(backend,model,tokens,error))return false;if(h.host)ggml_backend_tensor_set(g.x,h.host->data(),0,ggml_nbytes(g.x));else ggml_backend_tensor_copy(const_cast<ggml_tensor*>(h.device),g.x);ggml_backend_tensor_memset(g.up,0,0,ggml_nbytes(g.up));return compute(backend,g.g,error);};
        Yue2AitkArNormGraph base; // only its gathered rows outlive init
        if(!open(adapted,hidden)||(base_hidden&&!open(base,*base_hidden)))return false;
        if(!yue2_aitk_endpoints_detail::init(backend,&r,error))return false;
        pos=ggml_new_tensor_1d(r.ctx,GGML_TYPE_I32,n);grad=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,2048,n);if(!pos||!grad)return fail(error,"AR head rows allocation failed");
        // Gather from leaf aliases of the computed norm outputs: get_rows on y itself would pull the whole norm graph into gather.
        auto alias=[&](const ggml_tensor*y){ggml_tensor*t=ggml_new_tensor(r.ctx,y->type,GGML_MAX_DIMS,y->ne);if(t){t->buffer=y->buffer;t->data=y->data;}return t;};
        ggml_tensor*ya=alias(adapted.y),*yb=base_hidden?alias(base.y):nullptr;if(!ya||(base_hidden&&!yb))return fail(error,"AR head rows allocation failed");
        rows=ggml_get_rows(r.ctx,ya,pos);if(base_hidden)base_rows=ggml_get_rows(r.ctx,yb,pos);
        ggml_tensor*dy=yue2_aitk_graph::round(r.ctx,grad);if(grad_scale!=1.0f)dy=ggml_scale(r.ctx,dy,grad_scale);dy=ggml_set_rows(r.ctx,adapted.up,dy,pos);
        gather=ggml_new_graph_custom(r.ctx,16,false);scatter=ggml_new_graph_custom(r.ctx,16,false);if(!rows||(base_hidden&&!base_rows)||!dy||!gather||!scatter)return fail(error,"AR head rows graph allocation failed");
        ggml_build_forward_expand(gather,rows);if(base_rows)ggml_build_forward_expand(gather,base_rows);ggml_build_forward_expand(scatter,dy);
        if(!support(backend,gather,error)||!support(backend,scatter,error)||!alloc(&r,backend,error))return false;
        ggml_backend_tensor_set(pos,index.data(),0,ggml_nbytes(pos));
        return ggml_backend_graph_compute(backend,gather)==GGML_STATUS_SUCCESS?true:fail(error,"AR head rows gather failed");
    }
    // After the head loss has filled grad: scatter it into dy, run the norm backward, read dx
    // (or, with device_dx, copy it there D2D and leave dx empty).
    bool backward(ggml_backend_t backend,std::vector<float> * dx,std::string * error,ggml_tensor * device_dx=nullptr) {
        using namespace yue2_aitk_endpoints_detail;
        if(ggml_backend_graph_compute(backend,scatter)!=GGML_STATUS_SUCCESS)return fail(error,"AR head rows scatter failed");
        if(!compute(backend,adapted.g,error))return false;
        if(device_dx){if(ggml_nbytes(device_dx)!=ggml_nbytes(adapted.gx))return fail(error,"AR head rows dx shape mismatch");dx->clear();ggml_backend_tensor_copy(adapted.gx,device_dx);return true;}
        dx->resize(ggml_nelements(adapted.gx));ggml_backend_tensor_get(adapted.gx,dx->data(),0,ggml_nbytes(adapted.gx));return true;
    }
};
