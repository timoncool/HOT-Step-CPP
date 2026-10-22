#pragma once
#include "lm-optim.h"
#include "yue2-aitk-stack.h"
#include "yue2-aitk-endpoints.h"
#include "yue2-aitk-train-state.h"
#include "yue2-aitk-head-loss.h"
#include "yue2-aitk-optimizer.h"
#include "yue2-aitk-batch.h"
#include "yue2-aitk-cursor-loss.h"

#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

// One explicit, deterministic joint update. The runner supplies sampled noise,
// timestep and target so this seam can be compared without conflating RNGs.
// No prefix is retained across calls, and neither expert updates before both
// objectives have supplied all adapter gradients.
namespace yue2_aitk_joint {
// The frozen teacher's final hidden is a function of the AR input ids alone
// (no adapters, no prefix, a frozen base), so a repeated item reuses it
// instead of running the 28 base blocks again. Host memory up to
// YUE2_AITK_TEACHER_CACHE_MB (default 2048; 0 turns it off). Items are kept
// as they come until it is full and never evicted: epochs revisit items in
// a cycle, which evicting the oldest would turn into all misses.
struct TeacherCache {
    std::unordered_map<std::string, std::vector<float>> entries;
    size_t bytes = 0;
    static size_t limit() {
        static const size_t mb = [] { const char * v = std::getenv("YUE2_AITK_TEACHER_CACHE_MB"); return v && *v ? size_t(std::strtoull(v, nullptr, 10)) : size_t(2048); }();
        return mb << 20;
    }
    static std::string key(const std::vector<int32_t> & ids) {
        return std::string(reinterpret_cast<const char *>(ids.data()), ids.size() * sizeof(int32_t));
    }
    const std::vector<float> * find(const std::string & k) const {
        auto it = entries.find(k);
        return it == entries.end() ? nullptr : &it->second;
    }
    bool has_room(size_t floats) const { return bytes + floats * sizeof(float) <= limit(); }
    void put(const std::string & k, std::vector<float> && hidden) {
        bytes += hidden.size() * sizeof(float);
        entries.emplace(k, std::move(hidden));
    }
};
inline TeacherCache & teacher_cache() { static TeacherCache c; return c; }

struct Metrics { double ar_ce=0, ar_kl=0, nar_mse=0, gradient_norm=0, cursor_ce=0; size_t cursor_frames=0; int step=0; bool skipped=false; };
struct Input {
    const yue2_aitk::Batch * batch=nullptr;
    std::vector<float> noisy_latents, flow_target; // [crop frames,64]
    float timestep=0; // reference normalized BF16-valued timestep
    float cursor_weight=0;
    int64_t lyric_start=0, lyric_count=0;
    std::vector<std::pair<int32_t,int32_t>> cursor_frames;
    // Planner KL anchor weight (--kl-weight). 0.2 was the hardcoded value.
    float kl_weight=0.2f;
    // Native AdamW8bit step config (--lr / --weight-decay). Ignored on the
    // LmOptim path, whose schedule lives on lm->opt.base_lr.
    float adamw_lr=1e-4f;
    float adamw_weight_decay=1e-4f;
    // Spike guard (--spike-factor): a step whose pre-clip gradient norm is
    // above this skips its update entirely. 0 = off.
    double skip_above=0;
    // --ar-loss-weight: the planner CE's weight against the flow loss. Acts on
    // the CE part of the head gradient before the shared clip; KL keeps its
    // own weight. 1.0 = the AITK recipe, byte-identical.
    float ar_loss_weight=1.0f;
    // Adam betas for the native AdamW8bit path (the LmOptim path reads them
    // from LmOptim itself). Defaults are torch's, what every run before had.
    float beta1=.9f, beta2=.999f;
    // false with kl_weight 0: the frozen teacher forward is skipped and ar_kl
    // reads 0. The runner sets it false only when nothing reads the KL.
    bool measure_kl=true;
    // --grad-accum: micro-steps of one update. accumulate adds this song's
    // gradients to the slots instead of replacing them; finalize clips (after
    // multiplying by accum_scale = 1/N) and steps the optimizer. A plain step
    // is {false, true, 1.0}.
    bool accumulate=false;
    bool finalize=true;
    float accum_scale=1.0f;
};
using Progress=std::function<void(const char *)>;
// `optimizer` is the native CUDA AdamW8bit; `lm`/`osched` select the shared
// LmOptim path (prodigy/muon). Exactly one of the two is used per run.
inline bool run(ggml_backend_t backend, const Yue2AitkModel & model,
                Yue2AitkTrainState & state, yue2_aitk::Optimizer * optimizer,
                const Input & input, Metrics * metrics, std::string * error,
                const Progress & progress={}, LmOptim * lm=nullptr, ggml_backend_sched_t osched=nullptr) {
    using yue2_aitk_executor_detail::fail;
    if (lm) {
        if (!osched) return fail(error, "LmOptim path requires an optimizer scheduler");
    } else if (!optimizer) {
        return fail(error, "no optimizer is bound to the joint update");
    }
    constexpr size_t H=2048,C=64;
    if (!input.batch || !metrics || !state.initialized() || !std::isfinite(input.timestep) ||
        input.timestep<0 || input.timestep>1 || !std::isfinite(input.ar_loss_weight) || !(input.ar_loss_weight>0) ||
        !std::isfinite(input.accum_scale) || !(input.accum_scale>0) ||
        (input.accumulate && input.cursor_weight>0)) return fail(error,"invalid joint update input");
    const auto & batch=*input.batch;
    const size_t frames=batch.nar.semantic_tokens.size(), N=batch.ar.target_ids.size();
    if (!frames || batch.nar.ar.input_ids.size()+frames>24576 /* context */ || !N || input.noisy_latents.size()!=frames*C ||
        input.flow_target.size()!=frames*C || batch.ar.prediction_positions.size()!=N ||
        batch.ar.input_ids.empty() || batch.nar.ar.input_ids.empty() ||
        !yue2_aitk_executor_detail::finite_all(input.noisy_latents.data(),input.noisy_latents.size()) ||
        !yue2_aitk_executor_detail::finite_all(input.flow_target.data(),input.flow_target.size()))
        return fail(error,"invalid joint update sequence or flow tensors");
    for(size_t i=0;i<N;++i) if(batch.ar.prediction_positions[i]>=batch.ar.input_ids.size() ||
        batch.ar.target_ids[i]<0 || batch.ar.target_ids[i]>=yue2_aitk::kVocabSize)
        return fail(error,"invalid AR prediction position or target");
    const auto notify=[&](const char * stage){if(progress)progress(stage);};
    Metrics result;
    if(!input.accumulate) state.clear_cursor_gradient();
    // A frozen planner has no objective left: its forward, teacher, CE/KL and
    // backward are skipped outright. The NAR's conditioning below still runs
    // through the (held) planner adapters, exactly as at generation.
    const bool planner_frozen=state.planner_frozen();
    // Whole-song decoder: the NAR's conditioning sequence is the AR loss
    // sequence token for token, and no update happens between the two passes,
    // so the AR forward captures the prefix itself and the refresh is skipped
    // (~9% of a base-matched update). A crop or a frozen planner keeps it.
    const bool reuse_prefix=!planner_frozen && batch.nar.ar.input_ids==batch.ar.input_ids;
    Yue2AitkPrefixHost prefix;
    prefix.device_count=static_cast<int64_t>(frames)+2;  // capture the NAR's conditioning on the device
    // Stage outputs stay on the device: the AR head and the NAR final take the
    // stack's final hidden and hand dx back as device tensors
    // (YUE2_AITK_HANDOFF_HOST=1: through host vectors, as before).
    static const bool handoff_host=std::getenv("YUE2_AITK_HANDOFF_HOST")!=nullptr;
    if(!planner_frozen) {
    notify("AR adapted forward");
    Yue2AitkEndpointHost embeds;
    if(!Yue2AitkEndpoints::token_embedding(backend,model,batch.ar.input_ids.data(),batch.ar.input_ids.size(),&embeds,error)) return false;
    static const bool head_host=std::getenv("YUE2_AITK_HEAD_HOST")!=nullptr;
    const bool device_head=!head_host && !(input.cursor_weight>0 && !input.cursor_frames.empty());
    const bool device_handoff=device_head && !handoff_host;
    Yue2AitkStackTape ar;
    ar.keep_device_final=device_handoff;
    if(!yue2_aitk_stack::forward(backend,model,&state.ar_adapters(),false,embeds.values,
        batch.ar.input_ids.size(),nullptr,true,&ar,reuse_prefix?&prefix:nullptr,error)) return false;
    // The frozen teacher exists for the KL term. With no KL weight and nothing
    // reading the KL, its forward is a wasted pass: the adapted hidden stands
    // in as "base", so the kernel's (p - q) term is zero and ar_kl reads 0.
    const bool teacher=input.kl_weight>0 || input.measure_kl;
    // The head kernel's gradient is d(CE + kl_weight * KL). --ar-loss-weight w
    // wants d(w * CE + kl_weight * KL) = w * d(CE + (kl_weight / w) * KL): the
    // kernel runs at kl_weight / w and the hidden gradient is scaled by w. At
    // w = 1 both are exact identities.
    const float ar_w=input.ar_loss_weight;
    // Device head: final norm, row gather, head loss and the norm's dy stay on
    // the device. The cursor loss reads and writes host rows, so it keeps the
    // host path, as does YUE2_AITK_HEAD_HOST=1 (bit-identical rollback).
    std::vector<float> ar_dx;
    if(device_head) {
    Yue2AitkArHeadRows rows;
    {
        Yue2AitkStackTape base;
        base.keep_device_final=device_handoff;
        // A host tape (YUE2_AITK_HOST_TAPE=1) leaves device_final null and fills final_hidden.
        auto source=[](const Yue2AitkStackTape & t){Yue2AitkArHeadRows::Source s;if(t.device_final)s.device=t.device_final;else s.host=&t.final_hidden;return s;};
        Yue2AitkArHeadRows::Source base_source;
        if(teacher) {
            notify("AR frozen teacher forward");
            const bool cache=TeacherCache::limit()>0;
            const std::string key=cache?TeacherCache::key(batch.ar.input_ids):std::string();
            if(const std::vector<float> * hit=cache?teacher_cache().find(key):nullptr) {
                base_source.host=hit;
            } else {
                if(!yue2_aitk_stack::forward(backend,model,nullptr,false,embeds.values,ar.length,nullptr,false,&base,nullptr,error)) return false;
                base_source=source(base);
                const size_t floats=size_t(ar.length)*2048;
                if(cache && teacher_cache().has_room(floats)) {
                    std::vector<float> hidden;
                    if(base.device_final){hidden.resize(size_t(ggml_nelements(base.device_final)));ggml_backend_tensor_get(base.device_final,hidden.data(),0,ggml_nbytes(base.device_final));}
                    else hidden=base.final_hidden;
                    teacher_cache().put(key,std::move(hidden));
                }
            }
        }
        YUE2_PROF("head: AR final norm + row gather (device)");
        if(!rows.init(backend,model,source(ar),teacher?&base_source:nullptr,ar.length,batch.ar.prediction_positions,ar_w,error)) return false;
    }
    std::vector<uint32_t> targets(N);
    for(size_t i=0;i<N;++i) targets[i]=uint32_t(batch.ar.target_ids[i]);
    float ce=0,kl=0;
    yue2_aitk_head_loss::Request head;
    head.backend=backend;head.head=&model.lm_head();head.targets=targets.data();head.positions=N;head.hidden=H;
    head.adapted_rows=rows.rows;head.base_rows=teacher?rows.base_rows:rows.rows;head.adapted_grad_rows=rows.grad;
    head.kl_weight=input.kl_weight/ar_w;head.ce_sum=&ce;head.kl_sum=&kl;
    notify("AR CE and KL backward");
    { YUE2_PROF("head: CE/KL compute"); if(yue2_aitk_head_loss::compute(head,error)!=yue2_aitk_head_loss::Status::success) return false; }
    result.ar_ce=double(ce)/N;result.ar_kl=teacher?double(kl)/N:0.0;
    notify("AR transformer backward");
    { YUE2_PROF("head: dy scatter + AR final norm backward (device)"); if(!rows.backward(backend,&ar_dx,error,device_handoff&&ar.device_tape?ar.device_grad[0]:nullptr)) return false; }
    } else {
    // Final norm's small graph currently shares a forward/backward entry point.
    std::vector<float> zeros(ar.final_hidden.size(),0.0f);
    Yue2AitkEndpointHost adapted_norm;
    if(!Yue2AitkEndpoints::ar_final_norm(backend,model,ar.final_hidden.data(),zeros.data(),ar.length,&adapted_norm,error)) return false;
    std::vector<float> selected(N*H),selected_base(teacher?N*H:0);
    std::vector<uint32_t> targets(N);
    if(teacher) {
        notify("AR frozen teacher forward");
        Yue2AitkStackTape base;
        if(!yue2_aitk_stack::forward(backend,model,nullptr,false,embeds.values,ar.length,nullptr,false,&base,nullptr,error)) return false;
        Yue2AitkEndpointHost base_norm;
        if(!Yue2AitkEndpoints::ar_final_norm(backend,model,base.final_hidden.data(),zeros.data(),base.length,&base_norm,error)) return false;
        for(size_t i=0;i<N;++i) std::copy_n(base_norm.values.data()+batch.ar.prediction_positions[i]*H,H,selected_base.data()+i*H);
    }
    for(size_t i=0;i<N;++i) {
        const size_t row=batch.ar.prediction_positions[i];
        std::copy_n(adapted_norm.values.data()+row*H,H,selected.data()+i*H);
        targets[i]=uint32_t(batch.ar.target_ids[i]);
    }
    std::vector<uint16_t> selected_grad(N*H); float ce=0,kl=0;
    yue2_aitk_head_loss::Request head;
    head.backend=backend;head.head=&model.lm_head();head.adapted_hidden=selected.data();
    head.base_hidden=teacher?selected_base.data():selected.data();head.targets=targets.data();head.positions=N;
    head.hidden=H;head.kl_weight=input.kl_weight/input.ar_loss_weight;head.adapted_hidden_grad_bf16=selected_grad.data();head.ce_sum=&ce;head.kl_sum=&kl;
    notify("AR CE and KL backward");
    { YUE2_PROF("head: CE/KL compute"); if(yue2_aitk_head_loss::compute(head,error)!=yue2_aitk_head_loss::Status::success) return false; }
    result.ar_ce=double(ce)/N;result.ar_kl=teacher?double(kl)/N:0.0;
    notify("AR transformer backward");
    for(size_t i=0;i<N;++i) for(size_t d=0;d<H;++d)
        zeros[batch.ar.prediction_positions[i]*H+d]=ggml_bf16_to_fp32(ggml_bf16_t{selected_grad[i*H+d]})*ar_w;
    if(input.cursor_weight>0 && !input.cursor_frames.empty()) {
        size_t audio_index=0;
        while(audio_index<N && (batch.ar.target_ids[audio_index]<yue2_aitk::kCodecOffset ||
              batch.ar.target_ids[audio_index]>=yue2_aitk::kCodecOffset+32768)) ++audio_index;
        if(audio_index==N || input.cursor_frames.size()>N-audio_index) return fail(error,"cursor audio positions are missing");
        for(size_t frame=0;frame<input.cursor_frames.size();++frame)
            if(batch.ar.target_ids[audio_index+frame]<yue2_aitk::kCodecOffset ||
               batch.ar.target_ids[audio_index+frame]>=yue2_aitk::kCodecOffset+32768)
                return fail(error,"cursor target is not a semantic audio frame");
        yue2_aitk_cursor_loss::Request cursor;
        cursor.backend=backend; cursor.head=state.cursor_head(); cursor.head_gradient=state.cursor_gradient();
        cursor.hidden=&adapted_norm.values; cursor.hidden_gradient=&zeros; cursor.sequence=ar.length;
        cursor.lyric_start=input.lyric_start; cursor.lyric_count=input.lyric_count;
        cursor.audio_start=batch.ar.prediction_positions[audio_index]; cursor.frame_tokens=&input.cursor_frames;
        cursor.weight=input.cursor_weight;
        if(!yue2_aitk_cursor_loss::compute(cursor,&result.cursor_ce,error))return false;
        result.cursor_frames=input.cursor_frames.size();
    }
    if(!Yue2AitkEndpoints::ar_final_norm(backend,model,ar.final_hidden.data(),zeros.data(),ar.length,&adapted_norm,error)) return false;
    ar_dx=std::move(adapted_norm.dx);
    }
    if(!yue2_aitk_stack::backward(backend,model,state.ar_adapters(),ar,nullptr,std::move(ar_dx),
        [&](size_t layer,const Yue2AitkBlockBackwardHost & g,std::string * why){return state.upload_gradients(false,int(layer),g,why,input.accumulate);},error,
        [&](size_t layer){return state.gradient_tensors(false,int(layer));},input.accumulate)) return false;
    }
    // Recompute with the current adapted AR. This cache is detached host data.
    // It is local to this step and cannot survive the optimizer update below.
    if(!reuse_prefix) {
    notify("Refresh detached AR conditioning");
    Yue2AitkEndpointHost condition;
    if(!Yue2AitkEndpoints::token_embedding(backend,model,batch.nar.ar.input_ids.data(),batch.nar.ar.input_ids.size(),&condition,error)) return false;
    Yue2AitkStackTape condition_tape;
    if(!yue2_aitk_stack::forward(backend,model,&state.ar_adapters(),false,condition.values,
        batch.nar.ar.input_ids.size(),nullptr,false,&condition_tape,&prefix,error)) return false;
    }
    notify("NAR forward");
    Yue2AitkEndpointHost frontend;
    if(!Yue2AitkEndpoints::nar_frontend(backend,model,input.noisy_latents.data(),frames,input.timestep,&frontend,error)) return false;
    Yue2AitkStackTape nar;
    nar.keep_device_final=!handoff_host;
    if(!yue2_aitk_stack::forward(backend,model,&state.nar_adapters(),true,frontend.values,frames+2,&prefix,true,&nar,nullptr,error)) return false;
    std::vector<float> flow_grad((frames+2)*C,0.0f); Yue2AitkEndpointHost prediction;
    if(!Yue2AitkEndpoints::nar_final(backend,model,nar.final_hidden.data(),flow_grad.data(),nar.length,&prediction,error,nar.device_final)) return false;
    for(size_t i=0;i<frames*C;++i) {
        const float diff=prediction.values[C+i]-input.flow_target[i];
        result.nar_mse+=double(diff)*diff;
        flow_grad[C+i]=2.0f*diff/float(frames*C);
    }
    result.nar_mse/=double(frames*C);
    notify("NAR flow backward");
    if(!Yue2AitkEndpoints::nar_final(backend,model,nar.final_hidden.data(),flow_grad.data(),nar.length,&prediction,error,
        nar.device_final,nar.device_final?nar.device_grad[0]:nullptr)) return false;
    if(!yue2_aitk_stack::backward(backend,model,state.nar_adapters(),nar,&prefix,std::move(prediction.dx),
        [&](size_t layer,const Yue2AitkBlockBackwardHost & g,std::string * why){return state.upload_gradients(true,int(layer),g,why,input.accumulate);},error,
        [&](size_t layer){return state.gradient_tensors(true,int(layer));},input.accumulate)) return false;
    if(!std::isfinite(result.ar_ce)||!std::isfinite(result.ar_kl)||!std::isfinite(result.nar_mse)) return fail(error,"nonfinite joint loss");
    if(!input.finalize) {
        // A micro-step of --grad-accum: the gradients stay in the slots for the
        // next song; no clip, no update, the step counter does not move.
        ggml_backend_synchronize(backend);
        result.step=lm?lm->opt_step:optimizer->step();
        *metrics=result;
        return true;
    }
    notify("Joint clipping and optimizer update");
    if(planner_frozen) state.zero_planner_gradients();
    { YUE2_PROF("opt: clip (device; YUE2_AITK_CLIP_HOST=1: D2H + host norm + H2D)"); if(!state.clip_gradients(1.0f,&result.gradient_norm,error,input.accum_scale)) return false; }
    ggml_backend_synchronize(backend);
    if (input.skip_above>0 && result.gradient_norm>input.skip_above) {
        // A spike: no parameter or optimizer-moment changes. Clipping alone is
        // not enough — Adam-style updates are normalized, so a clipped spike
        // still takes a full-size step in its (bad) direction. The step
        // counters advance so resume records stay aligned with the run.
        result.skipped=true;
        if (lm) { lm->opt_step++; result.step=lm->opt_step; }
        else { try { optimizer->skip_once(); } catch(const std::exception & e){return fail(error,e.what());} result.step=optimizer->step(); }
        *metrics=result;
        return true;
    }
    if (lm) {
        // After the clip the device slots equal the clipped host values; copy
        // them into LmOptim's accumulators (own buffer), step on its scheduler.
        { YUE2_PROF("opt: fill LmOptim accumulators (D2D)"); if (!state.fill_device_gradients(lm->acc, error)) return false; }
        LmStepStats step_stats{};
        { YUE2_PROF("opt: lm_optim_step"); if (!lm_optim_step(lm, osched, &step_stats)) return fail(error, "optimizer step failed"); }
        result.step = lm->opt_step;
    } else {
        yue2_aitk::StepConfig cfg;
        cfg.learning_rate=input.adamw_lr; cfg.weight_decay=input.adamw_weight_decay;
        cfg.beta1=input.beta1; cfg.beta2=input.beta2;
        try { YUE2_PROF("opt: step_once (Prodigy/8-bit AdamW)"); optimizer->step_once(cfg); }
        catch(const std::exception & e){return fail(error,e.what());}
        result.step=optimizer->step();
    }
    if(planner_frozen) { state.hold_planner(); ggml_backend_synchronize(backend); }
    // DoRA: A/B just moved (or were held back), so the detached norms follow.
    // A skipped update returns above with the norms still exact.
    if(state.is_dora()) { YUE2_PROF("opt: DoRA norm refresh"); if(!state.refresh_dora(model,error)) return false; }
    *metrics=result;
    return true;
}

// Decoder drift probe (--nar-drift). The decoder's flow prediction on a fixed
// input, conditioned by the BASE planner so the number measures the decoder
// alone, however far the planner has moved. `nar` null = the base decoder.
inline bool nar_probe_prefix(ggml_backend_t backend, const Yue2AitkModel & model, const yue2_aitk::Batch & batch,
                             Yue2AitkPrefixHost * prefix, std::string * error, const Yue2AitkExpertAdapters * ar = nullptr) {
    Yue2AitkEndpointHost condition;
    if(!Yue2AitkEndpoints::token_embedding(backend,model,batch.nar.ar.input_ids.data(),batch.nar.ar.input_ids.size(),&condition,error)) return false;
    Yue2AitkStackTape tape;
    return yue2_aitk_stack::forward(backend,model,ar,false,condition.values,batch.nar.ar.input_ids.size(),nullptr,false,&tape,prefix,error);
}
inline bool nar_probe_predict(ggml_backend_t backend, const Yue2AitkModel & model, const Yue2AitkExpertAdapters * nar,
                              const Yue2AitkPrefixHost & prefix, const std::vector<float> & noisy, float timestep,
                              std::vector<float> * out, std::string * error) {
    constexpr size_t C=64; const size_t frames=noisy.size()/C;
    Yue2AitkEndpointHost frontend;
    if(!Yue2AitkEndpoints::nar_frontend(backend,model,noisy.data(),frames,timestep,&frontend,error)) return false;
    Yue2AitkStackTape tape;
    if(!yue2_aitk_stack::forward(backend,model,nar,true,frontend.values,frames+2,&prefix,false,&tape,nullptr,error)) return false;
    std::vector<float> zero((frames+2)*C,0.0f); Yue2AitkEndpointHost prediction;
    if(!Yue2AitkEndpoints::nar_final(backend,model,tape.final_hidden.data(),zero.data(),tape.length,&prediction,error)) return false;
    out->assign(prediction.values.begin()+C,prediction.values.begin()+C+frames*C);
    return true;
}

// Planner CE on one batch (--eval-base-loss; --eval-adapted passes the AR
// adapters): the forward with `ar` null = the frozen base (no adapter, no tape) and the head kernel at KL weight 0. The kernel always
// writes a hidden gradient; it lands in a scratch buffer and is dropped.
inline bool base_ar_ce(ggml_backend_t backend, const Yue2AitkModel & model, const yue2_aitk::Batch & batch,
                       double * ce_out, size_t * tokens, std::string * error,
                       const Yue2AitkExpertAdapters * ar = nullptr) {
    using yue2_aitk_executor_detail::fail;
    constexpr size_t H=2048;
    const size_t N=batch.ar.target_ids.size(), len=batch.ar.input_ids.size();
    if(!N || !len || batch.ar.prediction_positions.size()!=N) return fail(error,"invalid AR evaluation batch");
    for(size_t i=0;i<N;++i) if(batch.ar.prediction_positions[i]>=len || batch.ar.target_ids[i]<0 || batch.ar.target_ids[i]>=yue2_aitk::kVocabSize)
        return fail(error,"invalid AR prediction position or target");
    Yue2AitkEndpointHost embeds;
    if(!Yue2AitkEndpoints::token_embedding(backend,model,batch.ar.input_ids.data(),len,&embeds,error)) return false;
    Yue2AitkStackTape tape;
    if(!yue2_aitk_stack::forward(backend,model,ar,false,embeds.values,len,nullptr,false,&tape,nullptr,error)) return false;
    std::vector<float> zeros(tape.final_hidden.size(),0.0f);
    Yue2AitkEndpointHost norm;
    if(!Yue2AitkEndpoints::ar_final_norm(backend,model,tape.final_hidden.data(),zeros.data(),tape.length,&norm,error)) return false;
    std::vector<float> selected(N*H); std::vector<uint32_t> targets(N);
    for(size_t i=0;i<N;++i) { std::copy_n(norm.values.data()+batch.ar.prediction_positions[i]*H,H,selected.data()+i*H); targets[i]=uint32_t(batch.ar.target_ids[i]); }
    std::vector<uint16_t> scratch(N*H); float ce=0,kl=0;
    yue2_aitk_head_loss::Request head;
    head.backend=backend;head.head=&model.lm_head();head.adapted_hidden=selected.data();head.base_hidden=selected.data();
    head.targets=targets.data();head.positions=N;head.hidden=H;head.kl_weight=0.0f;
    head.adapted_hidden_grad_bf16=scratch.data();head.ce_sum=&ce;head.kl_sum=&kl;
    if(yue2_aitk_head_loss::compute(head,error)!=yue2_aitk_head_loss::Status::success) return false;
    *ce_out=double(ce)/N; if(tokens)*tokens=N;
    return std::isfinite(*ce_out) || fail(error,"nonfinite base planner CE");
}

// Reference overload for call sites that bind the native AdamW8bit optimizer
// by name; forwards to the dispatching overload above.
inline bool run(ggml_backend_t backend, const Yue2AitkModel & model,
                Yue2AitkTrainState & state, yue2_aitk::Optimizer & optimizer,
                const Input & input, Metrics * metrics, std::string * error,
                const Progress & progress={}) {
    return run(backend, model, state, &optimizer, input, metrics, error, progress);
}
} // namespace yue2_aitk_joint
