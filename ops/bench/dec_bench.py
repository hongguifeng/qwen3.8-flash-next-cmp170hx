import json, time, urllib.request, statistics, re, sys
import _cfg
M=_cfg.api_url()
KEYS=["spec_decode_num_draft_tokens_total","spec_decode_num_accepted_tokens_total",
      "time_per_output_token_seconds_sum","time_per_output_token_seconds_count"]
def met():
    t=urllib.request.urlopen(M+"/metrics",timeout=10).read().decode(); d={}
    for k in KEYS:
        m=re.search(r'^vllm:%s\{[^}]*\}\s+([\d.eE+-]+)'%k,t,re.M); d[k]=float(m.group(1)) if m else -1.0
    return d
def disk():
    o={}
    for l in open("/proc/diskstats"):
        f=l.split()
        if re.match(r'^(sd[a-z]|vd[a-z]|nvme\d+n\d+)$',f[2]): o[f[2]]=int(f[7])
    return o
def run(n,tag,prompt="请详细说明 vLLM 中 PagedAttention 的工作原理。"):
    body=json.dumps({"model":_cfg.model_name(),"prompt":prompt,"max_tokens":n,"temperature":0.6,
        "ignore_eos":True,"stream":True,"stream_options":{"include_usage":True}}).encode()
    req=urllib.request.Request(M+"/v1/completions",body,{"Content-Type":"application/json"})
    m0=met(); d0=disk(); t0=time.time(); ts=[]; ct=None
    for line in urllib.request.urlopen(req,timeout=600):
        s=line.decode(errors="ignore")
        if s.startswith("data: ") and "[DONE]" not in s:
            try:
                j=json.loads(s[6:])
                if j.get("usage"): ct=j["usage"]["completion_tokens"]
                elif j.get("choices"): ts.append(time.time()-t0)
            except Exception: pass
    m1=met(); d1=disk()
    d=[(ts[i]-ts[i-1])*1000 for i in range(1,len(ts))]
    dr=m1[KEYS[0]]-m0[KEYS[0]]; ac=m1[KEYS[1]]-m0[KEYS[1]]
    dc=m1[KEYS[3]]-m0[KEYS[3]]; ds=m1[KEYS[2]]-m0[KEYS[2]]
    tp=(ds/dc) if dc>0 else float('nan')
    dec=ts[-1]-ts[0]; rd=(sum(d1.values())-sum(d0.values()))/2048
    print(f"  [{tag}] 步数={len(ts)} completion={ct} 接受率={(ac/dr*100 if dr else 0):.1f}%")
    tptxt = f"   引擎TPOT={tp*1000:.2f}ms (n={dc:.0f})" if (tp == tp and tp) else ""
    print(f"       TTFT={ts[0]*1000:.0f}ms  纯解码={dec:.2f}s ⇒ {((ct-1)/dec if dec else 0):.1f} tok/s{tptxt}")
    print(f"       步间 p50={statistics.median(d):.1f} p95={sorted(d)[int(len(d)*.95)-1]:.1f} max={max(d):.1f} ms   解码期磁盘读={rd:.1f} MiB")
run(64,"warm1"); run(256,"run2"); run(256,"run3")
run(256,"run4-短prompt",prompt="你好，介绍一下你自己。")
