import json,os,collections
D="/Users/jeonghamin/Desktop/naccl/model/KTM-LLM/stage5_final"
TAGS=[('qwen2_5_7b','Qwen2.5-7B'),('exaone3_5_7_8b','EXAONE-3.5-7.8B'),('solar_10_7b','SOLAR-10.7B'),
      ('mistral_7b','Mistral-7B'),('huatuogpt-o1-7b','HuatuoGPT-o1-7B'),('biomistral-7b','BioMistral-7B'),
      ('medllama2-7b','MedLLaMA2-7B'),('openbiollm-8b','OpenBioLLM-8B'),('medgemma-4b','MedGemma-4B')]
YEARS=['2022','2023','2024','2025']
inj={}
for y in YEARS:
    for qid,v in json.load(open(f"{D}/gate/rag_injections_{y}.json")).items(): inj[qid]=bool(v)
def load(tag,kind):
    o={}
    for y in YEARS:
        p=f"{D}/predictions/RES_{tag}_{y}_{kind}.json"
        if not os.path.exists(p): return None
        for r in json.load(open(p))['results']: o[r['id']]=bool(r['is_correct'])
    return o
print(f"{'모델':17s} | {'주입O(29%)':>22s} | {'주입X(71%) = 노이즈':>24s}")
print(f"{'':17s} | {'plain':>6s} {'RAG':>6s} {'Δ':>6s} | {'plain':>6s} {'RAG':>6s} {'Δ':>6s}")
print('-'*70)
agg=collections.defaultdict(lambda:[0,0,0,0])  # inj: p,n ; noninj: p,n  (correct counts)
tot={'i':[0,0,0],'n':[0,0,0]}
for tag,name in TAGS:
    b=load(tag,'base'); r=load(tag,'rag')
    ci=[0,0,0]; cn=[0,0,0]   # [plain_correct, rag_correct, N]
    for q in b:
        if q not in r: continue
        t=ci if inj.get(q) else cn
        t[0]+=b[q]; t[1]+=r[q]; t[2]+=1
    for k,t in (('i',ci),('n',cn)):
        for j in range(3): tot[k][j]+=t[j]
    f=lambda t,j:100*t[j]/t[2] if t[2] else 0
    print(f"{name:17s} | {f(ci,0):5.1f}% {f(ci,1):5.1f}% {f(ci,1)-f(ci,0):+5.1f}p | {f(cn,0):5.1f}% {f(cn,1):5.1f}% {f(cn,1)-f(cn,0):+5.1f}p")
print('-'*70)
f=lambda t,j:100*t[j]/t[2]
print(f"{'전체 합산':17s} | {f(tot['i'],0):5.1f}% {f(tot['i'],1):5.1f}% {f(tot['i'],1)-f(tot['i'],0):+5.1f}p | {f(tot['n'],0):5.1f}% {f(tot['n'],1):5.1f}% {f(tot['n'],1)-f(tot['n'],0):+5.1f}p")
print(f"\n주입O 문항수(9모델합)={tot['i'][2]}, 주입X={tot['n'][2]}")
